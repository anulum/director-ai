# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — live web-search refresh for stale temporal claims

"""Refresh stale temporal claims against live web search.

:func:`~director_ai.core.scoring.temporal_freshness.score_temporal_freshness`
flags claims that *may* rely on outdated knowledge (a named office-holder, a
statistic, a record) but cannot tell whether they are still true — it only knows
they are the kind of thing that goes stale. This module closes that gap: it takes
the flagged claims, queries a live web-search provider, and reports whether
current sources still echo the claim.

The reliable payload is the *retrieved evidence* itself; the verdict is decided
by whichever engine is available:

* with an injected NLI :class:`ContradictionEngine`, the fresh evidence is the
  premise and the claim the hypothesis — a high ``P(contradiction)`` means
  current sources **refute** the claim (``contradicted``), otherwise
  ``supported``. This is the dependable path and reuses the same scorer as the
  streaming contradiction halt;
* without it, a lexical heuristic checks whether the claim's asserted value
  (the incumbent of an office, the asserted number) appears in the top result —
  ``supported`` if so, else ``drift_suspected``. This is a coarse triage only: a
  former office-holder's name persisting in current coverage can mask drift, and
  a sparse top result can over-flag it, so treat ``drift_suspected`` as "verify",
  not "false".

Two diagnostic signals are always reported: ``coverage`` (asserted-value
containment in the top result) and ``topical_overlap`` (Jaccard of claim and best
result, computed through the Rust ``rust_word_overlap`` kernel with a bit-exact
pure-Python fallback). A search returning nothing yields ``no_fresh_evidence``;
a claim with no extractable value and no NLI engine yields ``review``.

The HTTP layer is injected through
the :class:`~director_ai.core.citation_grounding.fetch.HttpGetter` protocol, so
the query construction, the DuckDuckGo HTML parsing, and the agreement scoring
are deterministic and fully tested with a stub — no network is touched in tests.
The default provider uses ``httpx`` lazily.

This is an advisory enrichment, not a fact oracle: it surfaces current evidence
and flags disagreement; it does not perform natural-language inference, and a
``drift_suspected`` verdict means "verify", not "false".
"""

from __future__ import annotations

import html as _html
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import parse_qs, unquote, urlsplit

from ..citation_grounding.fetch import HttpGetter
from ..text_overlap import word_overlap
from .temporal_freshness import FreshnessClaim, FreshnessResult

__all__ = [
    "ClaimRefresh",
    "DuckDuckGoSearchProvider",
    "RefreshReport",
    "SearchHit",
    "TemporalRefresher",
    "WebSearchProvider",
]

_DDG_HTML_URL = "https://html.duckduckgo.com/html/"
_RESULT_ANCHOR_RE = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S
)
_RESULT_SNIPPET_RE = re.compile(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
# Tokens dropped when turning a claim span into a search query.
_QUERY_NOISE = frozenset({"is", "was", "the", "of", "a", "an", "as", "at", "in", "on"})


def _lexical_overlap(text_a: str, text_b: str) -> float:
    """Lexical Jaccard overlap in ``[0, 1]``.

    Delegates to the shared measured-fast-path helper (pure Python below a large
    -input threshold, Rust above it). See :mod:`director_ai.core.text_overlap`.
    """
    return word_overlap(text_a, text_b, logger_name=__name__)


def _content_tokens(text: str) -> set[str]:
    """Lower-cased content tokens with surrounding punctuation trimmed."""
    return {
        tok
        for raw in text.lower().split()
        if (tok := raw.strip(".,?!:;\"'()")) and tok not in _QUERY_NOISE
    }


def _numeric_tokens(text: str) -> set[str]:
    """Lower-cased tokens carrying a digit (asserted statistics, years, records)."""
    return {
        tok
        for raw in text.lower().split()
        if (tok := raw.strip(".,?!:;\"'()%")) and any(ch.isdigit() for ch in tok)
    }


def _coverage(value_tokens: set[str], evidence: str) -> float:
    """Fraction of asserted *value_tokens* present in *evidence* (containment)."""
    if not value_tokens:
        return 0.0
    e = _content_tokens(evidence) | _numeric_tokens(evidence)
    return len(value_tokens & e) / len(value_tokens)


def _strip_markup(raw: str) -> str:
    """Drop HTML tags, unescape entities, and collapse whitespace."""
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", raw))).strip()


def _resolve_ddg_href(href: str) -> str:
    """Turn a DuckDuckGo redirect href into the underlying result URL.

    DuckDuckGo wraps every result URL as ``//duckduckgo.com/l/?uddg=<encoded>``;
    the real URL is the ``uddg`` query parameter. A bare href is returned as-is,
    with a leading ``//`` promoted to ``https://``.
    """
    clean = _html.unescape(href).strip()
    split = urlsplit(clean if "//" not in clean[:2] else "https:" + clean)
    # Exact host match (strip any user:pass@/:port) so a look-alike host such as
    # ``duckduckgo.com.evil.test`` cannot be mistaken for DuckDuckGo (CodeQL
    # py/incomplete-url-substring-sanitization).
    host = split.netloc.rsplit("@", 1)[-1].split(":", 1)[0].lower()
    if (host == "duckduckgo.com" or host.endswith(".duckduckgo.com")) and (
        split.path.startswith("/l/")
    ):
        target = parse_qs(split.query).get("uddg", [])
        if target:
            return unquote(target[0])
    if clean.startswith("//"):
        return "https:" + clean
    return clean


@dataclass(frozen=True)
class SearchHit:
    """One web-search result."""

    title: str
    snippet: str
    url: str
    rank: int


class WebSearchProvider(Protocol):
    """Return ranked web-search hits for a query."""

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        """Return up to ``max_results`` ranked web-search hits for ``query``."""
        ...


@dataclass(frozen=True)
class ClaimRefresh:
    """The live-evidence verdict for one flagged claim.

    ``verdict`` is one of ``supported``, ``contradicted`` (NLI found fresh
    evidence that refutes the claim), ``drift_suspected`` (lexical heuristic:
    the asserted value is absent from the top result), ``review`` (evidence
    retrieved but not adjudicated) or ``no_fresh_evidence``. ``verdict_source``
    records which engine decided: ``nli``, ``lexical`` or ``none``.
    """

    claim: FreshnessClaim
    query: str
    verdict: str
    verdict_source: str
    coverage: float
    topical_overlap: float
    contradiction: float
    evidence: tuple[SearchHit, ...]


@dataclass
class RefreshReport:
    """Aggregate live-refresh result over a freshness analysis."""

    refreshes: list[ClaimRefresh] = field(default_factory=list)
    checked: int = 0
    skipped: int = 0

    @property
    def drift_suspected(self) -> list[ClaimRefresh]:
        """Return refreshes whose verdict flags suspected drift."""
        return [r for r in self.refreshes if r.verdict == "drift_suspected"]

    @property
    def supported(self) -> list[ClaimRefresh]:
        """Return refreshes whose verdict is supported."""
        return [r for r in self.refreshes if r.verdict == "supported"]

    @property
    def contradicted(self) -> list[ClaimRefresh]:
        """Return refreshes whose verdict is contradicted."""
        return [r for r in self.refreshes if r.verdict == "contradicted"]


class ContradictionEngine(Protocol):
    """An NLI contradiction scorer (e.g. ``ContradictionScorer``).

    Duck-typed against :class:`director_ai.core.scoring.contradiction.ContradictionScorer`
    so the refresher need not import the heavy ``transformers`` stack.
    """

    def contradiction(self, premise: str, hypothesis: str) -> float: ...

    @property
    def threshold(self) -> float: ...


def _ddg_hits(body: bytes, max_results: int) -> list[SearchHit]:
    """Parse a DuckDuckGo HTML response into ranked :class:`SearchHit`s."""
    text = body.decode("utf-8", "ignore")
    titles = _RESULT_ANCHOR_RE.findall(text)
    snippets = [_strip_markup(s) for s in _RESULT_SNIPPET_RE.findall(text)]
    hits: list[SearchHit] = []
    for rank, (href, raw_title) in enumerate(titles):
        snippet = snippets[rank] if rank < len(snippets) else ""
        hits.append(
            SearchHit(
                title=_strip_markup(raw_title),
                snippet=snippet,
                url=_resolve_ddg_href(href),
                rank=rank,
            )
        )
        if len(hits) >= max_results:
            break
    return hits


class DuckDuckGoSearchProvider:
    """:class:`WebSearchProvider` backed by the DuckDuckGo HTML endpoint.

    No API key is required. The HTTP client is injected through
    :class:`HttpGetter` (defaulting to a lazy ``httpx`` getter), so the request
    and the HTML parsing are testable without a network.
    """

    def __init__(
        self, *, http: HttpGetter | None = None, timeout: float = 10.0
    ) -> None:
        self._http = http if http is not None else _HttpxGetter(timeout)

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        """Return up to ``max_results`` hits scraped from DuckDuckGo HTML."""
        url = f"{_DDG_HTML_URL}?q={_quote(query)}"
        try:
            status, body, _ = self._http.get(url, headers={"User-Agent": _USER_AGENT})
        except Exception:  # noqa: BLE001 - network failure → no evidence
            return []
        if status != 200 or not body:
            return []
        return _ddg_hits(body, max_results)


class TemporalRefresher:
    """Refresh stale temporal claims against a live web-search provider.

    Parameters
    ----------
    provider:
        The :class:`WebSearchProvider`; defaults to
        :class:`DuckDuckGoSearchProvider`.
    nli:
        Optional :class:`ContradictionEngine` (an NLI contradiction scorer). When
        supplied, the verdict is adjudicated by natural-language inference against
        the fresh evidence — the reliable path. When omitted, the verdict falls
        back to the lexical coverage heuristic, which is a coarse triage signal
        only (a former office-holder's name persisting in current coverage can
        mask genuine drift, and a sparse top result can over-flag it).
    staleness_threshold:
        Only claims with ``staleness_risk`` at or above this are checked.
    support_threshold:
        Minimum asserted-value coverage in the top result to call a claim
        ``supported`` on the lexical path.
    max_results:
        Web-search hits fetched per claim.
    """

    def __init__(
        self,
        provider: WebSearchProvider | None = None,
        *,
        nli: ContradictionEngine | None = None,
        staleness_threshold: float = 0.5,
        support_threshold: float = 0.35,
        max_results: int = 3,
    ) -> None:
        if not 0.0 <= staleness_threshold <= 1.0:
            raise ValueError("staleness_threshold must be in [0, 1]")
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError("support_threshold must be in [0, 1]")
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        self._provider = (
            provider if provider is not None else DuckDuckGoSearchProvider()
        )
        self._nli = nli
        self._staleness_threshold = staleness_threshold
        self._support_threshold = support_threshold
        self._max_results = max_results

    @staticmethod
    def _claim_context(claim: FreshnessClaim, source_text: str) -> str:
        """Return the claim span, widened to include the asserted value if possible.

        The freshness regexes capture a position claim's *office* ("the CEO of
        Twitter is") but stop before the *incumbent* ("Jack Dorsey"). When the
        original response is available, the span is located and extended forward
        to the end of its sentence so the asserted value enters the query and the
        coverage check — without which drift on a changed office-holder cannot be
        seen.
        """
        if not source_text:
            return claim.text
        idx = source_text.find(claim.text)
        if idx < 0:
            return claim.text
        window = source_text[idx : idx + len(claim.text) + 60]
        return re.split(r"(?<=[.!?])\s", window.strip(), maxsplit=1)[0]

    @staticmethod
    def _subject_and_value(claim: FreshnessClaim, context: str) -> tuple[str, set[str]]:
        """Split a claim into an unbiased search *subject* and its asserted *value*.

        The query must name the topic without leaking the claimed value, or the
        search engine simply returns pages echoing that value and every claim
        looks confirmed. For a position the subject is the office ("CEO of
        Twitter") and the value is the incumbent that follows it ("Jack Dorsey");
        for a statistic or record the subject is the metric ("population of
        France") and the value is the asserted number ("67 million").
        """
        if claim.claim_type == "position":
            subject = re.sub(
                r"\s*\b(?:is|was)\b\s*$", "", claim.text.strip(), flags=re.IGNORECASE
            )
            tail = context[len(claim.text) :] if context.startswith(claim.text) else ""
            value = _content_tokens(tail)
        else:
            subject_text = re.sub(r"\b[\d][\d,.%]*\b", " ", context)
            subject = subject_text
            value = _numeric_tokens(context)
        query_tokens = [
            t for t in subject.split() if t.lower().strip(".,?!") not in _QUERY_NOISE
        ]
        query = " ".join(query_tokens).strip() or subject.strip() or claim.text.strip()
        return query, value

    def _verdict(
        self, context: str, value_tokens: set[str], hits: Sequence[SearchHit]
    ) -> tuple[str, str, float, float, float]:
        """Return ``(verdict, source, coverage, topical_overlap, contradiction)``."""
        if not hits:
            return "no_fresh_evidence", "none", 0.0, 0.0, 0.0
        evidence_texts = [f"{h.title} {h.snippet}" for h in hits]
        topical = max(_lexical_overlap(context, ev) for ev in evidence_texts)
        coverage = _coverage(value_tokens, evidence_texts[0]) if value_tokens else 0.0

        if self._nli is not None:
            # Reliable path: does any fresh result refute the claim? Premise is
            # the evidence, hypothesis is the claim.
            contradiction = max(
                self._nli.contradiction(ev, context) for ev in evidence_texts
            )
            verdict = (
                "contradicted" if contradiction >= self._nli.threshold else "supported"
            )
            return verdict, "nli", round(coverage, 4), topical, round(contradiction, 4)

        if not value_tokens:
            return "review", "none", 0.0, topical, 0.0
        # Lexical heuristic: the asserted value must appear in the single
        # highest-ranked result (the current authority). Pooling all hits lets a
        # historical mention mask drift; this is triage, not adjudication.
        verdict = (
            "supported" if coverage >= self._support_threshold else "drift_suspected"
        )
        return verdict, "lexical", coverage, topical, 0.0

    def refresh(
        self, result: FreshnessResult, *, source_text: str = ""
    ) -> RefreshReport:
        """Refresh every sufficiently-stale claim in *result*.

        Pass ``source_text`` (the original response) so position claim spans can
        be widened to their asserted incumbent for sharper drift detection.
        """
        report = RefreshReport()
        for claim in result.claims:
            if claim.staleness_risk < self._staleness_threshold:
                report.skipped += 1
                continue
            report.checked += 1
            context = self._claim_context(claim, source_text)
            query, value_tokens = self._subject_and_value(claim, context)
            hits = self._provider.search(query, max_results=self._max_results)
            verdict, source, coverage, topical, contradiction = self._verdict(
                context, value_tokens, hits
            )
            report.refreshes.append(
                ClaimRefresh(
                    claim=claim,
                    query=query,
                    verdict=verdict,
                    verdict_source=source,
                    coverage=round(coverage, 4),
                    topical_overlap=round(topical, 4),
                    contradiction=contradiction,
                    evidence=tuple(hits),
                )
            )
        return report

    def refresh_response(
        self,
        text: str,
        *,
        source_timestamp: float | None = None,
        max_age_days: float = 180,
        domain: str = "",
    ) -> RefreshReport:
        """Score *text* for temporal freshness, then live-refresh its stale claims."""
        from .temporal_freshness import score_temporal_freshness

        result = score_temporal_freshness(
            text,
            source_timestamp=source_timestamp,
            max_age_days=max_age_days,
            domain=domain,
        )
        return self.refresh(result, source_text=text)

    def refresh_claims(self, claims: Iterable[FreshnessClaim]) -> RefreshReport:
        """Refresh an explicit iterable of claims (bypassing the score step)."""
        return self.refresh(FreshnessResult(claims=list(claims)))


_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) director-ai-temporal-refresh/1.0"


def _quote(query: str) -> str:
    from urllib.parse import quote_plus

    return quote_plus(query)


class _HttpxGetter:
    """Default :class:`HttpGetter` backed by ``httpx`` (lazy import)."""

    def __init__(self, timeout: float) -> None:
        self._timeout = timeout

    def get(  # pragma: no cover -- requires network
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> tuple[int, bytes, str]:
        import httpx

        response = httpx.get(
            url,
            timeout=self._timeout,
            headers=dict(headers or {}),
            follow_redirects=True,
        )
        return (
            response.status_code,
            response.content,
            response.headers.get("content-type", ""),
        )
