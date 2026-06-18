# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation source fetcher

"""Fetch the text behind a citation so it can be scored as evidence.

A resolved :class:`~director_ai.core.citation_grounding.citations.Citation`
names a source; this module turns that identifier into the source's text:

* **arXiv** ids → the arXiv API (``export.arxiv.org/api/query``), returning the
  paper title and abstract from the Atom ``<entry>``;
* **DOIs** → the Crossref REST API (``api.crossref.org/works/<doi>``), returning
  the title and the JATS-tagged abstract (markup stripped);
* **URLs** → fetched directly and parsed by
  :func:`~director_ai.core.retrieval.doc_parser.parse` (PDF / HTML / text).

The HTTP layer is injected through the :class:`HttpGetter` protocol, so the URL
construction, the Crossref/arXiv response parsing, the markup stripping, and the
content-type dispatch are all deterministic and fully tested with a stub — no
network is touched in tests. The default getter uses ``httpx`` lazily.

Author-year and unresolved numeric citations name no retrievable endpoint and
return an unsuccessful :class:`FetchedSource`.
"""

from __future__ import annotations

import html as _html
import json as _json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .citations import Citation, CitationKind

__all__ = ["FetchedSource", "HttpGetter", "SourceFetcher"]

_CROSSREF_URL = "https://api.crossref.org/works/{doi}"
_ARXIV_URL = "https://export.arxiv.org/api/query?id_list={arxiv_id}"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ARXIV_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.S)
_ARXIV_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)
_ARXIV_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)


class HttpGetter(Protocol):
    """Fetch a URL, returning ``(status_code, body, content_type)``."""

    def get(
        self, url: str, *, headers: Mapping[str, str] | None = None
    ) -> tuple[int, bytes, str]:
        """Fetch ``url`` with optional headers."""
        ...  # pragma: no cover


@dataclass(frozen=True)
class FetchedSource:
    """The text retrieved for a citation (or the reason it could not be)."""

    identifier: str
    kind: CitationKind
    ok: bool
    title: str = ""
    text: str = ""
    url: str = ""
    error: str = ""


def _strip_markup(raw: str) -> str:
    """Drop XML/HTML tags, collapse whitespace, and unescape entities."""
    return _WS_RE.sub(" ", _html.unescape(_TAG_RE.sub(" ", raw))).strip()


def _parse_crossref(body: bytes) -> tuple[str, str]:
    """Return ``(title, abstract)`` from a Crossref works response."""
    message = _json.loads(body).get("message", {})
    titles = message.get("title") or []
    title = _strip_markup(titles[0]) if titles else ""
    abstract = _strip_markup(message.get("abstract", ""))
    return title, abstract


def _parse_arxiv(body: bytes) -> tuple[str, str]:
    """Return ``(title, abstract)`` from an arXiv API Atom feed.

    The feed's own ``<title>`` describes the query; the paper title and abstract
    live inside the ``<entry>`` element, which is read in isolation.
    """
    entry = _ARXIV_ENTRY_RE.search(body.decode("utf-8", "ignore"))
    if entry is None:
        return "", ""
    block = entry.group(1)
    title_m = _ARXIV_TITLE_RE.search(block)
    summary_m = _ARXIV_SUMMARY_RE.search(block)
    title = _strip_markup(title_m.group(1)) if title_m else ""
    abstract = _strip_markup(summary_m.group(1)) if summary_m else ""
    return title, abstract


def _content_filename(content_type: str) -> str:
    """Map a content-type to a filename ``doc_parser.parse`` dispatches on."""
    ct = content_type.lower()
    if "pdf" in ct:
        return "source.pdf"
    if "html" in ct:
        return "source.html"
    return "source.txt"


class SourceFetcher:
    """Retrieve the evidence text behind resolved citations.

    Parameters
    ----------
    http : HttpGetter | None
        Injected HTTP client; defaults to a lazy ``httpx`` getter.
    timeout : float
        Per-request timeout for the default getter (seconds).
    mailto : str
        Contact e-mail sent to Crossref's polite pool via the User-Agent.
    """

    def __init__(
        self,
        *,
        http: HttpGetter | None = None,
        timeout: float = 10.0,
        mailto: str = "",
    ) -> None:
        self._http = http if http is not None else _HttpxGetter(timeout)
        self._mailto = mailto

    @property
    def _user_agent(self) -> str:
        base = "director-ai-citation-grounding/1.0"
        return f"{base} (mailto:{self._mailto})" if self._mailto else base

    def fetch(self, citation: Citation) -> FetchedSource:
        """Fetch the source text for one citation."""
        if citation.kind is CitationKind.ARXIV:
            return self._fetch_arxiv(citation)
        if citation.kind is CitationKind.DOI:
            return self._fetch_doi(citation)
        if citation.kind is CitationKind.URL:
            return self._fetch_url(citation)
        return FetchedSource(
            citation.identifier,
            citation.kind,
            ok=False,
            error=f"{citation.kind.value} citations name no retrievable source",
        )

    def fetch_all(self, citations: Iterable[Citation]) -> dict[str, str]:
        """Fetch every citation and return an ``{identifier: text}`` map.

        Only successful fetches with non-empty text are included, so the result
        slots straight into :meth:`CitationGroundingJudge.assess`. Duplicate
        identifiers are fetched once.
        """
        sources: dict[str, str] = {}
        seen: set[str] = set()
        for citation in citations:
            if citation.identifier in seen:
                continue
            seen.add(citation.identifier)
            result = self.fetch(citation)
            if result.ok and result.text:
                sources[citation.identifier] = result.text
        return sources

    # -- per-kind retrieval -------------------------------------------------

    def _fetch_arxiv(self, citation: Citation) -> FetchedSource:
        url = _ARXIV_URL.format(arxiv_id=citation.identifier)
        status, body, _ = self._get(url)
        if status != 200:
            return self._failure(citation, url, f"arxiv HTTP {status}")
        title, abstract = _parse_arxiv(body)
        if not abstract:
            return self._failure(citation, url, "arxiv abstract not found")
        return FetchedSource(
            citation.identifier, citation.kind, True, title, abstract, url
        )

    def _fetch_doi(self, citation: Citation) -> FetchedSource:
        url = _CROSSREF_URL.format(doi=citation.identifier)
        status, body, _ = self._get(url)
        if status != 200:
            return self._failure(citation, url, f"crossref HTTP {status}")
        title, abstract = _parse_crossref(body)
        if not abstract:
            return self._failure(citation, url, "crossref abstract not found")
        return FetchedSource(
            citation.identifier, citation.kind, True, title, abstract, url
        )

    def _fetch_url(self, citation: Citation) -> FetchedSource:
        url = citation.identifier
        status, body, content_type = self._get(url)
        if status != 200:
            return self._failure(citation, url, f"url HTTP {status}")
        from ..retrieval.doc_parser import parse

        try:
            text = parse(body, _content_filename(content_type)).strip()
        except Exception as exc:  # noqa: BLE001 - any parser failure → unfetched
            return self._failure(citation, url, f"parse failed: {exc}")
        if not text:
            return self._failure(citation, url, "empty document")
        return FetchedSource(citation.identifier, citation.kind, True, "", text, url)

    def _get(self, url: str) -> tuple[int, bytes, str]:
        try:
            return self._http.get(url, headers={"User-Agent": self._user_agent})
        except Exception:  # noqa: BLE001 - network failure surfaces as status 0
            return 0, b"", ""

    @staticmethod
    def _failure(citation: Citation, url: str, error: str) -> FetchedSource:
        return FetchedSource(
            citation.identifier, citation.kind, ok=False, url=url, error=error
        )


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
