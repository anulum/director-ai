# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — live temporal-refresh tests

"""Parser, provider, verdict-engine, polyglot-parity and guard-wiring coverage
for the live web-search temporal-claim refresher. No network is touched."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from director_ai.core.scoring import temporal_refresh as tr
from director_ai.core.scoring.temporal_freshness import FreshnessClaim, FreshnessResult
from director_ai.core.scoring.temporal_refresh import (
    ClaimRefresh,
    DuckDuckGoSearchProvider,
    RefreshReport,
    SearchHit,
    TemporalRefresher,
)
from director_ai.guard import ProductionGuard

# A faithful fragment of a DuckDuckGo HTML response (two results), matching the
# live ``result__a`` / ``result__snippet`` markup and the ``/l/?uddg=`` redirect.
_DDG_HTML = b"""
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSam_Altman&amp;rut=abc">Sam Altman - Wikipedia</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSam_Altman&amp;rut=abc">Samuel Harris Altman is the <b>CEO</b> <b>of</b> <b>OpenAI</b>.</a>
</div>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.forbes.com%2Fprofile%2Fsam-altman%2F&amp;rut=def">Sam Altman - Forbes</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.forbes.com%2Fprofile%2Fsam-altman%2F&amp;rut=def">Sam Altman is the CEO of OpenAI and an investor.</a>
</div>
"""


class _StubGetter:
    """An :class:`HttpGetter` returning a canned response, or raising."""

    def __init__(self, *, status=200, body=_DDG_HTML, raise_exc=False):
        self._status = status
        self._body = body
        self._raise = raise_exc
        self.last_url = ""

    def get(self, url, *, headers: Mapping[str, str] | None = None):
        self.last_url = url
        if self._raise:
            raise RuntimeError("network down")
        return self._status, self._body, "text/html"


class _StubProvider:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, *, max_results):
        return list(self._hits[:max_results])


class _StubNLI:
    def __init__(self, contradiction):
        self._c = contradiction

    def contradiction(self, premise, hypothesis):
        return self._c

    @property
    def threshold(self):
        return 0.2


def _position_claim(text="The CEO of Twitter is", risk=0.8):
    return FreshnessClaim(text=text, claim_type="position", staleness_risk=risk, reason="r")


# --------------------------------------------------------------------------- #
# lexical / token helpers — both backends                                      #
# --------------------------------------------------------------------------- #


def test_lexical_overlap_rust_and_python_parity(monkeypatch):
    rust = tr._lexical_overlap("a b c", "b c d")
    monkeypatch.setattr(tr, "_RUST_REFRESH", False)
    monkeypatch.setattr(tr, "rust_word_overlap", None)
    py = tr._lexical_overlap("a b c", "b c d")
    assert rust == pytest.approx(py) == pytest.approx(0.5)


def test_lexical_overlap_python_empty(monkeypatch):
    monkeypatch.setattr(tr, "_RUST_REFRESH", False)
    monkeypatch.setattr(tr, "rust_word_overlap", None)
    assert tr._lexical_overlap("", "x") == 0.0


def test_content_and_numeric_tokens():
    assert tr._content_tokens("The CEO of Twitter!") == {"ceo", "twitter"}
    assert tr._numeric_tokens("population is 67 million in 2026") == {"67", "2026"}


def test_coverage_containment_and_empty():
    assert tr._coverage(set(), "anything") == 0.0
    assert tr._coverage({"sam", "altman"}, "Sam Altman is CEO") == pytest.approx(1.0)
    assert tr._coverage({"jack", "dorsey"}, "Linda Yaccarino is CEO") == 0.0


def test_strip_markup():
    assert tr._strip_markup("a <b>bold</b>  &amp; clean") == "a bold & clean"


# --------------------------------------------------------------------------- #
# DDG href resolution & result parsing                                         #
# --------------------------------------------------------------------------- #


def test_resolve_ddg_redirect_href():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FX&amp;rut=z"
    assert tr._resolve_ddg_href(href) == "https://en.wikipedia.org/wiki/X"


def test_resolve_bare_and_plain_hrefs():
    assert tr._resolve_ddg_href("//example.com/page") == "https://example.com/page"
    assert tr._resolve_ddg_href("https://example.com/x") == "https://example.com/x"


def test_resolve_ddg_redirect_without_uddg_param():
    assert tr._resolve_ddg_href("//duckduckgo.com/l/?rut=z") == "https://duckduckgo.com/l/?rut=z"


def test_ddg_hits_parsing_and_cap():
    hits = tr._ddg_hits(_DDG_HTML, max_results=5)
    assert len(hits) == 2
    assert hits[0].title == "Sam Altman - Wikipedia"
    assert hits[0].url == "https://en.wikipedia.org/wiki/Sam_Altman"
    assert "CEO of OpenAI" in hits[0].snippet
    assert tr._ddg_hits(_DDG_HTML, max_results=1)[0].rank == 0
    assert len(tr._ddg_hits(_DDG_HTML, max_results=1)) == 1


def test_ddg_hits_fewer_snippets_than_titles():
    html = b'<a class="result__a" href="//x">Title only</a>'
    hits = tr._ddg_hits(html, max_results=3)
    assert hits[0].title == "Title only" and hits[0].snippet == ""


# --------------------------------------------------------------------------- #
# DuckDuckGoSearchProvider                                                     #
# --------------------------------------------------------------------------- #


def test_provider_parses_results_and_builds_query_url():
    getter = _StubGetter()
    provider = DuckDuckGoSearchProvider(http=getter)
    hits = provider.search("CEO of OpenAI", max_results=2)
    assert len(hits) == 2
    assert "q=CEO+of+OpenAI" in getter.last_url


def test_provider_non_200_and_empty_body_and_error_return_empty():
    assert DuckDuckGoSearchProvider(http=_StubGetter(status=503)).search("q", max_results=3) == []
    assert DuckDuckGoSearchProvider(http=_StubGetter(body=b"")).search("q", max_results=3) == []
    assert DuckDuckGoSearchProvider(http=_StubGetter(raise_exc=True)).search("q", max_results=3) == []


# --------------------------------------------------------------------------- #
# TemporalRefresher — validation & internals                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"staleness_threshold": 1.5},
        {"support_threshold": -0.1},
        {"max_results": 0},
    ],
)
def test_refresher_rejects_bad_config(kwargs):
    with pytest.raises(ValueError):
        TemporalRefresher(**kwargs)


def test_claim_context_widens_with_source_text():
    claim = _position_claim()
    assert TemporalRefresher._claim_context(claim, "") == "The CEO of Twitter is"
    assert TemporalRefresher._claim_context(claim, "Unrelated text") == "The CEO of Twitter is"
    ctx = TemporalRefresher._claim_context(claim, "The CEO of Twitter is Jack Dorsey. Next.")
    assert ctx == "The CEO of Twitter is Jack Dorsey."


def test_subject_and_value_position_and_statistic():
    refresher = TemporalRefresher(provider=_StubProvider([]))
    q_pos, v_pos = refresher._subject_and_value(
        _position_claim(), "The CEO of Twitter is Jack Dorsey."
    )
    assert q_pos == "CEO Twitter" and v_pos == {"jack", "dorsey"}

    stat = FreshnessClaim(
        text="population of France is 67 million",
        claim_type="statistic",
        staleness_risk=0.6,
        reason="r",
    )
    q_stat, v_stat = refresher._subject_and_value(stat, stat.text)
    assert "population" in q_stat and "67" in v_stat


def test_subject_query_falls_back_when_all_noise():
    # A claim whose content is entirely stop words still yields a non-empty query.
    refresher = TemporalRefresher(provider=_StubProvider([]))
    noise = FreshnessClaim(text="the of a an", claim_type="record", staleness_risk=0.6, reason="r")
    query, value = refresher._subject_and_value(noise, "the of a an")
    assert query == "the of a an" and value == set()


# --------------------------------------------------------------------------- #
# Verdict paths                                                                #
# --------------------------------------------------------------------------- #


def _hits_yaccarino():
    return [SearchHit("Linda Yaccarino - Wikipedia", "Linda Yaccarino is CEO of X", "u", 0)]


def _hits_altman():
    return [SearchHit("Sam Altman - Wikipedia", "Sam Altman is CEO of OpenAI", "u", 0)]


def test_verdict_no_hits_is_no_fresh_evidence():
    refresher = TemporalRefresher(provider=_StubProvider([]))
    report = refresher.refresh(FreshnessResult(claims=[_position_claim()]))
    assert report.refreshes[0].verdict == "no_fresh_evidence"
    assert report.refreshes[0].verdict_source == "none"


def test_verdict_nli_contradicted_and_supported():
    src = "The CEO of Twitter is Jack Dorsey."
    fr = FreshnessResult(claims=[_position_claim()])

    high = TemporalRefresher(provider=_StubProvider(_hits_yaccarino()), nli=_StubNLI(0.9))
    r_high = high.refresh(fr, source_text=src).refreshes[0]
    assert r_high.verdict == "contradicted" and r_high.verdict_source == "nli"
    assert r_high.contradiction == pytest.approx(0.9)

    low = TemporalRefresher(provider=_StubProvider(_hits_altman()), nli=_StubNLI(0.01))
    r_low = low.refresh(fr, source_text=src).refreshes[0]
    assert r_low.verdict == "supported" and r_low.verdict_source == "nli"


def test_verdict_lexical_drift_and_supported():
    fr = FreshnessResult(claims=[_position_claim()])
    drift = TemporalRefresher(provider=_StubProvider(_hits_yaccarino()))
    r_drift = drift.refresh(fr, source_text="The CEO of Twitter is Jack Dorsey.").refreshes[0]
    assert r_drift.verdict == "drift_suspected" and r_drift.verdict_source == "lexical"

    ok = TemporalRefresher(provider=_StubProvider(_hits_altman()))
    r_ok = ok.refresh(
        FreshnessResult(claims=[_position_claim("The CEO of OpenAI is")]),
        source_text="The CEO of OpenAI is Sam Altman.",
    ).refreshes[0]
    assert r_ok.verdict == "supported" and r_ok.coverage == pytest.approx(1.0)


def test_verdict_review_when_no_value_and_no_nli():
    # No source_text -> no asserted value extractable for a position claim.
    refresher = TemporalRefresher(provider=_StubProvider(_hits_altman()))
    r = refresher.refresh(FreshnessResult(claims=[_position_claim()])).refreshes[0]
    assert r.verdict == "review" and r.verdict_source == "none"
    assert r.topical_overlap >= 0.0


# --------------------------------------------------------------------------- #
# refresh / refresh_response / refresh_claims & report                         #
# --------------------------------------------------------------------------- #


def test_refresh_skips_claims_below_staleness_threshold():
    fresh = FreshnessClaim(text="x is", claim_type="position", staleness_risk=0.1, reason="r")
    refresher = TemporalRefresher(provider=_StubProvider(_hits_altman()))
    report = refresher.refresh(FreshnessResult(claims=[fresh]))
    assert report.checked == 0 and report.skipped == 1 and report.refreshes == []


def test_refresh_response_scores_then_refreshes():
    refresher = TemporalRefresher(provider=_StubProvider(_hits_yaccarino()))
    report = refresher.refresh_response("The CEO of Twitter is Jack Dorsey.")
    assert report.checked >= 1
    assert all(isinstance(r, ClaimRefresh) for r in report.refreshes)


def test_refresh_claims_bypasses_scoring():
    refresher = TemporalRefresher(provider=_StubProvider(_hits_altman()))
    report = refresher.refresh_claims([_position_claim()])
    assert isinstance(report, RefreshReport) and report.checked == 1


def test_report_buckets():
    report = RefreshReport(
        refreshes=[
            ClaimRefresh(_position_claim(), "q", "supported", "nli", 1.0, 0.2, 0.0, ()),
            ClaimRefresh(_position_claim(), "q", "drift_suspected", "lexical", 0.0, 0.1, 0.0, ()),
            ClaimRefresh(_position_claim(), "q", "contradicted", "nli", 0.0, 0.1, 0.9, ()),
        ]
    )
    assert len(report.supported) == 1
    assert len(report.drift_suspected) == 1
    assert len(report.contradicted) == 1


# --------------------------------------------------------------------------- #
# ProductionGuard wiring                                                       #
# --------------------------------------------------------------------------- #


def test_guard_temporal_refresher_persists_and_refreshes(monkeypatch):
    guard = ProductionGuard()
    # Swap the lazily-built refresher's provider for a stub (no network).
    refresher = guard.temporal_refresher
    assert guard.temporal_refresher is refresher  # persisted
    monkeypatch.setattr(refresher, "_provider", _StubProvider(_hits_yaccarino()))
    report = guard.refresh_temporal("The CEO of Twitter is Jack Dorsey.")
    assert isinstance(report, RefreshReport)
    assert report.checked >= 1
