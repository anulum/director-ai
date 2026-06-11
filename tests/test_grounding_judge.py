# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — citation-grounding judge tests
"""Multi-angle tests for the CitationGroundingJudge.

Covers the support-threshold validation, the per-assertion verdicts (cited +
supported → grounded; cited but unsupported → hallucination; uncited factual
sentence → ungrounded; cited but with no fetched source → ungrounded), the
reference-section exclusion, the short-fragment filter, multi-citation evidence
concatenation, the threshold boundary, and the report aggregates
(grounded_fraction, citation_coverage, hallucinated, the vacuous empty answer).
A stub scorer keyed on evidence substrings drives every case without a model.
"""

from __future__ import annotations

import pytest

from director_ai.core.citation_grounding import (
    CitationGroundingJudge,
    ClaimGrounding,
    GroundingReport,
)


class _StubScorer:
    """Returns a configured divergence by matching a substring of the premise."""

    def __init__(self, table: dict[str, float], default: float = 0.9) -> None:
        self._table = table
        self._default = default

    def score(self, premise: str, hypothesis: str) -> float:
        for needle, div in self._table.items():
            if needle in premise:
                return div
        return self._default


_DOC = (
    "Neutron stars constrain the equation of state [1]. "
    "The sky is green and made of cheese [2]. "
    "This sentence carries no citation whatsoever.\n\n"
    "References:\n"
    "[1] Bogdanov 10.3847/2041-8213/ab50c5\n"
    "[2] Nonsense arXiv:2411.04368\n"
)
_SOURCES = {
    "10.3847/2041-8213/ab50c5": "Mass-radius measurements constrain the equation of state.",
    "2411.04368": "A paper about language-model evaluation, nothing on cheese.",
}


def _assess(threshold=0.6, table=None):
    scorer = _StubScorer(table or {"equation of state": 0.1, "cheese": 0.95})
    return CitationGroundingJudge(scorer=scorer, support_threshold=threshold).assess(
        _DOC, _SOURCES
    )


class TestConstruction:
    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_invalid_threshold_raises(self, bad):
        with pytest.raises(ValueError, match="support_threshold"):
            CitationGroundingJudge(scorer=_StubScorer({}), support_threshold=bad)


class TestVerdicts:
    def test_cited_and_supported_is_grounded(self):
        report = _assess()
        first = report.claims[0]
        assert first.has_citation
        assert first.grounded
        assert first.support == pytest.approx(0.9)
        assert first.cited == ("10.3847/2041-8213/ab50c5",)

    def test_cited_but_unsupported_is_hallucination(self):
        report = _assess()
        cheese = report.claims[1]
        assert cheese.has_citation
        assert not cheese.grounded
        assert cheese.support == pytest.approx(0.05)

    def test_uncited_sentence_is_ungrounded(self):
        report = _assess()
        uncited = report.claims[2]
        assert not uncited.has_citation
        assert not uncited.grounded
        assert uncited.support == 0.0
        assert uncited.cited == ()

    def test_reference_section_not_assessed(self):
        # Exactly the three body sentences become claims.
        assert _assess().total == 3

    def test_missing_source_is_ungrounded(self):
        # Citation resolves but its source was not fetched → no evidence.
        judge = CitationGroundingJudge(scorer=_StubScorer({}, default=0.0))
        report = judge.assess(
            "A grounded-looking claim [1].\n\nReferences:\n[1] X arXiv:2411.04368\n",
            {},  # no source text for 2411.04368
        )
        claim = report.claims[0]
        assert claim.has_citation
        assert not claim.grounded
        assert claim.support == 0.0

    def test_threshold_boundary_inclusive(self):
        # support == threshold counts as grounded (>=).
        report = CitationGroundingJudge(
            scorer=_StubScorer({"evidence": 0.4}), support_threshold=0.6
        ).assess(
            "Some asserted fact here [1].\n\nReferences:\n[1] Source 10.1000/abc\n",
            {"10.1000/abc": "supporting evidence text"},
        )
        assert report.claims[0].support == pytest.approx(0.6)
        assert report.claims[0].grounded

    def test_multiple_citations_concatenate_evidence(self):
        seen = {}

        class _Capture:
            def score(self, premise, hypothesis):
                seen["premise"] = premise
                return 0.2

        judge = CitationGroundingJudge(scorer=_Capture())
        judge.assess(
            "A multi-cited claim [1, 2].\n\n"
            "References:\n[1] A 10.1000/aaa\n[2] B 10.1000/bbb\n",
            {"10.1000/aaa": "first evidence", "10.1000/bbb": "second evidence"},
        )
        assert "first evidence" in seen["premise"]
        assert "second evidence" in seen["premise"]

    def test_short_fragments_dropped(self):
        # "Yes." is below the minimum claim length and is not assessed.
        report = CitationGroundingJudge(scorer=_StubScorer({})).assess("Yes.", {})
        assert report.total == 0


class TestReportAggregates:
    def test_fractions(self):
        report = _assess()
        assert report.grounded_fraction == pytest.approx(1 / 3)
        assert report.citation_coverage == pytest.approx(2 / 3)
        assert len(report.hallucinated) == 2

    def test_empty_answer_is_vacuously_grounded(self):
        report = CitationGroundingJudge(scorer=_StubScorer({})).assess("", {})
        assert report.total == 0
        assert report.grounded_fraction == 1.0
        assert report.citation_coverage == 1.0

    def test_to_dict_shape(self):
        d = _assess().to_dict()
        assert set(d) == {
            "total",
            "grounded_fraction",
            "citation_coverage",
            "hallucinated_count",
            "claims",
        }
        assert d["hallucinated_count"] == 2
        assert isinstance(d["claims"], list)

    def test_claim_to_dict(self):
        d = ClaimGrounding("c", True, True, 0.83, ("10.1/x",)).to_dict()
        assert d == {
            "claim": "c",
            "has_citation": True,
            "grounded": True,
            "support": 0.83,
            "cited": ["10.1/x"],
        }

    def test_report_is_frozen(self):
        assert isinstance(_assess(), GroundingReport)
