# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ContradictionExplainer tests

"""Tests for the counterfactual contradiction explainer.

Covers threshold filtering, strongest-first ordering, the stated rationale and
preserved chunk index, empty-text and empty-claim handling, unattributed
sources, excerpt clipping, and scorer/threshold validation."""

from __future__ import annotations

import pytest

from director_ai.core.causal_verifier import (
    ContradictionExplainer,
    ContradictionExplanation,
    ContradictionReport,
)
from director_ai.core.types import EvidenceChunk

_CLAIM = "Refunds are available within 30 days."


def _scorer(scores: dict[str, float]):
    """Return a scorer that maps a passage's text to a fixed score."""
    return lambda passage, _claim: scores.get(passage, 0.0)


def _chunk(text: str, *, source: str = "kb", distance: float = 0.1) -> EvidenceChunk:
    return EvidenceChunk(text=text, distance=distance, source=source)


class TestExplain:
    def test_finds_contradicting_passage(self):
        explainer = ContradictionExplainer(
            scorer=_scorer({"Refunds are never available.": 0.92})
        )
        report = explainer.explain(
            _CLAIM, [_chunk("Refunds are never available.", source="policy.md")]
        )
        assert report.has_contradiction
        best = report.best
        assert best.chunk_source == "policy.md"
        assert best.score == pytest.approx(0.92)
        assert "contradicts" in best.rationale
        assert "policy.md" in best.rationale
        assert "0.92" in best.rationale

    def test_strongest_first(self):
        explainer = ContradictionExplainer(
            scorer=_scorer({"mild": 0.6, "severe": 0.95})
        )
        report = explainer.explain(
            _CLAIM, [_chunk("mild", source="a"), _chunk("severe", source="b")]
        )
        assert [e.chunk_source for e in report.contradictions] == ["b", "a"]
        assert report.best.chunk_source == "b"

    def test_below_threshold_excluded(self):
        explainer = ContradictionExplainer(scorer=_scorer({"mild": 0.4}), threshold=0.5)
        report = explainer.explain(_CLAIM, [_chunk("mild")])
        assert not report.has_contradiction
        assert report.best is None

    def test_no_contradiction_when_supportive(self):
        explainer = ContradictionExplainer(scorer=lambda p, c: 0.0)
        report = explainer.explain(_CLAIM, [_chunk("Refunds within 30 days.")])
        assert report.contradictions == ()

    def test_empty_text_chunk_skipped(self):
        explainer = ContradictionExplainer(scorer=lambda p, c: 1.0)
        report = explainer.explain(_CLAIM, [_chunk("   "), _chunk("real")])
        assert len(report.contradictions) == 1
        assert report.best.chunk_index == 1

    def test_chunk_index_preserved(self):
        explainer = ContradictionExplainer(scorer=_scorer({"hit": 0.8}))
        report = explainer.explain(
            _CLAIM, [_chunk("miss"), _chunk("miss2"), _chunk("hit")]
        )
        assert report.best.chunk_index == 2

    def test_unattributed_source(self):
        explainer = ContradictionExplainer(scorer=lambda p, c: 0.9)
        report = explainer.explain(_CLAIM, [_chunk("contra", source="")])
        assert report.best.chunk_source == ""
        assert "an unattributed source" in report.best.rationale

    def test_long_text_excerpt_clipped(self):
        long_text = "x " * 200
        explainer = ContradictionExplainer(scorer=lambda p, c: 0.9)
        report = explainer.explain(_CLAIM, [_chunk(long_text)])
        assert report.best.chunk_excerpt.endswith("…")
        assert len(report.best.chunk_excerpt) <= 160

    def test_empty_claim_rejected(self):
        explainer = ContradictionExplainer(scorer=lambda p, c: 0.9)
        with pytest.raises(ValueError, match="claim must be a non-empty string"):
            explainer.explain("   ", [_chunk("x")])

    def test_no_chunks(self):
        explainer = ContradictionExplainer(scorer=lambda p, c: 0.9)
        report = explainer.explain(_CLAIM, [])
        assert isinstance(report, ContradictionReport)
        assert not report.has_contradiction


class TestValidation:
    def test_scorer_must_be_callable(self):
        with pytest.raises(TypeError, match="scorer must be callable"):
            ContradictionExplainer(scorer="not-callable")  # type: ignore[arg-type]

    def test_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="threshold"):
            ContradictionExplainer(scorer=lambda p, c: 0.5, threshold=1.5)

    def test_scorer_out_of_range_rejected(self):
        explainer = ContradictionExplainer(scorer=lambda p, c: 2.0)
        with pytest.raises(ValueError, match="contradiction score"):
            explainer.explain(_CLAIM, [_chunk("x")])

    def test_explanation_negative_index_rejected(self):
        with pytest.raises(ValueError, match="chunk_index"):
            ContradictionExplanation(
                claim="c",
                chunk_index=-1,
                chunk_source="s",
                chunk_excerpt="e",
                score=0.5,
                rationale="r",
            )

    def test_explanation_score_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="score"):
            ContradictionExplanation(
                claim="c",
                chunk_index=0,
                chunk_source="s",
                chunk_excerpt="e",
                score=1.5,
                rationale="r",
            )
