from __future__ import annotations

from director_ai.core.types import (
    CoherenceScore,
    EvidenceChunk,
    ReviewResult,
    ScoringEvidence,
    _clamp,
)


class TestClamp:
    def test_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_below_lo(self):
        assert _clamp(-0.1) == 0.0

    def test_above_hi(self):
        assert _clamp(1.5) == 1.0

    def test_at_boundaries(self):
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_nan_returns_lo(self):
        assert _clamp(float("nan")) == 0.0

    def test_positive_inf_returns_hi(self):
        assert _clamp(float("inf")) == 1.0

    def test_negative_inf_returns_lo(self):
        assert _clamp(float("-inf")) == 0.0

    def test_custom_range(self):
        assert _clamp(5.0, lo=2.0, hi=8.0) == 5.0
        assert _clamp(1.0, lo=2.0, hi=8.0) == 2.0
        assert _clamp(9.0, lo=2.0, hi=8.0) == 8.0

    def test_nan_custom_range(self):
        assert _clamp(float("nan"), lo=3.0, hi=7.0) == 3.0


class TestEvidenceChunk:
    def test_fields(self):
        chunk = EvidenceChunk(text="hello", distance=0.1, source="test")
        assert chunk.text == "hello"
        assert chunk.distance == 0.1
        assert chunk.source == "test"

    def test_default_source(self):
        chunk = EvidenceChunk(text="x", distance=0.0)
        assert chunk.source == ""


class TestScoringEvidence:
    def test_fields(self):
        chunks = [EvidenceChunk(text="a", distance=0.1)]
        ev = ScoringEvidence(
            chunks=chunks,
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.8,
            chunk_scores=[0.7, 0.8],
        )
        assert ev.nli_score == 0.8
        assert len(ev.chunk_scores) == 2

    def test_default_chunk_scores_none(self):
        ev = ScoringEvidence(
            chunks=[], nli_premise="", nli_hypothesis="", nli_score=0.5
        )
        assert ev.chunk_scores is None


class TestCoherenceScore:
    def test_fields(self):
        cs = CoherenceScore(score=0.9, approved=True, h_logical=0.05, h_factual=0.1)
        assert cs.score == 0.9
        assert cs.evidence is None
        assert cs.warning is False


class TestReviewResult:
    def test_fields(self):
        rr = ReviewResult(
            output="ok", coherence=None, halted=False, candidates_evaluated=1
        )
        assert rr.output == "ok"
        assert rr.fallback_used is False
