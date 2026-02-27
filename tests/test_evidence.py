# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Evidence, Fallback, and Soft Zone Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

from director_ai.core import (
    CoherenceAgent,
    CoherenceScore,
    CoherenceScorer,
    EvidenceChunk,
    GroundTruthStore,
    ReviewResult,
    ScoringEvidence,
)
from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore


class TestEvidenceChunk:
    def test_fields(self):
        c = EvidenceChunk(text="sky is blue", distance=0.1, source="builtin")
        assert c.text == "sky is blue"
        assert c.distance == 0.1
        assert c.source == "builtin"

    def test_default_source(self):
        c = EvidenceChunk(text="x", distance=0.0)
        assert c.source == ""


class TestScoringEvidence:
    def test_fields(self):
        chunk = EvidenceChunk(text="fact", distance=0.2)
        ev = ScoringEvidence(
            chunks=[chunk],
            nli_premise="premise",
            nli_hypothesis="hypothesis",
            nli_score=0.3,
        )
        assert len(ev.chunks) == 1
        assert ev.nli_score == 0.3


class TestCoherenceScoreEvidence:
    def test_evidence_field_none_by_default(self):
        cs = CoherenceScore(score=0.8, approved=True, h_logical=0.1, h_factual=0.1)
        assert cs.evidence is None
        assert cs.warning is False

    def test_evidence_field_present(self):
        ev = ScoringEvidence(
            chunks=[], nli_premise="p", nli_hypothesis="h", nli_score=0.1
        )
        cs = CoherenceScore(
            score=0.8, approved=True, h_logical=0.1, h_factual=0.1, evidence=ev
        )
        assert cs.evidence is ev

    def test_warning_field(self):
        cs = CoherenceScore(
            score=0.55, approved=True, h_logical=0.2, h_factual=0.2, warning=True
        )
        assert cs.warning is True


class TestReviewResultFallback:
    def test_fallback_used_default_false(self):
        rr = ReviewResult(
            output="test", coherence=None, halted=True, candidates_evaluated=1
        )
        assert rr.fallback_used is False

    def test_fallback_used_true(self):
        rr = ReviewResult(
            output="test",
            coherence=None,
            halted=False,
            candidates_evaluated=1,
            fallback_used=True,
        )
        assert rr.fallback_used is True


class TestVectorStoreDistances:
    def test_in_memory_backend_returns_distance(self):
        backend = InMemoryBackend()
        backend.add("1", "The sky is blue")
        results = backend.query("sky blue", n_results=1)
        assert len(results) == 1
        assert "distance" in results[0]
        assert 0.0 <= results[0]["distance"] <= 1.0

    def test_retrieve_context_with_chunks(self):
        store = VectorGroundTruthStore(auto_index=True)
        chunks = store.retrieve_context_with_chunks("What color is the sky?")
        assert len(chunks) > 0
        assert isinstance(chunks[0], EvidenceChunk)
        assert chunks[0].distance >= 0.0


class TestScorerEvidence:
    def test_review_returns_evidence_on_match(self):
        store = GroundTruthStore()
        scorer = CoherenceScorer(threshold=0.5, ground_truth_store=store, use_nli=False)
        approved, score = scorer.review("What color is the sky?", "The sky is blue.")
        assert score.evidence is not None
        assert len(score.evidence.chunks) > 0
        assert score.evidence.nli_premise != ""
        assert score.evidence.nli_hypothesis == "The sky is blue."

    def test_review_evidence_none_without_store(self):
        scorer = CoherenceScorer(threshold=0.5, use_nli=False)
        _, score = scorer.review("test", "consistent with reality")
        assert score.evidence is None

    def test_rejected_output_has_evidence(self):
        store = GroundTruthStore()
        scorer = CoherenceScorer(threshold=0.9, ground_truth_store=store, use_nli=False)
        approved, score = scorer.review(
            "What color is the sky?", "The sky color is green."
        )
        assert not approved
        assert score.evidence is not None
        assert len(score.evidence.chunks) > 0


class TestSoftZone:
    def test_soft_zone_sets_warning(self):
        store = GroundTruthStore()
        # threshold=0.5, soft_limit=0.8 — anything between 0.5 and 0.8 gets warning
        scorer = CoherenceScorer(
            threshold=0.5,
            ground_truth_store=store,
            use_nli=False,
            soft_limit=0.8,
        )
        approved, score = scorer.review(
            "sky", "The sky color is blue. This is consistent with reality"
        )
        assert approved
        # Score should be between 0.5 and 0.8 with heuristic scoring
        if score.score < 0.8:
            assert score.warning is True

    def test_above_soft_limit_no_warning(self):
        store = GroundTruthStore()
        scorer = CoherenceScorer(
            threshold=0.3,
            ground_truth_store=store,
            use_nli=False,
            soft_limit=0.4,
        )
        approved, score = scorer.review(
            "sky", "The sky color is blue. This is consistent with reality"
        )
        assert approved
        if score.score >= 0.4:
            assert score.warning is False


class TestFallbackRetrieval:
    def test_fallback_retrieval_on_halt(self):
        agent = CoherenceAgent(fallback="retrieval")
        # Force a high threshold so everything fails
        agent.scorer.threshold = 0.99
        result = agent.process("What color is the sky?")
        assert not result.halted
        assert result.fallback_used is True
        assert "verified sources" in result.output.lower()

    def test_fallback_disclaimer_on_halt(self):
        agent = CoherenceAgent(fallback="disclaimer")
        agent.scorer.threshold = 0.99
        result = agent.process("What color is the sky?")
        assert not result.halted
        assert result.fallback_used is True
        assert "could not be fully verified" in result.output.lower()

    def test_no_fallback_returns_halt(self):
        agent = CoherenceAgent()
        agent.scorer.threshold = 0.99
        result = agent.process("What color is the sky?")
        assert result.halted
        assert result.fallback_used is False
        assert "SYSTEM HALT" in result.output

    def test_best_rejected_score_on_halt(self):
        agent = CoherenceAgent()
        agent.scorer.threshold = 0.99
        result = agent.process("What color is the sky?")
        assert result.halted
        # best_rejected_score should be populated instead of None
        assert result.coherence is not None
        assert result.coherence.score > 0.0


class TestAgentDisclaimer:
    def test_warning_adds_disclaimer(self):
        agent = CoherenceAgent()
        agent.scorer.soft_limit = 1.0  # everything approved gets a warning
        result = agent.process("What color is the sky?")
        if not result.halted:
            assert result.output.startswith("[Confidence: moderate]")

    def test_custom_disclaimer_prefix(self):
        agent = CoherenceAgent(disclaimer_prefix="[LOW CONFIDENCE] ")
        agent.scorer.soft_limit = 1.0
        result = agent.process("What color is the sky?")
        if not result.halted:
            assert result.output.startswith("[LOW CONFIDENCE]")
