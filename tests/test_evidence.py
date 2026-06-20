# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence, Fallback, and Soft Zone Tests
"""Multi-angle tests for evidence collection, fallback, and soft zone.

Covers: evidence retrieval, chunk scoring, fallback mode, soft warning zone,
halt evidence generation, pipeline integration with CoherenceScorer, and
performance documentation.
"""

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
            chunks=[],
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.1,
        )
        cs = CoherenceScore(
            score=0.8,
            approved=True,
            h_logical=0.1,
            h_factual=0.1,
            evidence=ev,
        )
        assert cs.evidence is ev

    def test_warning_field(self):
        cs = CoherenceScore(
            score=0.55,
            approved=True,
            h_logical=0.2,
            h_factual=0.2,
            warning=True,
        )
        assert cs.warning is True


class TestReviewResultFallback:
    def test_fallback_used_default_false(self):
        rr = ReviewResult(
            output="test",
            coherence=None,
            halted=True,
            candidates_evaluated=1,
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
        store = VectorGroundTruthStore()
        store.ingest(["sky color is blue", "SCPN has 16 layers"])
        chunks = store.retrieve_context_with_chunks("What color is the sky?")
        assert len(chunks) > 0
        assert isinstance(chunks[0], EvidenceChunk)
        assert chunks[0].distance >= 0.0


class TestScorerEvidence:
    def test_review_returns_evidence_on_match(self):
        store = GroundTruthStore.with_demo_facts()
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
        store = GroundTruthStore.with_demo_facts()
        scorer = CoherenceScorer(threshold=0.9, ground_truth_store=store, use_nli=False)
        approved, score = scorer.review(
            "What color is the sky?",
            "The sky color is green.",
        )
        assert not approved
        assert score.evidence is not None
        assert len(score.evidence.chunks) > 0


class TestSoftZone:
    def test_soft_zone_sets_warning(self):
        store = GroundTruthStore.with_demo_facts()
        # threshold=0.5, soft_limit=0.8 — anything between 0.5 and 0.8 gets warning
        scorer = CoherenceScorer(
            threshold=0.5,
            ground_truth_store=store,
            use_nli=False,
            soft_limit=0.8,
        )
        approved, score = scorer.review(
            "sky",
            "The sky color is blue. This is consistent with reality",
        )
        assert approved
        # Score should be between 0.5 and 0.8 with heuristic scoring
        if score.score < 0.8:
            assert score.warning is True

    def test_above_soft_limit_no_warning(self):
        store = GroundTruthStore.with_demo_facts()
        scorer = CoherenceScorer(
            threshold=0.3,
            ground_truth_store=store,
            use_nli=False,
            soft_limit=0.4,
        )
        approved, score = scorer.review(
            "sky",
            "The sky color is blue. This is consistent with reality",
        )
        assert approved
        if score.score >= 0.4:
            assert score.warning is False


class TestFallbackRetrieval:
    def test_fallback_retrieval_on_halt(self):
        store = GroundTruthStore.with_demo_facts()
        agent = CoherenceAgent(fallback="retrieval", _store=store)
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
        assert "HALT" in result.output

    def test_vector_fallback_joins_chunk_context(self):
        class _ListContextStore(VectorGroundTruthStore):
            def retrieve_context(self, query, top_k=3, tenant_id=""):
                del query, top_k, tenant_id
                return [
                    EvidenceChunk(text="alpha fact", distance=0.1),
                    EvidenceChunk(text="beta fact", distance=0.2),
                ]

        agent = CoherenceAgent(fallback="retrieval", _store=_ListContextStore())
        rejected_score = CoherenceScore(
            score=0.2,
            approved=False,
            h_logical=0.4,
            h_factual=0.4,
        )
        result = agent._retrieval_fallback("question", "", rejected_score, 2)
        assert result is not None
        assert result.output == "Based on verified sources: alpha fact; beta fact"
        assert result.fallback_used is True

    def test_retrieval_fallback_returns_none_without_context(self):
        class _EmptyStore(GroundTruthStore):
            def retrieve_context(self, query, tenant_id=""):
                del query, tenant_id
                return ""

        agent = CoherenceAgent(fallback="retrieval", _store=_EmptyStore())
        rejected_score = CoherenceScore(
            score=0.2,
            approved=False,
            h_logical=0.4,
            h_factual=0.4,
        )
        assert agent._retrieval_fallback("question", "", rejected_score, 1) is None

    def test_vector_fallback_accepts_string_context(self):
        class _StringContextStore(VectorGroundTruthStore):
            def retrieve_context(self, query, top_k=3, tenant_id=""):
                del query, top_k, tenant_id
                return "verified vector context"

        agent = CoherenceAgent(fallback="retrieval", _store=_StringContextStore())
        rejected_score = CoherenceScore(
            score=0.2,
            approved=False,
            h_logical=0.4,
            h_factual=0.4,
        )
        result = agent._retrieval_fallback("question", "", rejected_score, 1)
        assert result is not None
        assert result.output == "Based on verified sources: verified vector context"

    def test_retrieval_fallback_miss_continues_to_halt(self):
        class _EmptyStore(GroundTruthStore):
            def retrieve_context(self, query, tenant_id=""):
                del query, tenant_id
                return ""

        agent = CoherenceAgent(fallback="retrieval", _store=_EmptyStore())
        rejected_score = CoherenceScore(
            score=0.2,
            approved=False,
            h_logical=0.4,
            h_factual=0.4,
        )
        result = agent._handle_rejection(
            "question",
            "",
            ("bad", rejected_score, 0.2),
            1,
        )
        assert result.halted is True
        assert result.fallback_used is False

    def test_halt_evidence_preserves_chunk_scores(self):
        evidence = ScoringEvidence(
            chunks=[EvidenceChunk(text="fact", distance=0.1)],
            nli_premise="fact",
            nli_hypothesis="claim",
            nli_score=0.7,
            chunk_scores=[0.7],
        )
        rejected_score = CoherenceScore(
            score=0.2,
            approved=False,
            h_logical=0.4,
            h_factual=0.4,
            evidence=evidence,
        )
        agent = CoherenceAgent()
        result = agent._handle_rejection(
            "question", "", ("bad", rejected_score, 0.2), 1
        )
        assert result.halted is True
        assert result.halt_evidence is not None
        assert result.halt_evidence.nli_scores == [0.7]

    def test_best_rejected_score_on_halt(self):
        agent = CoherenceAgent()
        agent.scorer.threshold = 0.99
        result = agent.process("What color is the sky?")
        assert result.halted
        assert result.coherence is not None
        assert result.coherence.score > 0.0


class TestAgentDisclaimer:
    def test_warning_adds_disclaimer(self):
        agent = CoherenceAgent()
        agent.scorer.soft_limit = 1.0  # everything approved gets a warning
        result = agent.process("What color is the sky?")
        if not result.halted:
            assert result.output.startswith("[Unverified]")

    def test_custom_disclaimer_prefix(self):
        agent = CoherenceAgent(disclaimer_prefix="[LOW CONFIDENCE] ")
        agent.scorer.soft_limit = 1.0
        result = agent.process("What color is the sky?")
        if not result.halted:
            assert result.output.startswith("[LOW CONFIDENCE]")
