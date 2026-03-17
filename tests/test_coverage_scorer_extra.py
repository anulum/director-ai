"""Coverage tests for scorer.py — LLM judge, VectorGroundTruthStore evidence."""

from __future__ import annotations

from unittest.mock import patch

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.vector_store import VectorGroundTruthStore


class TestLLMJudge:
    def test_llm_judge_unknown_provider(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="unknown",
        )
        result = scorer._llm_judge_check("q", "a", 0.5)
        assert result == 0.5

    def test_llm_judge_import_error(self):
        scorer = CoherenceScorer(
            use_nli=False,
            llm_judge_enabled=True,
            llm_judge_provider="openai",
        )
        with patch.dict("sys.modules", {"openai": None}):
            result = scorer._llm_judge_check("q", "a", 0.5)
            assert result == 0.5


class TestScorerWithVectorStore:
    def test_evidence_with_vector_store(self):
        store = VectorGroundTruthStore()
        store.add_fact("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=store,
        )
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "sky",
            "The sky is blue.",
        )
        assert 0.0 <= div <= 1.0

    def test_evidence_no_context(self):
        store = VectorGroundTruthStore()
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=store,
        )
        div, ev = scorer.calculate_factual_divergence_with_evidence(
            "zzz_nonexistent_xyz",
            "anything",
        )
        assert div == 0.5
        assert ev is None


class TestScorerComputeDivergence:
    def test_compute_divergence(self):
        scorer = CoherenceScorer(use_nli=False)
        div = scorer.compute_divergence("sky?", "The sky is blue.")
        assert 0.0 <= div <= 1.0

    def test_compute_divergence_strict(self):
        store = GroundTruthStore()
        store.add("sky", "The sky is blue.")
        scorer = CoherenceScorer(
            use_nli=False,
            strict_mode=True,
            ground_truth_store=store,
        )
        div = scorer.compute_divergence("sky", "The sky is blue.")
        assert div > 0.5
