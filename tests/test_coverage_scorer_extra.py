# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for scorer.py â€” LLM judge, VectorGroundTruthStore evidence."""

from __future__ import annotations

from unittest.mock import patch

from director_ai.core import CoherenceScorer, GroundTruthStore
from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore


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

    def test_evidence_falls_back_to_keyword_chunks_when_vector_misses(self):
        class EmptyBackend(InMemoryBackend):
            def query(self, text, n_results=3, tenant_id=""):
                return []

        store = VectorGroundTruthStore(backend=EmptyBackend())
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
        assert ev is not None
        assert ev.chunks
        assert ev.chunks[0].source == "keyword"


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
