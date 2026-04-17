# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAG Integration Tests
"""Multi-angle tests for RAG-enabled CoherenceScorer pipeline.

Covers: retrieval from demo facts, factual divergence scoring,
truth vs lie divergence comparison, scorer + store wiring,
edge cases (empty store, missing facts), backend selection,
and pipeline performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceScorer, GroundTruthStore


@pytest.fixture
def store():
    return GroundTruthStore.with_demo_facts()


@pytest.fixture
def rag_scorer(store):
    return CoherenceScorer(ground_truth_store=store, use_nli=False)


# ── Retrieval ─────────────────────────────────────────────────────


class TestRetrieval:
    """Verify retrieval from demo knowledge base."""

    def test_scpn_layers_retrieval(self, store):
        context = store.retrieve_context("How many layers are in the SCPN?")
        assert context is not None
        assert "16" in context

    def test_demo_facts_not_empty(self, store):
        assert len(store.facts) > 0

    def test_scpn_query_returns_context(self, store):
        result = store.retrieve_context("SCPN layers")
        assert result is not None
        assert len(result) > 0


# ── Factual divergence ────────────────────────────────────────────


class TestFactualDivergence:
    """Test factual divergence scoring via RAG pipeline."""

    def test_truthful_response_low_divergence(self, rag_scorer):
        h = rag_scorer.calculate_factual_divergence(
            "What color is the sky?", "The sky color is blue."
        )
        assert h < 0.5

    def test_false_response_high_divergence(self, rag_scorer):
        h = rag_scorer.calculate_factual_divergence(
            "What color is the sky?", "The sky color is green."
        )
        assert h > 0.8

    def test_truth_less_divergent_than_lie(self, rag_scorer):
        h_truth = rag_scorer.calculate_factual_divergence(
            "sky colour?", "The sky is blue."
        )
        h_lie = rag_scorer.calculate_factual_divergence(
            "sky colour?", "The sky is green."
        )
        assert h_truth < h_lie

    def test_divergence_in_range(self, rag_scorer):
        h = rag_scorer.calculate_factual_divergence("test", "test")
        assert 0.0 <= h <= 1.0


# ── Scorer + store integration ────────────────────────────────────


class TestScorerStoreWiring:
    """Verify scorer correctly uses ground truth store in full pipeline."""

    def test_review_with_store(self, rag_scorer):
        approved, score = rag_scorer.review(
            "What is the SCPN?", "The SCPN has 16 layers."
        )
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0
        assert hasattr(score, "h_factual")

    def test_review_without_store(self):
        scorer = CoherenceScorer(use_nli=False)
        approved, score = scorer.review("Q", "A")
        assert isinstance(approved, bool)

    def test_scorer_with_empty_store(self):
        empty_store = GroundTruthStore()
        scorer = CoherenceScorer(ground_truth_store=empty_store, use_nli=False)
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)


# ── Backend selection with RAG ────────────────────────────────────


class TestRAGBackendSelection:
    """Test RAG works with different scorer backends."""

    @pytest.mark.parametrize("backend", ["deberta", "lite"])
    def test_various_backends_with_store(self, store, backend):
        scorer = CoherenceScorer(
            ground_truth_store=store,
            use_nli=False,
            scorer_backend=backend,
        )
        approved, score = scorer.review("test", "test")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


# ── Pipeline performance ─────────────────────────────────────────


class TestRAGPerformance:
    """Document RAG pipeline performance characteristics."""

    def test_score_has_factual_component(self, rag_scorer):
        _, score = rag_scorer.review("SCPN", "16 layers")
        assert hasattr(score, "h_factual")
        assert 0.0 <= score.h_factual <= 1.0

    def test_score_has_logical_component(self, rag_scorer):
        _, score = rag_scorer.review("test", "test")
        assert hasattr(score, "h_logical")
        assert 0.0 <= score.h_logical <= 1.0

    def test_evidence_field_present(self, rag_scorer):
        _, score = rag_scorer.review("SCPN layers", "16")
        assert hasattr(score, "evidence")
