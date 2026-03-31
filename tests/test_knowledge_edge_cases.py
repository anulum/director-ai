# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Knowledge Store Edge Cases (STRONG)
"""Multi-angle edge case tests for GroundTruthStore.

Covers: empty store, add/retrieve, duplicate handling, empty/null queries,
very long facts, Unicode facts, retrieval relevance, capacity, and
pipeline integration with scorer.
"""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceScorer, GroundTruthStore

# ── Empty store ────────────────────────────────────────────────────


class TestEmptyStore:
    """Retrieving from empty store must not crash."""

    def test_empty_store_returns_none_or_empty(self):
        store = GroundTruthStore()
        result = store.retrieve_context("What is the refund policy?")
        assert result == "" or result is None

    @pytest.mark.parametrize(
        "query",
        [
            "anything",
            "",
            "   ",
            "🎉",
            "a" * 10000,
        ],
    )
    def test_empty_store_various_queries(self, query):
        store = GroundTruthStore()
        result = store.retrieve_context(query)
        assert result == "" or result is None


# ── Add and retrieve ───────────────────────────────────────────────


class TestAddRetrieve:
    """Test fact insertion and retrieval accuracy."""

    def test_add_then_retrieve(self):
        store = GroundTruthStore()
        store.add_fact("refund", "Refunds within 30 days only")
        result = store.retrieve_context("refund policy")
        assert result is not None
        assert "30 days" in result

    def test_add_multiple_facts_retrieve_relevant(self):
        store = GroundTruthStore()
        store.add_fact("refund", "Refunds within 30 days")
        store.add_fact("shipping", "Free shipping over $50")
        store.add_fact("warranty", "1 year warranty included")
        result = store.retrieve_context("refund")
        assert result is not None
        assert "30 days" in result

    def test_add_same_key_updates(self):
        store = GroundTruthStore()
        store.add_fact("policy", "Version 1")
        store.add_fact("policy", "Version 2")
        result = store.retrieve_context("policy")
        assert result is not None

    @pytest.mark.parametrize(
        "key,value,query",
        [
            ("capital", "Paris is the capital of France", "capital of France"),
            ("pi", "Pi is approximately 3.14159", "value of pi"),
            ("water", "Water boils at 100°C", "water boiling"),
        ],
    )
    def test_various_facts(self, key, value, query):
        store = GroundTruthStore()
        store.add_fact(key, value)
        result = store.retrieve_context(query)
        assert result is not None
        assert len(result) > 0


# ── Edge case inputs ──────────────────────────────────────────────


class TestEdgeCaseInputs:
    """Handle unusual input gracefully."""

    def test_empty_query(self):
        store = GroundTruthStore()
        store.add_fact("test", "test value")
        result = store.retrieve_context("")
        assert result is None or isinstance(result, str)

    def test_very_long_fact(self):
        store = GroundTruthStore()
        long_fact = "word " * 50_000
        store.add_fact("long", long_fact)
        result = store.retrieve_context("long fact")
        assert isinstance(result, str)

    def test_unicode_facts(self):
        store = GroundTruthStore()
        store.add_fact("arabic", "الإجابة هي 42")
        result = store.retrieve_context("arabic")
        assert result is not None

    def test_null_bytes_in_fact(self):
        store = GroundTruthStore()
        store.add_fact("null", "value\x00with\x00nulls")
        result = store.retrieve_context("null")
        assert result is not None

    def test_empty_fact_value(self):
        store = GroundTruthStore()
        store.add_fact("empty", "")
        # Should not crash
        result = store.retrieve_context("empty")
        assert result is None or isinstance(result, str)


# ── Pipeline integration ──────────────────────────────────────────


class TestKnowledgePipelineIntegration:
    """Verify knowledge store integrates with scorer pipeline."""

    def test_scorer_with_ground_truth(self):
        store = GroundTruthStore()
        store.add_fact("earth", "Earth orbits the Sun")
        scorer = CoherenceScorer(
            use_nli=False,
            ground_truth_store=store,
        )
        approved, score = scorer.review(
            "What does Earth orbit?", "Earth orbits the Sun."
        )
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_scorer_without_ground_truth(self):
        scorer = CoherenceScorer(use_nli=False)
        approved, score = scorer.review("Q", "A")
        assert isinstance(approved, bool)

    def test_factual_divergence_with_store(self):
        store = GroundTruthStore()
        store.add_fact("fact", "The sky is blue")
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        _, score = scorer.review("sky colour", "The sky is blue")
        assert hasattr(score, "h_factual")
