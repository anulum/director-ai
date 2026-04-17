# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Reranker Tests
"""Multi-angle tests for VectorStore RerankedBackend.

Covers: reranking order, top_k multiplier, delegation, empty query,
parametrised n_results, multiplier values, scorer pipeline integration,
and performance documentation.
"""

from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.vector_store import InMemoryBackend, RerankedBackend


class _MockCrossEncoder:
    """Simulates sentence_transformers.CrossEncoder.predict()."""

    def predict(self, pairs):
        return [float(len(pairs) - i) for i in range(len(pairs))]


def _make_reranked(base, top_k_multiplier=3):
    """Build RerankedBackend with mocked sentence-transformers import."""
    mock_st = MagicMock()
    mock_st.CrossEncoder.return_value = _MockCrossEncoder()
    with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
        return RerankedBackend(base, top_k_multiplier=top_k_multiplier)


class TestRerankedBackend:
    def test_reranking_reverses_order(self):
        base = InMemoryBackend()
        base.add("d1", "first doc about cats")
        base.add("d2", "second doc about cats")
        base.add("d3", "third doc about cats")

        reranker = _make_reranked(base, top_k_multiplier=3)
        results = reranker.query("doc about cats", n_results=3)
        assert len(results) == 3
        texts = [r["text"] for r in results]
        assert texts[0] == "first doc about cats"

    def test_top_k_multiplier_fetches_more(self):
        base = MagicMock()
        base.query.return_value = [
            {"id": f"d{i}", "text": f"doc{i}", "distance": float(i)} for i in range(6)
        ]
        base.count.return_value = 6

        reranker = _make_reranked(base, top_k_multiplier=3)
        results = reranker.query("test", n_results=2)
        base.query.assert_called_once_with("test", n_results=6, tenant_id="")
        assert len(results) == 2

    def test_add_delegates_to_base(self):
        base = MagicMock()
        reranker = _make_reranked(base)
        reranker.add("id1", "text1", {"key": "val"})
        base.add.assert_called_once_with("id1", "text1", {"key": "val"})

    def test_count_delegates_to_base(self):
        base = MagicMock()
        base.count.return_value = 42
        reranker = _make_reranked(base)
        assert reranker.count() == 42

    def test_empty_query_returns_empty(self):
        base = MagicMock()
        base.query.return_value = []
        reranker = _make_reranked(base)
        results = reranker.query("test", n_results=3)
        assert results == []

    @pytest.mark.parametrize("n_results", [1, 2, 3, 5])
    def test_parametrised_n_results(self, n_results):
        base = MagicMock()
        base.query.return_value = [
            {"id": f"d{i}", "text": f"doc{i}", "distance": float(i)} for i in range(10)
        ]
        base.count.return_value = 10
        reranker = _make_reranked(base)
        results = reranker.query("test", n_results=n_results)
        assert len(results) == n_results

    @pytest.mark.parametrize("multiplier", [1, 2, 3, 5])
    def test_parametrised_multiplier(self, multiplier):
        base = MagicMock()
        base.query.return_value = [
            {"id": f"d{i}", "text": f"doc{i}", "distance": float(i)}
            for i in range(multiplier * 2)
        ]
        base.count.return_value = multiplier * 2
        reranker = _make_reranked(base, top_k_multiplier=multiplier)
        results = reranker.query("test", n_results=2)
        assert len(results) == 2


class TestRerankerPipelineIntegration:
    """Verify reranker integrates into scorer pipeline."""

    def test_reranked_in_ground_truth_store(self):
        from director_ai.core.vector_store import VectorGroundTruthStore

        base = InMemoryBackend()
        base.add("d1", "Paris is the capital of France")
        reranker = _make_reranked(base)
        store = VectorGroundTruthStore(backend=reranker)
        result = store.retrieve_context("capital of France")
        assert result is not None
        assert "Paris" in result


class TestRerankerPerformanceDoc:
    """Document reranker performance characteristics."""

    def test_base_backend_count_stable(self):
        base = InMemoryBackend()
        base.add("d1", "test")
        reranker = _make_reranked(base)
        assert reranker.count() == 1
        reranker.query("test")
        assert reranker.count() == 1  # query must not modify store

    def test_add_increments_count(self):
        base = InMemoryBackend()
        reranker = _make_reranked(base)
        assert reranker.count() == 0
        reranker.add("d1", "test")
        assert reranker.count() == 1
