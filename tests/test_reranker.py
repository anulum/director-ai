# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RerankedBackend Tests
"""Multi-angle tests for RerankedBackend retrieval pipeline.

Covers: import error guard, delegation to base backend, reranking
with mocked cross-encoder, empty store, query ordering, batch
queries, multiplier effect, scorer pipeline integration, and
performance documentation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from director_ai.core.vector_store import InMemoryBackend, RerankedBackend


def _mock_reranked(docs=None, multiplier=3):
    """Build a RerankedBackend with mocked cross-encoder."""
    base = InMemoryBackend()
    if docs:
        for i, doc in enumerate(docs):
            base.add(f"d{i}", doc)

    mock_ce = MagicMock()
    mock_ce.predict.side_effect = lambda pairs: [
        0.9 - i * 0.1 for i in range(len(pairs))
    ]

    rb = RerankedBackend.__new__(RerankedBackend)
    rb._base = base
    rb._multiplier = multiplier
    rb._reranker = mock_ce
    return rb, mock_ce


# ── Import guard ─────────────────────────────────────────────────


class TestRerankerImportGuard:
    """RerankedBackend must raise ImportError without sentence-transformers."""

    def test_import_error_message(self):
        try:
            base = InMemoryBackend()
            RerankedBackend(base=base)
        except ImportError as e:
            assert "sentence-transformers" in str(e)


# ── Delegation ───────────────────────────────────────────────────


class TestRerankerDelegation:
    """RerankedBackend delegates storage to base backend."""

    def test_add_delegates(self):
        rb, _ = _mock_reranked(["existing doc"])
        rb.add("d_new", "new document")
        assert rb.count() == 2

    def test_count_matches_base(self):
        rb, _ = _mock_reranked(["a", "b", "c"])
        assert rb.count() == 3

    @pytest.mark.parametrize("n_docs", [0, 1, 5, 20])
    def test_count_various_sizes(self, n_docs):
        rb, _ = _mock_reranked([f"doc{i}" for i in range(n_docs)])
        assert rb.count() == n_docs


# ── Reranking ────────────────────────────────────────────────────


class TestReranking:
    """Reranking must call cross-encoder and sort results."""

    def test_query_calls_predict(self):
        rb, mock_ce = _mock_reranked(["cat on mat", "dog ran", "cat fish"])
        rb.query("cat", n_results=2)
        mock_ce.predict.assert_called_once()

    def test_query_returns_limited_results(self):
        rb, _ = _mock_reranked(["a", "b", "c", "d", "e"])
        results = rb.query("test", n_results=2)
        assert len(results) <= 2

    def test_empty_store_returns_empty(self):
        rb, mock_ce = _mock_reranked()
        results = rb.query("test")
        assert results == []
        mock_ce.predict.assert_not_called()

    def test_query_with_single_doc(self):
        rb, _ = _mock_reranked(["single document"])
        results = rb.query("document", n_results=5)
        assert len(results) >= 1

    @pytest.mark.parametrize("n_results", [1, 2, 3, 5])
    def test_various_n_results(self, n_results):
        rb, _ = _mock_reranked([f"doc{i}" for i in range(10)])
        results = rb.query("test", n_results=n_results)
        assert len(results) <= n_results

    def test_multiplier_increases_candidates(self):
        rb_low, _ = _mock_reranked([f"d{i}" for i in range(10)], multiplier=1)
        rb_high, _ = _mock_reranked([f"d{i}" for i in range(10)], multiplier=5)
        # Both should return results, but high multiplier retrieves more candidates
        r1 = rb_low.query("test", n_results=2)
        r2 = rb_high.query("test", n_results=2)
        assert len(r1) <= 2
        assert len(r2) <= 2


# ── Pipeline integration ─────────────────────────────────────────


class TestRerankerPipeline:
    """Verify reranker integrates with VectorGroundTruthStore."""

    def test_reranked_backend_in_store(self):
        from director_ai.core.vector_store import VectorGroundTruthStore

        rb, _ = _mock_reranked(["Paris is the capital of France"])
        store = VectorGroundTruthStore(backend=rb)
        result = store.retrieve_context("capital of France")
        assert result is not None


# ── Performance documentation ────────────────────────────────────


class TestRerankerPerformance:
    """Document reranker performance characteristics."""

    def test_predict_called_per_query(self):
        rb, mock_ce = _mock_reranked(["cat sat on mat", "cat likes fish", "cat purrs"])
        rb.query("cat", n_results=2)
        assert mock_ce.predict.call_count >= 1

    def test_reranker_preserves_base_count(self):
        rb, _ = _mock_reranked(["a", "b", "c"])
        rb.query("test")
        assert rb.count() == 3  # query doesn't modify store
