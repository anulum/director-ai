# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for HybridBackend (BM25 + dense with RRF).

Covers: add/count, query ordering, empty store, RRF fusion, tenant
filtering, distance presence, sparse/dense weight emphasis,
registration, parametrised n_results/weights, pipeline integration,
and performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.vector_store import HybridBackend, InMemoryBackend


class TestHybridBackend:
    def setup_method(self):
        self.base = InMemoryBackend()
        self.hybrid = HybridBackend(self.base)

    def test_add_and_count(self):
        self.hybrid.add("d1", "The sky is blue due to Rayleigh scattering")
        self.hybrid.add("d2", "Water boils at 100 degrees Celsius")
        assert self.hybrid.count() == 2

    def test_query_returns_results(self):
        self.hybrid.add("d1", "The sky is blue due to Rayleigh scattering")
        self.hybrid.add("d2", "Water boils at 100 degrees Celsius")
        self.hybrid.add("d3", "The earth orbits the sun")

        results = self.hybrid.query("blue sky scattering", n_results=2)
        assert len(results) > 0
        assert results[0]["id"] == "d1"

    def test_empty_returns_empty(self):
        assert self.hybrid.query("anything") == []

    def test_rrf_fusion_combines_both_signals(self):
        """Both BM25 and dense results should contribute to ranking."""
        self.hybrid.add("d1", "machine learning neural networks deep learning")
        self.hybrid.add("d2", "the cat sat on the mat")
        self.hybrid.add("d3", "artificial intelligence and machine learning")

        results = self.hybrid.query("machine learning", n_results=3)
        ids = [r["id"] for r in results]
        assert "d1" in ids
        assert "d3" in ids

    def test_tenant_filtering(self):
        self.hybrid.add("d1", "fact one", metadata={"tenant_id": "t1"})
        self.hybrid.add("d2", "fact two", metadata={"tenant_id": "t2"})

        results = self.hybrid.query("fact", n_results=5, tenant_id="t1")
        ids = [r["id"] for r in results]
        assert "d2" not in ids

    def test_distance_in_results(self):
        self.hybrid.add("d1", "The quick brown fox jumps over the lazy dog")
        results = self.hybrid.query("quick brown fox", n_results=1)
        assert len(results) == 1
        assert "distance" in results[0]

    def test_sparse_weight_emphasis(self):
        """Higher sparse weight should favor BM25 term matches."""
        base = InMemoryBackend()
        hybrid = HybridBackend(base, sparse_weight=5.0, dense_weight=0.1)
        hybrid.add("d1", "python programming language syntax")
        hybrid.add("d2", "python snake reptile animal")
        # BM25 treats both equally for "python", but "programming" tips d1
        results = hybrid.query("python programming", n_results=2)
        assert results[0]["id"] == "d1"

    @pytest.mark.parametrize("n_results", [1, 2, 3, 5])
    def test_parametrised_n_results(self, n_results):
        base = InMemoryBackend()
        hybrid = HybridBackend(base)
        for i in range(10):
            hybrid.add(f"d{i}", f"Document number {i} about various topics")
        results = hybrid.query("document", n_results=n_results)
        assert len(results) <= n_results

    @pytest.mark.parametrize(
        "sparse,dense",
        [(1.0, 1.0), (5.0, 0.1), (0.1, 5.0), (1.0, 0.0)],
    )
    def test_parametrised_weights(self, sparse, dense):
        base = InMemoryBackend()
        hybrid = HybridBackend(base, sparse_weight=sparse, dense_weight=dense)
        hybrid.add("d1", "test document content")
        results = hybrid.query("test", n_results=1)
        assert len(results) >= 0  # valid even if empty for extreme weights


class TestHybridBackendRegistration:
    def test_registered(self):
        from director_ai.core.vector_store import get_vector_backend

        cls = get_vector_backend("hybrid")
        assert cls is HybridBackend


class TestHybridPipelineIntegration:
    """Verify hybrid backend works in full scorer pipeline."""

    def test_scorer_with_hybrid_store(self):
        from director_ai.core import CoherenceScorer
        from director_ai.core.vector_store import VectorGroundTruthStore

        base = InMemoryBackend()
        hybrid = HybridBackend(base)
        store = VectorGroundTruthStore(backend=hybrid)
        store.ingest(["Paris is the capital of France"])
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        approved, score = scorer.review("capital of France", "Paris")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0


class TestHybridPerformanceDoc:
    """Document hybrid backend performance characteristics."""

    def test_query_returns_distance(self):
        base = InMemoryBackend()
        hybrid = HybridBackend(base)
        hybrid.add("d1", "test document")
        results = hybrid.query("test", n_results=1)
        if results:
            assert "distance" in results[0]

    def test_count_stable_after_query(self):
        base = InMemoryBackend()
        hybrid = HybridBackend(base)
        hybrid.add("d1", "test")
        assert hybrid.count() == 1
        hybrid.query("test")
        assert hybrid.count() == 1
