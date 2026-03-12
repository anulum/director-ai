"""Tests for HybridBackend (BM25 + dense with RRF)."""

from __future__ import annotations

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


class TestHybridBackendRegistration:
    def test_registered(self):
        from director_ai.core.vector_store import get_vector_backend
        cls = get_vector_backend("hybrid")
        assert cls is HybridBackend
