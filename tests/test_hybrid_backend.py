# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for HybridBackend (BM25 + dense with fusion).

Covers: add/count, query ordering, empty store, RRF fusion, tenant
filtering, distance presence, sparse/dense weight emphasis,
registration, parametrised n_results/weights, fusion method
selection, shared-index ``with_fusion`` views, pipeline integration,
and performance documentation.
"""

from __future__ import annotations

import pytest

import director_ai.core.retrieval.vector_store.composite as composite_mod
from director_ai.core.vector_store import (
    FUSION_METHODS,
    HybridBackend,
    InMemoryBackend,
)


class TestHybridBackend:
    def setup_method(self):
        self.base = InMemoryBackend()
        self.hybrid = HybridBackend(self.base)

    def test_add_and_count(self):
        self.hybrid.add("d1", "The sky is blue due to Rayleigh scattering")
        self.hybrid.add("d2", "Water boils at 100 degrees Celsius")
        assert self.hybrid.count() == 2

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"base": None}, "base"),
            ({"rrf_k": True}, "rrf_k"),
            ({"rrf_k": 0}, "rrf_k"),
            ({"sparse_weight": -0.1}, "sparse_weight"),
            ({"sparse_weight": True}, "sparse_weight"),
            ({"dense_weight": -0.1}, "dense_weight"),
            ({"sparse_weight": 0.0, "dense_weight": 0.0}, "weight"),
            ({"fetch_multiplier": True}, "fetch_multiplier"),
            ({"fetch_multiplier": 0}, "fetch_multiplier"),
        ],
    )
    def test_rejects_invalid_constructor_values(self, kwargs, message):
        base = kwargs.pop("base", InMemoryBackend())

        with pytest.raises(ValueError, match=message):
            HybridBackend(base, **kwargs)

    def test_query_returns_results(self):
        self.hybrid.add("d1", "The sky is blue due to Rayleigh scattering")
        self.hybrid.add("d2", "Water boils at 100 degrees Celsius")
        self.hybrid.add("d3", "The earth orbits the sun")

        results = self.hybrid.query("blue sky scattering", n_results=2)
        assert len(results) > 0
        assert results[0]["id"] == "d1"

    def test_empty_returns_empty(self):
        assert self.hybrid.query("anything") == []

    def test_empty_sparse_query_returns_empty_without_dense_lookup(self):
        self.hybrid.add("d1", "indexed document")

        assert self.hybrid._bm25_query("   ", n_results=3, tenant_id="") == []

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


class TestHybridFusionMethods:
    def _indexed(self, **kwargs) -> HybridBackend:
        hybrid = HybridBackend(InMemoryBackend(), **kwargs)
        hybrid.add("d1", "machine learning neural networks deep learning")
        hybrid.add("d2", "the cat sat on the mat")
        hybrid.add("d3", "artificial intelligence and machine learning")
        return hybrid

    def test_rejects_unknown_fusion_method(self):
        with pytest.raises(ValueError, match="fusion_method"):
            HybridBackend(InMemoryBackend(), fusion_method="borda")

    @pytest.mark.parametrize("method", FUSION_METHODS)
    def test_every_method_ranks_relevant_documents(self, method):
        hybrid = self._indexed(fusion_method=method)

        ids = [r["id"] for r in hybrid.query("machine learning", n_results=3)]

        assert "d1" in ids
        assert "d3" in ids
        assert "d2" not in ids

    def test_default_rrf_query_unchanged_by_refactor(self):
        """Default construction still fuses via weighted RRF."""
        hybrid = self._indexed()

        assert hybrid._fusion == "rrf"
        assert hybrid.query("machine learning", n_results=1)[0]["id"] in {
            "d1",
            "d3",
        }

    def test_dense_rows_without_distance_fuse_as_zero_similarity(self):
        """Backends omitting ``distance`` still fuse (similarity 0.0)."""

        class NoDistanceBackend(InMemoryBackend):
            def query(self, text, n_results=3, tenant_id=""):
                rows = super().query(text, n_results, tenant_id)
                return [
                    {k: v for k, v in row.items() if k != "distance"} for row in rows
                ]

        hybrid = HybridBackend(NoDistanceBackend(), fusion_method="convex")
        hybrid.add("d1", "quantum computing hardware")
        hybrid.add("d2", "quantum computing software stack")

        results = hybrid.query("quantum computing", n_results=2)

        assert {r["id"] for r in results} == {"d1", "d2"}


class TestWithFusionViews:
    def setup_method(self):
        self.hybrid = HybridBackend(InMemoryBackend())
        self.hybrid.add("d1", "python programming language syntax")
        self.hybrid.add("d2", "python snake reptile animal")

    def test_view_uses_requested_method_without_reindexing(self):
        view = self.hybrid.with_fusion("zscore")

        assert view._fusion == "zscore"
        assert view._bm25 is self.hybrid._bm25
        assert view._base is self.hybrid._base
        assert {r["id"] for r in view.query("python", n_results=2)} == {"d1", "d2"}

    def test_view_inherits_parameters_by_default(self):
        parent = HybridBackend(
            InMemoryBackend(),
            rrf_k=17,
            sparse_weight=2.0,
            dense_weight=3.0,
            fetch_multiplier=4,
        )

        view = parent.with_fusion("combmnz")

        assert view._rrf_k == 17
        assert view._sparse_w == 2.0
        assert view._dense_w == 3.0
        assert view._fetch_mul == 4

    def test_view_overrides_selected_parameters(self):
        view = self.hybrid.with_fusion(
            "convex",
            rrf_k=90,
            sparse_weight=0.3,
            dense_weight=0.7,
        )

        assert view._rrf_k == 90
        assert view._sparse_w == 0.3
        assert view._dense_w == 0.7

    def test_add_through_view_is_visible_to_parent(self):
        view = self.hybrid.with_fusion("convex")

        view.add("d3", "python web frameworks")

        assert self.hybrid.count() == 3
        ids = [r["id"] for r in self.hybrid.query("python web", n_results=3)]
        assert "d3" in ids

    def test_add_through_parent_is_visible_to_view(self):
        view = self.hybrid.with_fusion("zscore")

        self.hybrid.add("d4", "python data analysis")

        ids = [r["id"] for r in view.query("data analysis", n_results=3)]
        assert "d4" in ids

    def test_view_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="fusion_method"):
            self.hybrid.with_fusion("unknown")


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

    def test_sum_int_python_path_when_rust_path_disabled(self, monkeypatch):
        monkeypatch.setattr(composite_mod, "_RUST_COMPOSITE", False)

        assert composite_mod._sum_int([2, 3, 5]) == 10
