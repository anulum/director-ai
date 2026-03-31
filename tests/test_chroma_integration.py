# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ChromaDB Integration Tests

"""Multi-angle integration tests for ChromaBackend using in-memory ChromaDB.

Covers: add/count, multiple adds, query relevance, metadata, empty collection,
n_results limit, VectorGroundTruthStore integration, add_fact, semantic
similarity, parametrised n_results, pipeline integration with scorer,
and performance documentation.

Requires: pip install chromadb
"""

import pytest

try:
    import chromadb  # noqa: F401

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

from director_ai.core.vector_store import VectorGroundTruthStore

if _CHROMA_AVAILABLE:
    from director_ai.core.vector_store import ChromaBackend


@pytest.mark.integration
@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestChromaIntegration:
    """Integration tests for ChromaBackend with in-memory Chroma."""

    @pytest.fixture
    def backend(self):
        """Fresh in-memory Chroma backend for each test."""
        return ChromaBackend(
            collection_name=f"test_{id(self)}",
            persist_directory=None,  # In-memory
        )

    def test_add_and_count(self, backend):
        assert backend.count() == 0
        backend.add("doc1", "The sky is blue")
        assert backend.count() == 1

    def test_add_multiple(self, backend):
        backend.add("d1", "SCPN Layer 1: Quantum Biological")
        backend.add("d2", "SCPN Layer 2: Neurochemical")
        backend.add("d3", "SCPN Layer 7: Geometrical-Symbolic")
        assert backend.count() == 3

    def test_query_returns_results(self, backend):
        backend.add("d1", "The Earth orbits the Sun")
        backend.add("d2", "Water freezes at zero degrees Celsius")
        backend.add("d3", "SCPN couples 16 oscillator layers")
        results = backend.query("planetary orbits", n_results=2)
        assert len(results) >= 1
        assert any("Earth" in r["text"] for r in results)

    def test_query_with_metadata(self, backend):
        backend.add(
            "d1",
            "Kuramoto coupling K=0.302",
            metadata={"layer": "L1-L2"},
        )
        results = backend.query("coupling strength")
        assert len(results) >= 1
        assert results[0]["metadata"].get("layer") == "L1-L2"

    def test_query_empty_collection(self, backend):
        results = backend.query("anything")
        assert results == []

    def test_query_n_results(self, backend):
        for i in range(10):
            backend.add(f"d{i}", f"Document number {i}")
        results = backend.query("document", n_results=3)
        assert len(results) == 3

    def test_vector_store_with_chroma(self, backend):
        """VectorGroundTruthStore works with ChromaBackend."""
        store = VectorGroundTruthStore(backend=backend)
        store.ingest(["sky color is blue", "SCPN has 16 layers"])
        assert backend.count() > 0
        ctx = store.retrieve_context("What color is the sky?")
        assert ctx is not None

    def test_vector_store_add_fact(self, backend):
        store = VectorGroundTruthStore(backend=backend)
        initial = backend.count()
        store.add_fact("omega_1", "1.329 rad/s")
        assert backend.count() == initial + 1

    def test_semantic_similarity(self, backend):
        """Semantically similar queries retrieve relevant docs."""
        backend.add("d1", "Coherence measures alignment of oscillators")
        backend.add("d2", "The restaurant serves Italian food")
        backend.add("d3", "Phase synchronisation in neural networks")
        results = backend.query("oscillator synchrony", n_results=2)
        texts = [r["text"] for r in results]
        # At least one result should be about oscillators/synchrony
        assert any("oscillat" in t.lower() or "synchron" in t.lower() for t in texts)

    @pytest.mark.parametrize("n_results", [1, 2, 3, 5])
    def test_parametrised_n_results(self, backend, n_results):
        for i in range(10):
            backend.add(f"d{i}", f"Document about topic {i}")
        results = backend.query("topic", n_results=n_results)
        assert len(results) <= n_results

    def test_scorer_pipeline_with_chroma(self, backend):
        """Full pipeline: ChromaBackend → VectorGroundTruthStore → CoherenceScorer."""
        from director_ai.core import CoherenceScorer

        store = VectorGroundTruthStore(backend=backend)
        store.ingest(["Paris is the capital of France"])
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        approved, score = scorer.review("capital of France", "Paris")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_add_performance(self, backend):
        """Document add latency."""
        import time

        t0 = time.perf_counter()
        for i in range(100):
            backend.add(f"perf{i}", f"Performance test document {i}")
        per_add_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_add_ms < 500, f"Chroma add took {per_add_ms:.1f}ms/doc"
