# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — ChromaDB Integration Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Integration tests for ChromaBackend using in-memory ChromaDB.

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
        store = VectorGroundTruthStore(backend=backend, auto_index=True)
        # Sample facts should be indexed
        assert backend.count() > 0
        # Semantic retrieval should work
        ctx = store.retrieve_context("What color is the sky?")
        assert ctx is not None

    def test_vector_store_add_fact(self, backend):
        store = VectorGroundTruthStore(backend=backend, auto_index=True)
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
