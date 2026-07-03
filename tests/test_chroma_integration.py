# SPDX-License-Identifier: Apache-2.0
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

from __future__ import annotations

import math
import re
from importlib.util import find_spec
from typing import Final

import pytest

from director_ai.core.retrieval.vector_store import (
    ChromaBackend,
    VectorGroundTruthStore,
)

_CHROMA_AVAILABLE = find_spec("chromadb") is not None


_SEMANTIC_FEATURES: Final[tuple[tuple[str, ...], ...]] = (
    ("sky", "blue", "color"),
    ("scpn", "layer", "layers", "oscillator", "oscillators", "synchrony"),
    ("earth", "sun", "orbit", "orbits", "planetary"),
    ("water", "freeze", "freezes", "celsius"),
    ("restaurant", "italian", "food"),
    ("document", "documents", "topic", "sequential", "number"),
    ("kuramoto", "coupling", "strength"),
    ("paris", "france", "capital"),
)
_HASH_DIMENSIONS: Final[int] = 8
_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Return normalised lexical tokens for the deterministic test embedder."""
    return _TOKEN_PATTERN.findall(text.lower())


def _stable_index(token: str) -> int:
    """Map a token to a stable hash bucket without Python hash randomisation."""
    return sum((index + 1) * ord(char) for index, char in enumerate(token))


def _stable_chroma_embeddings(input: list[str]) -> list[list[float]]:
    """Return deterministic CPU-only embeddings for real Chroma integration tests."""
    vectors: list[list[float]] = []
    vector_size = len(_SEMANTIC_FEATURES) + _HASH_DIMENSIONS
    for text in input:
        vector = [0.0] * vector_size
        tokens = _tokens(text)
        for token in tokens:
            for feature_index, feature_terms in enumerate(_SEMANTIC_FEATURES):
                if token in feature_terms:
                    vector[feature_index] += 1.0
            hash_slot = len(_SEMANTIC_FEATURES) + (
                _stable_index(token) % _HASH_DIMENSIONS
            )
            vector[hash_slot] += 0.1
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vectors.append(vector)
        else:
            vectors.append([value / norm for value in vector])
    return vectors


@pytest.mark.integration
@pytest.mark.skipif(not _CHROMA_AVAILABLE, reason="chromadb not installed")
class TestChromaIntegration:
    """Integration tests for ChromaBackend with in-memory Chroma."""

    @pytest.fixture
    def backend(self) -> ChromaBackend:
        """Fresh in-memory Chroma backend for each test."""
        return ChromaBackend(
            collection_name=f"test_{id(self)}",
            persist_directory=None,  # In-memory
            embedding_function=_stable_chroma_embeddings,
        )

    def test_add_and_count(self, backend: ChromaBackend) -> None:
        """Chroma increments collection count after adding one document."""
        assert backend.count() == 0
        backend.add("doc1", "The sky is blue")
        assert backend.count() == 1

    def test_add_multiple(self, backend: ChromaBackend) -> None:
        """Chroma stores multiple independently identified documents."""
        backend.add("d1", "SCPN Layer 1: Quantum Biological")
        backend.add("d2", "SCPN Layer 2: Neurochemical")
        backend.add("d3", "SCPN Layer 7: Geometrical-Symbolic")
        assert backend.count() == 3

    def test_query_returns_results(self, backend: ChromaBackend) -> None:
        """A query returns at least one semantically relevant document."""
        backend.add("d1", "The Earth orbits the Sun")
        backend.add("d2", "Water freezes at zero degrees Celsius")
        backend.add("d3", "SCPN couples 16 oscillator layers")
        results = backend.query("planetary orbits", n_results=2)
        assert len(results) >= 1
        assert any("Earth" in r["text"] for r in results)

    def test_query_with_metadata(self, backend: ChromaBackend) -> None:
        """Chroma query results preserve metadata from indexed documents."""
        backend.add(
            "d1",
            "Kuramoto coupling K=0.302",
            metadata={"layer": "L1-L2"},
        )
        results = backend.query("coupling strength")
        assert len(results) >= 1
        assert results[0]["metadata"].get("layer") == "L1-L2"

    def test_query_empty_collection(self, backend: ChromaBackend) -> None:
        """Querying an empty Chroma collection returns an empty list."""
        results = backend.query("anything")
        assert results == []

    def test_query_n_results(self, backend: ChromaBackend) -> None:
        """Chroma respects the requested result limit for populated collections."""
        for i in range(10):
            backend.add(f"d{i}", f"Document number {i}")
        results = backend.query("document", n_results=3)
        assert len(results) == 3

    def test_vector_store_with_chroma(self, backend: ChromaBackend) -> None:
        """VectorGroundTruthStore works with ChromaBackend."""
        store = VectorGroundTruthStore(backend=backend)
        store.ingest(["sky color is blue", "SCPN has 16 layers"])
        assert backend.count() > 0
        ctx = store.retrieve_context("What color is the sky?")
        assert ctx is not None

    def test_vector_store_add_fact(self, backend: ChromaBackend) -> None:
        """VectorGroundTruthStore.add_fact writes through to Chroma."""
        store = VectorGroundTruthStore(backend=backend)
        initial = backend.count()
        store.add_fact("omega_1", "1.329 rad/s")
        assert backend.count() == initial + 1

    def test_semantic_similarity(self, backend: ChromaBackend) -> None:
        """Semantically similar queries retrieve relevant docs."""
        backend.add("d1", "Coherence measures alignment of oscillators")
        backend.add("d2", "The restaurant serves Italian food")
        backend.add("d3", "Phase synchronisation in neural networks")
        results = backend.query("oscillator synchrony", n_results=2)
        texts = [r["text"] for r in results]
        # At least one result should be about oscillators/synchrony
        assert any("oscillat" in t.lower() or "synchron" in t.lower() for t in texts)

    @pytest.mark.parametrize("n_results", [1, 2, 3, 5])
    def test_parametrised_n_results(
        self,
        backend: ChromaBackend,
        n_results: int,
    ) -> None:
        """Different requested result limits cap returned Chroma rows."""
        for i in range(10):
            backend.add(f"d{i}", f"Document about topic {i}")
        results = backend.query("topic", n_results=n_results)
        assert len(results) <= n_results

    def test_scorer_pipeline_with_chroma(self, backend: ChromaBackend) -> None:
        """Full pipeline: ChromaBackend → VectorGroundTruthStore → CoherenceScorer."""
        from director_ai.core import CoherenceScorer

        store = VectorGroundTruthStore(backend=backend)
        store.ingest(["Paris is the capital of France"])
        scorer = CoherenceScorer(use_nli=False, ground_truth_store=store)
        approved, score = scorer.review("capital of France", "Paris")
        assert isinstance(approved, bool)
        assert 0.0 <= score.score <= 1.0

    def test_sequential_adds_keep_documents_queryable(
        self,
        backend: ChromaBackend,
    ) -> None:
        """Sequential Chroma writes preserve all IDs and query visibility."""
        for i in range(5):
            backend.add(f"seq{i}", f"Sequential Chroma document {i}")

        assert backend.count() == 5
        results = backend.query("Sequential Chroma document", n_results=5)
        assert {result["id"] for result in results} == {
            "seq0",
            "seq1",
            "seq2",
            "seq3",
            "seq4",
        }
