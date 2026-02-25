# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Vector Store Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.vector_store import (
    InMemoryBackend,
    VectorGroundTruthStore,
)


@pytest.mark.consumer
class TestInMemoryBackend:
    def test_add_and_count(self):
        backend = InMemoryBackend()
        assert backend.count() == 0
        backend.add("doc1", "The sky is blue")
        assert backend.count() == 1

    def test_query_returns_relevant(self):
        backend = InMemoryBackend()
        backend.add("doc1", "The sky is blue")
        backend.add("doc2", "Water is wet")
        backend.add("doc3", "Fire is hot")
        results = backend.query("What color is the sky?", n_results=2)
        assert len(results) > 0
        assert any("sky" in r["text"].lower() for r in results)

    def test_query_empty_store(self):
        backend = InMemoryBackend()
        results = backend.query("anything")
        assert results == []


@pytest.mark.consumer
class TestVectorGroundTruthStore:
    def test_auto_index_builtin_facts(self):
        store = VectorGroundTruthStore()
        assert store.backend.count() == len(store.facts)

    def test_retrieve_context_vector(self):
        store = VectorGroundTruthStore()
        context = store.retrieve_context("How many layers in SCPN?")
        assert context is not None
        assert "16" in context

    def test_retrieve_context_sky_color(self):
        store = VectorGroundTruthStore()
        context = store.retrieve_context("What color is the sky?")
        assert context is not None
        assert "blue" in context.lower()

    def test_add_custom_fact(self):
        store = VectorGroundTruthStore()
        initial_count = store.backend.count()
        store.add_fact("gravity", "9.81 m/s²")
        assert store.backend.count() == initial_count + 1
        assert "gravity" in store.facts

    def test_retrieve_custom_fact(self):
        store = VectorGroundTruthStore()
        store.add_fact("planck constant", "6.626e-34 J·s")
        context = store.retrieve_context("What is the planck constant?")
        assert context is not None

    def test_keyword_fallback(self):
        """If vector search fails, keyword matching should still work."""
        store = VectorGroundTruthStore(backend=InMemoryBackend(), auto_index=False)
        # No vector-indexed facts, but keyword store still has them
        context = store.retrieve_context("sky color")
        assert context is not None
