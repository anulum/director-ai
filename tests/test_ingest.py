# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — VectorGroundTruthStore Ingest Tests
"""Multi-angle tests for document ingestion pipeline.

Covers: text list ingestion, empty input, retrieval after ingest,
add_fact interface, duplicate handling, batch sizes, Unicode content,
empty backend, capacity, and pipeline performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore


@pytest.fixture
def store():
    return VectorGroundTruthStore(backend=InMemoryBackend())


# ── Basic ingestion ─────────────────────────────────────────────────


class TestIngestBasic:
    """Core ingestion functionality."""

    def test_ingest_text_list(self, store):
        count = store.ingest(["fact one", "fact two", "fact three"])
        assert count == 3
        assert store.backend.count() >= 3

    def test_ingest_single_item(self, store):
        count = store.ingest(["single fact"])
        assert count == 1

    def test_ingest_returns_count(self, store):
        count = store.ingest(["a", "b"])
        assert isinstance(count, int)
        assert count == 2


# ── Empty / edge inputs ───────────────────────────────────────────


class TestIngestEdgeCases:
    """Ingestion must handle edge cases gracefully."""

    def test_ingest_empty_list(self, store):
        initial_count = store.backend.count()
        count = store.ingest([])
        assert count == 0
        assert store.backend.count() == initial_count

    @pytest.mark.parametrize(
        "docs",
        [
            [""],
            ["   "],
            ["\n\t"],
        ],
    )
    def test_ingest_whitespace_docs(self, store, docs):
        count = store.ingest(docs)
        assert isinstance(count, int)

    def test_ingest_unicode_docs(self, store):
        count = store.ingest(["日本語テスト", "العربية", "한국어"])
        assert count == 3

    def test_ingest_very_long_doc(self, store):
        long_doc = "x " * 100_000
        count = store.ingest([long_doc])
        assert count == 1

    def test_ingest_mixed_content(self, store):
        count = store.ingest(
            [
                "Short fact",
                "Medium fact with more detail about the topic at hand",
                "x " * 10_000,
            ]
        )
        assert count == 3


# ── Retrieval after ingest ─────────────────────────────────────────


class TestIngestRetrieval:
    """Verify retrieval works correctly after ingestion."""

    def test_ingest_retrieves(self, store):
        store.ingest(["The refund policy is 30 days from purchase."])
        result = store.retrieve_context("refund policy")
        assert result is not None
        assert "refund" in result.lower()

    def test_add_fact_retrieves(self, store):
        store.add_fact("capital", "Paris is the capital of France")
        result = store.retrieve_context("capital of France")
        assert result is not None
        assert "Paris" in result

    def test_retrieve_from_empty_backend(self):
        backend = InMemoryBackend()
        store = VectorGroundTruthStore(backend=backend)
        store.facts.clear()
        result = store.retrieve_context("anything")
        assert result is None

    def test_multi_fact_ingest_and_count(self, store):
        store.ingest(
            [
                "Refunds within 30 days of purchase",
                "Free shipping on orders over $50",
                "1 year warranty on all products",
            ]
        )
        assert store.backend.count() >= 3

    def test_retrieve_after_multi_ingest(self, store):
        store.ingest(["Paris is the capital of France"])
        result = store.retrieve_context("capital of France")
        assert result is not None
        assert "Paris" in result


# ── Batch ingestion ────────────────────────────────────────────────


class TestBatchIngestion:
    """Test various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 5, 10, 50, 100])
    def test_various_batch_sizes(self, batch_size):
        store = VectorGroundTruthStore(backend=InMemoryBackend())
        docs = [f"Fact number {i}" for i in range(batch_size)]
        count = store.ingest(docs)
        assert count == batch_size


# ── Pipeline performance ──────────────────────────────────────────


class TestIngestPerformance:
    """Document ingestion performance characteristics."""

    def test_backend_count_tracks_ingestion(self, store):
        assert store.backend.count() == 0
        store.ingest(["A", "B", "C"])
        assert store.backend.count() >= 3

    def test_ingest_returns_int(self, store):
        result = store.ingest(["test"])
        assert isinstance(result, int)
