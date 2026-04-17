# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.core.retrieval.parent_child``.

Covers construction, add/query round-trip, deduplication,
parent-returned-not-child, edge cases, and thread safety.
"""

from __future__ import annotations

from director_ai.core.retrieval.parent_child import ParentChildBackend, _chunk_id
from director_ai.core.retrieval.vector_store import InMemoryBackend


def _make_backend(**kwargs) -> ParentChildBackend:
    """Create a ParentChildBackend wrapping InMemoryBackend."""
    return ParentChildBackend(InMemoryBackend(), **kwargs)


# ── Chunk ID generation ────────────────────────────────────────────────


class TestChunkId:
    def test_parent_id(self):
        assert _chunk_id("doc1", 0, "parent") == "doc1::parent-0"

    def test_child_id(self):
        assert _chunk_id("doc1", 2, "child-0") == "doc1::child-0-2"

    def test_different_docs(self):
        assert _chunk_id("a", 0, "parent") != _chunk_id("b", 0, "parent")


# ── Construction ─��──────────────────────────────────────────────────────


class TestConstruction:
    def test_default_sizes(self):
        b = _make_backend()
        assert b._parent_size == 2048
        assert b._child_size == 256

    def test_custom_sizes(self):
        b = _make_backend(parent_size=1024, child_size=128)
        assert b._parent_size == 1024
        assert b._child_size == 128

    def test_starts_empty(self):
        b = _make_backend()
        assert b.count() == 0
        assert b.parent_count == 0


# ── Add and query ───────────────────────────────────────────────────────


class TestAddAndQuery:
    def test_short_text_single_parent(self):
        """Text shorter than parent_size → one parent, children indexed."""
        b = _make_backend(parent_size=500, child_size=100)
        b.add("d1", "A" * 200)
        assert b.parent_count == 1
        assert b.count() >= 1  # at least one child

    def test_long_text_multiple_parents(self):
        """Text longer than parent_size → multiple parents."""
        b = _make_backend(parent_size=100, child_size=30, parent_overlap=10)
        b.add("d1", "word " * 100)  # ~500 chars
        assert b.parent_count > 1

    def test_query_returns_parent_text(self):
        """Query should return the parent chunk, not the matched child."""
        b = _make_backend(parent_size=500, child_size=50)
        parent_text = "The capital of France is Paris. " * 10  # ~310 chars
        b.add("geo", parent_text)

        results = b.query("Paris capital")
        assert len(results) >= 1
        # Parent text should be longer than child_size
        assert len(results[0]["text"]) > 50

    def test_query_metadata_has_child_info(self):
        """Result metadata should include matched child details."""
        b = _make_backend(parent_size=500, child_size=50)
        b.add("d1", "The quick brown fox jumps over the lazy dog. " * 10)

        results = b.query("fox jumps")
        assert len(results) >= 1
        meta = results[0].get("metadata", {})
        assert "matched_child_id" in meta
        assert "parent_id" in meta

    def test_query_with_metadata(self):
        """Custom metadata should be preserved through add/query."""
        b = _make_backend(parent_size=500, child_size=50)
        b.add("d1", "Temperature is 25 degrees. " * 10, {"source": "sensor"})

        results = b.query("temperature")
        assert len(results) >= 1
        meta = results[0].get("metadata", {})
        assert meta.get("source") == "sensor"


# ── Deduplication ───────────────────────────────────────────────────────


class TestDeduplication:
    def test_no_duplicate_parents(self):
        """Multiple child matches from same parent → one result."""
        b = _make_backend(parent_size=500, child_size=30)
        # Long text with repeated keyword → many child matches from same parent
        b.add("d1", "Paris is great. Paris is beautiful. Paris is historic. " * 5)

        results = b.query("Paris", n_results=5)
        parent_ids = [r["id"] for r in results]
        assert len(parent_ids) == len(set(parent_ids))  # all unique

    def test_multiple_docs_distinct_parents(self):
        """Results from different documents should not be deduplicated."""
        b = _make_backend(parent_size=500, child_size=50)
        b.add("doc_a", "Water boils at 100 degrees Celsius. " * 8)
        b.add("doc_b", "Water freezes at 0 degrees Celsius. " * 8)

        results = b.query("water temperature", n_results=2)
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert len(ids) == 2  # two distinct parents


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_text(self):
        """Empty text should not crash."""
        b = _make_backend()
        b.add("empty", "")
        assert b.parent_count == 0

    def test_text_shorter_than_child(self):
        """Very short text → one parent, one child."""
        b = _make_backend(parent_size=1000, child_size=500)
        b.add("tiny", "Hello world")
        assert b.parent_count == 1
        assert b.count() == 1

    def test_query_empty_store(self):
        """Query on empty store should return empty list."""
        b = _make_backend()
        results = b.query("anything")
        assert results == []

    def test_n_results_respected(self):
        """Should not return more than n_results."""
        b = _make_backend(parent_size=100, child_size=30)
        for i in range(10):
            b.add(f"d{i}", f"Document {i} about topic {i}. " * 5)

        results = b.query("topic", n_results=3)
        assert len(results) <= 3

    def test_distance_preserved(self):
        """Distance from child match should be carried to parent result."""
        b = _make_backend(parent_size=500, child_size=50)
        b.add("d1", "The sky is blue and clear today. " * 8)
        results = b.query("sky blue")
        if results:
            assert "distance" in results[0]
            assert isinstance(results[0]["distance"], (int, float))


# ── Thread safety ───────────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_add(self):
        """Concurrent adds should not corrupt state."""
        import threading

        b = _make_backend(parent_size=200, child_size=50)
        errors: list[Exception] = []

        def add_docs(start: int) -> None:
            try:
                for i in range(start, start + 20):
                    b.add(f"doc_{i}", f"Content for document {i}. " * 10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=add_docs, args=(i * 20,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert b.parent_count > 0
        assert b.count() > 0

    def test_concurrent_query(self):
        """Concurrent queries should not crash."""
        import threading

        b = _make_backend(parent_size=200, child_size=50)
        for i in range(10):
            b.add(f"doc_{i}", f"Fact number {i} about science. " * 10)

        errors: list[Exception] = []
        results_collected: list[list] = []

        def query_fn() -> None:
            try:
                r = b.query("science", n_results=3)
                results_collected.append(r)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=query_fn) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results_collected) == 8
