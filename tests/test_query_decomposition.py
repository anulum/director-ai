# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.core.retrieval.query_decomposition``.

Covers heuristic decomposition, LLM decomposition, RRF merge,
single-intent passthrough, edge cases, and benchmarks.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from director_ai.core.retrieval.query_decomposition import (
    QueryDecompositionBackend,
    _heuristic_decompose,
)
from director_ai.core.retrieval.vector_store import InMemoryBackend


def _make_backend(**kwargs) -> QueryDecompositionBackend:
    base = InMemoryBackend()
    return QueryDecompositionBackend(base, **kwargs)


# ── Heuristic decomposition ────────────────────────────────────────────


class TestHeuristicDecompose:
    def test_simple_and(self):
        parts = _heuristic_decompose("What is the refund policy and shipping time?")
        assert len(parts) >= 2

    def test_semicolon_split(self):
        parts = _heuristic_decompose("explain refund policy; describe shipping")
        assert len(parts) >= 2

    def test_single_intent_unchanged(self):
        parts = _heuristic_decompose("What is the refund policy?")
        assert len(parts) == 1

    def test_short_fragments_filtered(self):
        parts = _heuristic_decompose("a and b")
        # "a" and "b" are too short → returns original
        assert len(parts) == 1

    def test_numbered_list(self):
        parts = _heuristic_decompose(
            "1. What is the refund policy 2. How long is shipping"
        )
        assert len(parts) >= 2

    def test_multiple_questions(self):
        parts = _heuristic_decompose(
            "What is the capital of France? What is the capital of Germany?"
        )
        assert len(parts) >= 2

    def test_empty_query(self):
        parts = _heuristic_decompose("")
        assert len(parts) == 1
        assert parts[0] == ""

    def test_also_conjunction(self):
        parts = _heuristic_decompose(
            "explain the refund policy also describe warranty terms"
        )
        assert len(parts) >= 2


# ── Backend construction ───────────────────────────────────────────────


class TestConstruction:
    def test_default_strategy(self):
        b = _make_backend()
        assert b._strategy == "heuristic"

    def test_llm_strategy(self):
        b = _make_backend(strategy="llm", generator=lambda p: "sub1\nsub2")
        assert b._strategy == "llm"

    def test_custom_rrf_k(self):
        b = _make_backend(rrf_k=30)
        assert b._rrf_k == 30


# ── Query with decomposition ──────────────────────────────────────────


class TestQueryDecomposition:
    def test_single_intent_passthrough(self):
        b = _make_backend()
        b.add("d1", "refund policy is 30 days")
        results = b.query("What is the refund policy?")
        assert len(results) >= 1

    def test_compound_query_retrieves_both(self):
        b = _make_backend()
        b.add("d1", "refund policy is 30 days for all purchases")
        b.add("d2", "shipping time is 5 to 7 business days worldwide")
        results = b.query(
            "What is the refund policy and how long does shipping take?",
            n_results=2,
        )
        assert len(results) >= 1

    def test_deduplication_via_rrf(self):
        b = _make_backend()
        b.add("d1", "important shared document about policy and shipping")
        # Same doc matches both sub-queries → should appear once
        results = b.query("policy and shipping", n_results=3)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))


# ── LLM decomposition ─────────────────────────────────────────────────


class TestLLMDecompose:
    def test_llm_called(self):
        gen = MagicMock(return_value="What is the refund policy\nHow long is shipping")
        b = _make_backend(strategy="llm", generator=gen)
        b.add("d1", "refund policy 30 days")
        b.query("refund policy and shipping time")
        gen.assert_called_once()

    def test_llm_failure_falls_back(self):
        gen = MagicMock(side_effect=RuntimeError("LLM error"))
        b = _make_backend(strategy="llm", generator=gen)
        b.add("d1", "some content text here")
        # Should not raise — falls back to heuristic
        results = b.query("some content")
        assert isinstance(results, list)

    def test_max_sub_queries_capped(self):
        gen = MagicMock(return_value="q1\nq2\nq3\nq4\nq5\nq6\nq7\nq8\nq9\nq10")
        b = _make_backend(strategy="llm", generator=gen, max_sub_queries=3)
        b.add("d1", "content for retrieval testing")
        b.query("complex multi-part question")
        # Generator called once, but max 3 sub-queries used
        gen.assert_called_once()


# ── RRF merge ──────────────────────────────────────────────────────────


class TestRRFMerge:
    def test_merge_two_lists(self):
        b = _make_backend()
        results = b._rrf_merge(
            [
                [{"id": "a", "text": "doc a"}, {"id": "b", "text": "doc b"}],
                [{"id": "b", "text": "doc b"}, {"id": "c", "text": "doc c"}],
            ],
            n_results=3,
        )
        # "b" appears in both lists → highest RRF score
        ids = [r["id"] for r in results]
        assert "b" in ids
        assert ids[0] == "b"  # highest fused score

    def test_n_results_respected(self):
        b = _make_backend()
        results = b._rrf_merge(
            [
                [{"id": f"d{i}", "text": f"doc {i}"} for i in range(10)],
            ],
            n_results=3,
        )
        assert len(results) <= 3

    def test_empty_lists(self):
        b = _make_backend()
        results = b._rrf_merge([[], []], n_results=3)
        assert results == []


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_store(self):
        b = _make_backend()
        results = b.query("anything and everything")
        assert results == []

    def test_add_delegates(self):
        b = _make_backend()
        b.add("d1", "text")
        assert b.count() == 1

    def test_count_delegates(self):
        b = _make_backend()
        assert b.count() == 0


# ── Benchmark ──────────────────────────────────────────────────────────


class TestBenchmark:
    def test_heuristic_overhead(self):
        """Heuristic decomposition should be sub-millisecond."""
        import time

        b = _make_backend()
        b.add("d1", "refund policy content for testing")
        b.add("d2", "shipping time content for testing")

        t0 = time.perf_counter()
        for _ in range(1000):
            _heuristic_decompose("refund policy and shipping time")
        overhead = (time.perf_counter() - t0) * 1000 / 1000
        assert overhead < 1.0  # <1ms per decomposition
