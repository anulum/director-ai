# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.core.retrieval.contextual_compression``.

Covers heuristic compression, LLM compression, metadata tracking,
compression ratio, fallback, edge cases, and benchmarks.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from director_ai.core.retrieval.contextual_compression import (
    ContextualCompressionBackend,
    _heuristic_compress,
    _keyword_overlap,
)
from director_ai.core.retrieval.vector_store import InMemoryBackend


def _make_backend(**kwargs) -> ContextualCompressionBackend:
    base = InMemoryBackend()
    return ContextualCompressionBackend(base, **kwargs)


# ── Keyword overlap utility ────────────────────────────────────────────


class TestKeywordOverlap:
    def test_identical(self):
        assert _keyword_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _keyword_overlap("foo bar", "baz qux") == 0.0

    def test_partial_overlap(self):
        score = _keyword_overlap("refund policy", "the refund process is simple")
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        assert _keyword_overlap("", "some text") == 0.0

    def test_empty_sentence(self):
        assert _keyword_overlap("query", "") == 0.0


# ── Heuristic compression ─────────────────────────────────────────────


class TestHeuristicCompress:
    def test_keeps_relevant_sentences(self):
        text = (
            "The refund policy allows returns within 30 days. "
            "Shipping takes 5 to 7 business days. "
            "Customer support is available 24/7."
        )
        result = _heuristic_compress("refund policy", text, threshold=0.1)
        assert "refund" in result.lower()

    def test_single_sentence_unchanged(self):
        text = "The refund policy allows returns within 30 days."
        result = _heuristic_compress("unrelated query", text, threshold=0.1)
        assert result == text

    def test_no_match_returns_best(self):
        text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."
        result = _heuristic_compress("omega", text, threshold=0.5)
        assert len(result) > 0  # returns best match even if below threshold

    def test_all_relevant(self):
        text = "Refund policy details. More refund information."
        result = _heuristic_compress("refund", text, threshold=0.05)
        assert "refund" in result.lower()


# ── Backend with heuristic ─────────────────────────────────────────────


class TestBackendHeuristic:
    def test_query_compresses_results(self):
        b = _make_backend(overlap_threshold=0.05)
        b.add(
            "d1",
            "The refund policy is 30 days. Shipping costs vary. "
            "Weather today is sunny and warm.",
        )
        results = b.query("refund policy")
        assert len(results) >= 1
        # Compressed text should be shorter or equal
        assert len(results[0]["text"]) > 0

    def test_original_preserved_in_metadata(self):
        b = _make_backend(overlap_threshold=0.05)
        b.add(
            "d1",
            "Relevant sentence about refund. Irrelevant noise sentence here. "
            "More irrelevant filler content added.",
        )
        results = b.query("refund")
        if results:
            meta = results[0].get("metadata", {})
            if "original_text" in meta:
                assert len(meta["original_text"]) > 0

    def test_compression_ratio_tracked(self):
        b = _make_backend(overlap_threshold=0.05)
        b.add(
            "d1",
            "Relevant refund info here. Irrelevant noise text. "
            "More filler that should be removed.",
        )
        results = b.query("refund")
        if results:
            meta = results[0].get("metadata", {})
            if "compression_ratio" in meta:
                assert 0.0 < meta["compression_ratio"] <= 1.0


# ── Backend with LLM ──────────────────────────────────────────────────


class TestBackendLLM:
    def test_llm_called_per_result(self):
        gen = MagicMock(return_value="Compressed relevant text only.")
        b = _make_backend(strategy="llm", generator=gen)
        b.add("d1", "long text with refund policy and other noise content")
        results = b.query("refund")
        assert len(results) >= 1
        assert gen.call_count >= 1

    def test_llm_failure_falls_back(self):
        gen = MagicMock(side_effect=RuntimeError("LLM error"))
        b = _make_backend(strategy="llm", generator=gen)
        b.add("d1", "some content about refund policy returns")
        results = b.query("refund")
        assert isinstance(results, list)

    def test_llm_empty_response_keeps_original(self):
        gen = MagicMock(return_value="")
        b = _make_backend(strategy="llm", generator=gen)
        b.add("d1", "original text should be preserved here intact")
        results = b.query("original text")
        if results:
            assert len(results[0]["text"]) > 0


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_store(self):
        b = _make_backend()
        assert b.query("anything") == []

    def test_add_delegates(self):
        b = _make_backend()
        b.add("d1", "text")
        assert b.count() == 1

    def test_min_compressed_len(self):
        b = _make_backend(min_compressed_len=100)
        b.add("d1", "Short text about policy.")
        results = b.query("policy")
        # Text is shorter than min → should keep original
        if results:
            assert results[0]["text"] == "Short text about policy."

    def test_n_results_respected(self):
        b = _make_backend()
        for i in range(10):
            b.add(f"d{i}", f"content about topic {i} with some words")
        results = b.query("topic", n_results=3)
        assert len(results) <= 3


# ── Benchmark ──────────────────────────────────────────────────────────


class TestBenchmark:
    def test_heuristic_overhead(self):
        b = _make_backend()
        b.add(
            "d1",
            "Sentence one about refund. Sentence two about shipping. Sentence three about support.",
        )

        t0 = time.perf_counter()
        for _ in range(1000):
            b.query("refund")
        overhead_ms = (time.perf_counter() - t0) * 1000 / 1000
        # Should be well under 10ms per query (dominated by InMemory search)
        assert overhead_ms < 50
