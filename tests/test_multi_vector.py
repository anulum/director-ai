# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.core.retrieval.multi_vector``.

Covers representation generation, multi-rep indexing, deduplication,
original text return, LLM summary, edge cases, and benchmarks.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from director_ai.core.retrieval.multi_vector import (
    MultiVectorBackend,
    _extract_summary,
    _extract_title,
)
from director_ai.core.retrieval.vector_store import InMemoryBackend


def _make_backend(**kwargs) -> MultiVectorBackend:
    return MultiVectorBackend(InMemoryBackend(), **kwargs)


# ── Title extraction ───────────────────────────────────────────────────


class TestTitleExtraction:
    def test_first_line(self):
        title = _extract_title("Refund Policy\nDetails about refunds.")
        assert "Refund Policy" in title

    def test_first_sentence(self):
        title = _extract_title("The refund policy is simple. More details follow.")
        assert "refund" in title.lower()

    def test_long_text_truncated(self):
        title = _extract_title("A" * 500)
        assert len(title) <= 200

    def test_empty_text(self):
        title = _extract_title("")
        assert isinstance(title, str)


# ── Summary extraction ─────────────────────────────────────────────────


class TestSummaryExtraction:
    def test_first_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth."
        summary = _extract_summary(text, max_sentences=2)
        assert "First" in summary
        assert "Second" in summary

    def test_fewer_sentences_than_max(self):
        summary = _extract_summary("Only one sentence.", max_sentences=5)
        assert summary == "Only one sentence."


# ── Construction ───────────────────────────────────────────────────────


class TestConstruction:
    def test_default_representations(self):
        b = _make_backend()
        assert set(b._representations) == {"content", "summary", "title"}

    def test_custom_representations(self):
        b = _make_backend(representations=["content", "title"])
        assert "summary" not in b._representations

    def test_starts_empty(self):
        b = _make_backend()
        assert b.count() == 0
        assert b.document_count == 0


# ── Add with multiple representations ─────────────────────────────────


class TestAdd:
    def test_three_reps_indexed(self):
        b = _make_backend()
        b.add("d1", "The refund policy allows returns within 30 days.")
        assert b.count() == 3  # content + summary + title
        assert b.document_count == 1

    def test_content_only(self):
        b = _make_backend(representations=["content"])
        b.add("d1", "Some text.")
        assert b.count() == 1

    def test_multiple_docs(self):
        b = _make_backend()
        b.add("d1", "First document about refunds.")
        b.add("d2", "Second document about shipping.")
        assert b.document_count == 2
        assert b.count() == 6  # 2 docs × 3 reps

    def test_metadata_preserved(self):
        b = _make_backend(representations=["content"])
        b.add("d1", "text", {"source": "test"})
        results = b._base.query("text", n_results=1)
        assert results[0]["metadata"]["source"] == "test"
        assert results[0]["metadata"]["representation"] == "content"


# ── Query deduplication ────────────────────────────────────────────────


class TestQuery:
    def test_deduplicates_by_doc(self):
        b = _make_backend()
        b.add("d1", "The refund policy is 30 days for all purchases.")
        results = b.query("refund policy")
        # Should get 1 result (deduplicated from 3 representations)
        doc_ids = [r.get("metadata", {}).get("doc_id", r["id"]) for r in results]
        assert len(set(doc_ids)) == len(doc_ids)

    def test_returns_original_text(self):
        b = _make_backend()
        original = (
            "The full original document text about refund policies and procedures."
        )
        b.add("d1", original)
        results = b.query("refund")
        assert len(results) >= 1
        # Should return original text, not summary or title
        assert results[0]["text"] == original

    def test_n_results_respected(self):
        b = _make_backend()
        for i in range(10):
            b.add(f"d{i}", f"Document {i} about topic {i} and related content.")
        results = b.query("topic", n_results=3)
        assert len(results) <= 3

    def test_empty_store(self):
        b = _make_backend()
        assert b.query("anything") == []


# ── LLM summary generator ─────────────────────────────────────────────


class TestLLMSummary:
    def test_llm_used(self):
        gen = MagicMock(return_value="LLM generated summary.")
        b = _make_backend(summary_generator=gen)
        b.add("d1", "Long text. " * 20)
        gen.assert_called_once()

    def test_llm_failure_falls_back(self):
        gen = MagicMock(side_effect=RuntimeError("fail"))
        b = _make_backend(summary_generator=gen)
        # Should not raise — falls back to heuristic
        b.add("d1", "Sentence one. Sentence two. Sentence three.")
        assert b.count() == 3


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_text(self):
        b = _make_backend()
        b.add("d1", "")
        assert b.document_count == 1

    def test_very_short_text(self):
        b = _make_backend()
        b.add("d1", "Hi.")
        assert b.count() == 3

    def test_count_includes_all_reps(self):
        b = _make_backend()
        b.add("d1", "text")
        b.add("d2", "text")
        assert b.count() == 6


# ── Benchmark ──────────────────────────────────────────────────────────


class TestBenchmark:
    def test_add_overhead(self):
        b = _make_backend()
        t0 = time.perf_counter()
        for i in range(100):
            b.add(f"d{i}", f"Document {i} about topic {i}. " * 5)
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 500  # generous bound

    def test_query_overhead(self):
        b = _make_backend()
        for i in range(50):
            b.add(f"d{i}", f"Document about topic {i} with details.")
        t0 = time.perf_counter()
        for _ in range(100):
            b.query("topic details", n_results=3)
        ms = (time.perf_counter() - t0) * 1000 / 100
        assert ms < 50  # per-query
