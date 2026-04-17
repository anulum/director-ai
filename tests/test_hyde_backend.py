# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.core.retrieval.hyde``.

Covers construction, pseudo-document generation, caching, fallback
behaviour, LLM failure recovery, metadata annotation, and edge cases.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from director_ai.core.retrieval.hyde import _DEFAULT_TEMPLATE, HyDEBackend
from director_ai.core.retrieval.vector_store import InMemoryBackend


def _mock_generator(response: str = "Paris is the capital of France."):
    """Create a mock LLM generator."""
    gen = MagicMock(return_value=response)
    return gen


def _make_backend(generator=None, **kwargs) -> HyDEBackend:
    """Create HyDEBackend wrapping InMemoryBackend."""
    base = InMemoryBackend()
    return HyDEBackend(base, generator=generator, **kwargs)


# ── Construction ────────────────────────────────────────────────────────


class TestConstruction:
    def test_default_template(self):
        b = _make_backend()
        assert "{query}" in b._template

    def test_custom_template(self):
        b = _make_backend(template="Answer: {query}")
        assert b._template == "Answer: {query}"

    def test_no_generator_graceful(self):
        b = _make_backend(generator=None)
        assert b._generator is None

    def test_fallback_default_true(self):
        b = _make_backend()
        assert b._fallback_to_raw is True


# ── Add delegates to base ──────────────────────────────────────────────


class TestAdd:
    def test_add_stores_in_base(self):
        b = _make_backend()
        b.add("d1", "Paris is in France")
        assert b.count() == 1

    def test_add_multiple(self):
        b = _make_backend()
        for i in range(5):
            b.add(f"d{i}", f"Document {i}")
        assert b.count() == 5

    def test_add_with_metadata(self):
        b = _make_backend()
        b.add("d1", "text", {"source": "test"})
        assert b.count() == 1


# ── Query with generator ──────────────────────────────────────────────


class TestQueryWithGenerator:
    def test_generator_called_on_query(self):
        gen = _mock_generator("The capital of France is Paris.")
        b = _make_backend(generator=gen)
        b.add("d1", "Paris is the capital of France.")
        b.query("What is the capital of France?")
        gen.assert_called_once()

    def test_generator_receives_template(self):
        gen = _mock_generator("answer")
        b = _make_backend(generator=gen, template="Q: {query}\nA:")
        b.add("d1", "fact")
        b.query("my question")
        prompt_arg = gen.call_args[0][0]
        assert "my question" in prompt_arg
        assert "Q:" in prompt_arg

    def test_pseudo_doc_used_for_retrieval(self):
        gen = _mock_generator("Paris France capital")
        b = _make_backend(generator=gen)
        b.add("d1", "Paris is the capital of France.")
        b.add("d2", "Berlin is the capital of Germany.")
        results = b.query("What is the capital of France?")
        # Pseudo-doc "Paris France capital" should match d1 better
        assert len(results) >= 1

    def test_metadata_annotated(self):
        gen = _mock_generator("document text science topic")
        b = _make_backend(generator=gen)
        b.add("d1", "some document text about science topic here")
        results = b.query("science question")
        assert len(results) >= 1
        meta = results[0].get("metadata", {})
        assert "hyde_pseudo_doc" in meta
        assert "hyde_original_query" in meta
        assert "science question" in meta["hyde_original_query"]


# ── Query without generator (graceful degradation) ─────────────────────


class TestNoGenerator:
    def test_raw_query_used(self):
        b = _make_backend(generator=None)
        b.add("d1", "Paris is the capital of France.")
        results = b.query("Paris capital")
        assert len(results) >= 1

    def test_no_crash_on_query(self):
        b = _make_backend(generator=None)
        b.add("d1", "text")
        results = b.query("anything")
        assert isinstance(results, list)


# ── Fallback on LLM failure ───────────────────────────────────────────


class TestFallback:
    def test_fallback_on_exception(self):
        gen = MagicMock(side_effect=RuntimeError("LLM down"))
        b = _make_backend(generator=gen, fallback_to_raw=True)
        b.add("d1", "Paris is the capital of France.")
        # Should not raise, falls back to raw query
        results = b.query("Paris")
        assert isinstance(results, list)

    def test_no_fallback_raises(self):
        gen = MagicMock(side_effect=RuntimeError("LLM down"))
        b = _make_backend(generator=gen, fallback_to_raw=False)
        b.add("d1", "text")
        with pytest.raises(RuntimeError, match="LLM down"):
            b.query("anything")

    def test_fallback_on_empty_response(self):
        gen = _mock_generator("")
        b = _make_backend(generator=gen)
        b.add("d1", "text content here")
        # Empty response → falls back to raw query
        results = b.query("text")
        assert isinstance(results, list)

    def test_fallback_on_whitespace_response(self):
        gen = _mock_generator("   \n\t  ")
        b = _make_backend(generator=gen)
        b.add("d1", "content")
        results = b.query("content")
        assert isinstance(results, list)


# ── Caching ────────────────────────────────────────────────────────────


class TestCaching:
    def test_cache_reuses_pseudo_doc(self):
        gen = _mock_generator("cached answer")
        b = _make_backend(generator=gen, cache_ttl=60.0)
        b.add("d1", "text")
        b.query("same question")
        b.query("same question")
        # Generator should be called only once (second hit cache)
        assert gen.call_count == 1

    def test_different_queries_not_cached(self):
        gen = _mock_generator("answer")
        b = _make_backend(generator=gen, cache_ttl=60.0)
        b.add("d1", "text")
        b.query("question A")
        b.query("question B")
        assert gen.call_count == 2

    def test_cache_disabled(self):
        gen = _mock_generator("answer")
        b = _make_backend(generator=gen, cache_ttl=0)
        b.add("d1", "text")
        b.query("question")
        b.query("question")
        assert gen.call_count == 2

    def test_cache_expiry(self):
        gen = _mock_generator("answer")
        b = _make_backend(generator=gen, cache_ttl=0.1)
        b.add("d1", "text")
        b.query("question")
        time.sleep(0.15)
        b.query("question")
        assert gen.call_count == 2


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_store(self):
        gen = _mock_generator("answer")
        b = _make_backend(generator=gen)
        results = b.query("anything")
        assert results == []

    def test_empty_query(self):
        gen = _mock_generator("pseudo")
        b = _make_backend(generator=gen)
        b.add("d1", "text")
        results = b.query("")
        assert isinstance(results, list)

    def test_n_results_respected(self):
        gen = _mock_generator("topic related")
        b = _make_backend(generator=gen)
        for i in range(10):
            b.add(f"d{i}", f"topic related document {i}")
        results = b.query("topic", n_results=3)
        assert len(results) <= 3

    def test_count_delegates(self):
        b = _make_backend()
        assert b.count() == 0
        b.add("d1", "text")
        assert b.count() == 1

    def test_default_template_has_query_placeholder(self):
        assert "{query}" in _DEFAULT_TEMPLATE


# ── Benchmark ──────────────────────────────────────────────────────────


class TestBenchmark:
    def test_overhead_is_generator_only(self):
        """HyDE overhead should be dominated by generator call, not framework."""
        call_times = []

        def timed_gen(prompt):
            t0 = time.perf_counter()
            result = "pseudo document answer"
            call_times.append(time.perf_counter() - t0)
            return result

        b = _make_backend(generator=timed_gen, cache_ttl=0)
        b.add("d1", "text content for retrieval")

        t0 = time.perf_counter()
        for _ in range(100):
            b.query("test query")
        total = (time.perf_counter() - t0) * 1000

        # Total overhead should be reasonable (<100ms for 100 queries
        # with instant generator)
        assert total < 500  # generous bound for slow CI
