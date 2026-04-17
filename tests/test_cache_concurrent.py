# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Cache Concurrent Stress Tests
"""Multi-angle tests for ScoreCache thread safety and LRU correctness.

Covers: concurrent put/get, concurrent invalidation, LRU eviction order,
zero-size cache, TTL expiry, parametrised max sizes, cache hit/miss,
pipeline integration with scorer, and performance documentation.
"""

from __future__ import annotations

import threading
import time

import pytest

from director_ai.core.cache import ScoreCache


class TestCacheConcurrency:
    def test_concurrent_put_get(self):
        cache = ScoreCache(max_size=100, ttl_seconds=60.0)
        errors: list[str] = []

        def writer(tid: int):
            for i in range(50):
                cache.put(f"q_{tid}_{i}", f"p_{tid}", 0.5 + i * 0.01, 0.1, 0.2)

        def reader(tid: int):
            for i in range(50):
                entry = cache.get(f"q_{tid}_{i}", f"p_{tid}")
                if entry is not None and not (0.0 <= entry.score <= 1.5):
                    errors.append(f"bad score {entry.score}")

        threads = []
        for t in range(4):
            threads.append(threading.Thread(target=writer, args=(t,)))
            threads.append(threading.Thread(target=reader, args=(t,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent errors: {errors}"
        assert cache.size <= 100

    def test_concurrent_put_invalidate(self):
        cache = ScoreCache(max_size=50, ttl_seconds=60.0)
        barrier = threading.Barrier(3)

        def writer():
            barrier.wait()
            for i in range(100):
                cache.put(f"q_{i}", "p", 0.8, 0.1, 0.1)

        def invalidator():
            barrier.wait()
            for _ in range(10):
                cache.invalidate()

        def reader():
            barrier.wait()
            for i in range(100):
                cache.get(f"q_{i}", "p")

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=invalidator),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No crash = pass; verify invariant
        assert cache.size <= 50

    def test_lru_ordering(self):
        cache = ScoreCache(max_size=3, ttl_seconds=60.0)
        cache.put("a", "p", 0.9, 0.1, 0.1)
        cache.put("b", "p", 0.8, 0.1, 0.1)
        cache.put("c", "p", 0.7, 0.1, 0.1)

        # Access 'a' to promote it
        assert cache.get("a", "p") is not None

        # Insert 'd' — should evict 'b' (oldest untouched), not 'a'
        cache.put("d", "p", 0.6, 0.1, 0.1)

        assert cache.get("a", "p") is not None, "LRU should have kept 'a'"
        assert cache.get("b", "p") is None, "LRU should have evicted 'b'"
        assert cache.get("c", "p") is not None
        assert cache.get("d", "p") is not None

    def test_zero_max_size(self):
        cache = ScoreCache(max_size=0, ttl_seconds=60.0)
        cache.put("q", "p", 0.9, 0.1, 0.1)
        assert cache.size == 0
        assert cache.get("q", "p") is None

    @pytest.mark.parametrize("max_size", [1, 5, 10, 50, 100])
    def test_parametrised_max_size(self, max_size):
        cache = ScoreCache(max_size=max_size, ttl_seconds=60.0)
        for i in range(max_size + 10):
            cache.put(f"q{i}", "p", 0.5, 0.1, 0.1)
        assert cache.size <= max_size

    def test_cache_hit(self):
        cache = ScoreCache(max_size=10, ttl_seconds=60.0)
        cache.put("q", "p", 0.8, 0.1, 0.2)
        entry = cache.get("q", "p")
        assert entry is not None
        assert entry.score == 0.8

    def test_cache_miss(self):
        cache = ScoreCache(max_size=10, ttl_seconds=60.0)
        entry = cache.get("nonexistent", "p")
        assert entry is None

    def test_ttl_expiry(self):
        cache = ScoreCache(max_size=10, ttl_seconds=0.05)
        cache.put("q", "p", 0.8, 0.1, 0.2)
        time.sleep(0.1)
        entry = cache.get("q", "p")
        assert entry is None

    def test_invalidate_makes_entries_stale(self):
        cache = ScoreCache(max_size=10, ttl_seconds=60.0)
        for i in range(5):
            cache.put(f"q{i}", "p", 0.5, 0.1, 0.1)
        assert cache.size == 5
        cache.invalidate()
        # After invalidation, gets should miss (entries stale)
        for i in range(5):
            entry = cache.get(f"q{i}", "p")
            assert entry is None, f"q{i} should be stale after invalidate()"

    def test_many_threads_no_crash(self):
        cache = ScoreCache(max_size=50, ttl_seconds=60.0)

        def worker(tid):
            for i in range(200):
                cache.put(f"q_{tid}_{i}", "p", 0.5, 0.1, 0.1)
                cache.get(f"q_{tid}_{i}", "p")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert cache.size <= 50


class TestCachePerformanceDoc:
    """Document cache pipeline performance characteristics."""

    def test_cache_put_get_fast(self):
        cache = ScoreCache(max_size=1000, ttl_seconds=60.0)
        for i in range(1000):
            cache.put(f"q{i}", "p", 0.5, 0.1, 0.1)

        t0 = time.perf_counter()
        for i in range(1000):
            cache.get(f"q{i}", "p")
        per_call_us = (time.perf_counter() - t0) / 1000 * 1_000_000
        assert per_call_us < 50, f"Cache get took {per_call_us:.1f}µs (expected <50µs)"

    def test_cache_integrates_with_scorer(self):
        from director_ai.core import CoherenceScorer

        scorer = CoherenceScorer(use_nli=False, cache_size=100, cache_ttl=60.0)
        _, s1 = scorer.review("What?", "Answer.")
        _, s2 = scorer.review("What?", "Answer.")
        assert s1.score == s2.score
