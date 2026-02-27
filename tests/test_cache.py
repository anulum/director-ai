# Tests for ScoreCache

import time

from director_ai.core.cache import ScoreCache


class TestScoreCache:
    def test_put_and_get(self):
        cache = ScoreCache(max_size=100)
        cache.put("q1", "prefix1", 0.8, 0.1, 0.2)
        entry = cache.get("q1", "prefix1")
        assert entry is not None
        assert entry.score == 0.8
        assert entry.h_logical == 0.1
        assert entry.h_factual == 0.2

    def test_miss(self):
        cache = ScoreCache(max_size=100)
        assert cache.get("q1", "prefix1") is None
        assert cache.misses == 1

    def test_hit_rate(self):
        cache = ScoreCache(max_size=100)
        cache.put("q1", "p1", 0.8, 0.1, 0.2)
        cache.get("q1", "p1")  # hit
        cache.get("q1", "p2")  # miss
        assert cache.hits == 1
        assert cache.misses == 1
        assert cache.hit_rate == 0.5

    def test_lru_eviction(self):
        cache = ScoreCache(max_size=2)
        cache.put("q1", "p1", 0.8, 0.1, 0.2)
        cache.put("q2", "p2", 0.7, 0.2, 0.3)
        cache.put("q3", "p3", 0.6, 0.3, 0.4)  # evicts q1
        assert cache.get("q1", "p1") is None
        assert cache.get("q2", "p2") is not None

    def test_ttl_expiry(self):
        cache = ScoreCache(max_size=100, ttl_seconds=0.01)
        cache.put("q1", "p1", 0.8, 0.1, 0.2)
        time.sleep(0.02)
        assert cache.get("q1", "p1") is None

    def test_clear(self):
        cache = ScoreCache(max_size=100)
        cache.put("q1", "p1", 0.8, 0.1, 0.2)
        cache.get("q1", "p1")
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_size(self):
        cache = ScoreCache(max_size=100)
        assert cache.size == 0
        cache.put("q1", "p1", 0.8, 0.1, 0.2)
        assert cache.size == 1

    def test_scorer_with_cache(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.5, cache_size=100)
        assert scorer.cache is not None

        _, score1 = scorer.review("What is 2+2?", "2+2 is 4")
        _, score2 = scorer.review("What is 2+2?", "2+2 is 4")

        # Second call should hit cache
        assert scorer.cache.hits >= 1
        assert score1.score == score2.score

    def test_scorer_without_cache(self):
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.5)
        assert scorer.cache is None
