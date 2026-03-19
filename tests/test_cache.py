# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
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

    def test_invalidate_bumps_generation(self):
        cache = ScoreCache(max_size=100, ttl_seconds=60)
        cache.put("q1", "p1", 0.85, 0.1, 0.2)
        gen_before = cache.generation
        cache.invalidate()
        assert cache.generation == gen_before + 1
        assert cache.get("q1", "p1") is None

    def test_empty_hit_rate(self):
        cache = ScoreCache(max_size=100)
        assert cache.hit_rate == 0.0

    def test_tenant_id_changes_cache_key(self):
        cache = ScoreCache(max_size=100)
        cache.put("q1", "p1", 0.8, 0.1, 0.2, tenant_id="tenant-a")
        assert cache.get("q1", "p1", tenant_id="tenant-a") is not None
        assert cache.get("q1", "p1", tenant_id="tenant-b") is None

    def test_scope_changes_cache_key(self):
        cache = ScoreCache(max_size=100)
        cache.put("q1", "p1", 0.8, 0.1, 0.2, scope="session-a")
        assert cache.get("q1", "p1", scope="session-a") is not None
        assert cache.get("q1", "p1", scope="session-b") is None

    def test_scorer_cache_uses_session_context_scope(self, monkeypatch):
        from director_ai.core.runtime.session import ConversationSession
        from director_ai.core.scorer import CoherenceScorer

        scorer = CoherenceScorer(threshold=0.0, use_nli=False, cache_size=100)
        values = iter(
            [
                (0.1, 0.1, 0.9, None),
                (0.2, 0.2, 0.7, None),
            ]
        )

        def fake_heuristic(prompt, action, tenant_id=""):
            return next(values)

        monkeypatch.setattr(scorer, "_heuristic_coherence", fake_heuristic)

        session_a = ConversationSession()
        session_a.add_turn("prev", "context A", 0.9)
        session_b = ConversationSession()
        session_b.add_turn("prev", "context B", 0.9)

        _, score_a = scorer.review("prompt", "answer", session=session_a)
        _, score_b = scorer.review("prompt", "answer", session=session_b)

        assert score_a.score == 0.9
        assert score_b.score == 0.7
