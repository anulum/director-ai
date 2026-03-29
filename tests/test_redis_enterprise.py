# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Redis Enterprise Store Tests (Fakeredis)

"""Tests for RedisGroundTruthStore and RedisScoreCache using fakeredis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_fake_redis():
    """Create a dict-backed fake Redis client."""
    store: dict[str, str] = {}
    hashes: dict[str, dict[str, str]] = {}

    client = MagicMock()

    def hset(name, key, value):
        hashes.setdefault(name, {})[key] = value

    def hgetall(name):
        return dict(hashes.get(name, {}))

    def hlen(name):
        return len(hashes.get(name, {}))

    def get(key):
        return store.get(key)

    def setex(key, ttl, value):
        store[key] = value

    def delete(*keys):
        for k in keys:
            store.pop(k, None)

    def scan(cursor, match="*", count=100):
        import fnmatch

        matched = [k for k in store if fnmatch.fnmatch(k, match)]
        return 0, matched

    def pipeline():
        pipe = MagicMock()
        ops = []

        def pipe_hset(name, key, value):
            ops.append(("hset", name, key, value))

        def pipe_execute():
            for op in ops:
                if op[0] == "hset":
                    hset(op[1], op[2], op[3])
            ops.clear()

        pipe.hset = pipe_hset
        pipe.execute = pipe_execute
        return pipe

    client.hset = hset
    client.hgetall = hgetall
    client.hlen = hlen
    client.get = get
    client.setex = setex
    client.delete = delete
    client.scan = scan
    client.pipeline = pipeline
    return client


class TestRedisGroundTruthStore:
    def test_add_and_retrieve(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            store.add("sky color", "The sky is blue")
            result = store.retrieve_context("What is the sky color?")
            assert result is not None
            assert "blue" in result

    def test_add_many(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            count = store.add_many({"sky": "blue", "grass": "green"})
            assert count == 2
            assert store.count() == 2

    def test_tenant_isolation(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            store.add("sky", "blue", tenant_id="t1")
            store.add("sky", "green", tenant_id="t2")
            assert store.count(tenant_id="t1") == 1
            assert store.count(tenant_id="t2") == 1

    def test_retrieve_no_match(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            store.add("sky", "blue")
            assert store.retrieve_context("unrelated query") is None


class TestRedisScoreCache:
    def test_put_and_get(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake", ttl_seconds=60)
            cache.put("query", "prefix", 0.85, 0.1, 0.2)
            entry = cache.get("query", "prefix")
            assert entry is not None
            assert entry.score == pytest.approx(0.85)
            assert entry.h_logical == pytest.approx(0.1)
            assert entry.h_factual == pytest.approx(0.2)

    def test_miss(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            assert cache.get("missing", "query") is None
            assert cache.misses == 1

    def test_generation_staleness(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.1, 0.2)
            cache.invalidate()
            assert cache.get("q", "p") is None
            assert cache.misses >= 1

    def test_clear(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q1", "p1", 0.8, 0.1, 0.1)
            cache.put("q2", "p2", 0.7, 0.2, 0.2)
            cache.clear()
            assert cache.size == 0
            assert cache.hits == 0
            assert cache.misses == 0

    def test_hit_miss_counters(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.0, 0.0)
            cache.get("q", "p")
            cache.get("missing", "p")
            assert cache.hits == 1
            assert cache.misses == 1

    def test_tenant_id_changes_cache_key(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.0, 0.0, tenant_id="tenant-a")
            assert cache.get("q", "p", tenant_id="tenant-a") is not None
            assert cache.get("q", "p", tenant_id="tenant-b") is None

    def test_scope_changes_cache_key(self):
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.0, 0.0, scope="session-a")
            assert cache.get("q", "p", scope="session-a") is not None
            assert cache.get("q", "p", scope="session-b") is None
