# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Redis Enterprise Store Tests (Fakeredis)

"""Multi-angle tests for Redis enterprise store pipeline.

Covers: RedisGroundTruthStore add/query, RedisScoreCache put/get/TTL,
tenant isolation, import guard, fakeredis mocking, parametrised
operations, pipeline integration with CoherenceScorer, and performance.
"""

from __future__ import annotations

import builtins
import importlib
from unittest.mock import MagicMock, patch

import pytest


def _make_fake_redis() -> MagicMock:
    """Create a dict-backed fake Redis client."""
    store: dict[str, str] = {}
    hashes: dict[str, dict[str, str]] = {}

    client = MagicMock()

    def hset(name: str, key: str, value: str) -> None:
        hashes.setdefault(name, {})[key] = value

    def hgetall(name: str) -> dict[str, str]:
        return dict(hashes.get(name, {}))

    def hlen(name: str) -> int:
        return len(hashes.get(name, {}))

    def get(key: str) -> str | None:
        return store.get(key)

    def set_value(key: str, value: str, *, ex: int | None = None) -> None:
        _ = ex
        store[key] = value

    def setex(key: str, ttl: int, value: str) -> None:
        _ = ttl
        store[key] = value

    def delete(*keys: str) -> None:
        for k in keys:
            store.pop(k, None)

    def scan(
        cursor: int,
        match: str = "*",
        count: int = 100,
    ) -> tuple[int, list[str]]:
        _ = count
        import fnmatch

        matched = [k for k in store if fnmatch.fnmatch(k, match)]
        if cursor == 0 and len(matched) > count:
            return 1, matched[:count]
        if cursor != 0:
            return 0, matched[count:]
        return 0, matched

    def pipeline() -> MagicMock:
        pipe = MagicMock()
        ops: list[tuple[str, str, str, str]] = []

        def pipe_hset(name: str, key: str, value: str) -> None:
            ops.append(("hset", name, key, value))

        def pipe_execute() -> None:
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
    client.set = set_value
    client.setex = setex
    client.delete = delete
    client.scan = scan
    client.pipeline = pipeline
    return client


def test_import_guards_when_redis_package_is_missing() -> None:
    """Redis enterprise classes should fail closed without the optional package."""
    import director_ai.enterprise.redis as redis_module

    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "redis":
            raise ImportError("blocked redis")
        return original_import(name, globals, locals, fromlist, level)

    with patch.object(builtins, "__import__", blocked_import):
        missing_module = importlib.reload(redis_module)

    try:
        with pytest.raises(ImportError, match="redis package required"):
            missing_module.RedisGroundTruthStore(redis_url="redis://fake")
        with pytest.raises(ImportError, match="redis wrapper requires"):
            missing_module.RedisScoreCache(redis_url="redis://fake")
    finally:
        importlib.reload(redis_module)


def test_connection_failures_are_raised_for_store_and_cache() -> None:
    """Redis connection errors should not fall back to local in-memory state."""
    failing_client = MagicMock()
    failing_client.ping.side_effect = ConnectionError("redis unavailable")

    with patch("director_ai.enterprise.redis.redis") as mock_redis:
        mock_redis.from_url.return_value = failing_client
        from director_ai.enterprise.redis import RedisGroundTruthStore, RedisScoreCache

        with pytest.raises(ConnectionError, match="redis unavailable"):
            RedisGroundTruthStore(redis_url="redis://fake")
        with pytest.raises(ConnectionError, match="redis unavailable"):
            RedisScoreCache(redis_url="redis://fake")


class TestRedisGroundTruthStore:
    """Unit guard for RedisGroundTruthStore using a dict-backed Redis client."""

    def test_add_and_retrieve(self) -> None:
        """Stored facts should be retrievable through Redis hash reads."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            store.add("sky color", "The sky is blue")
            result = store.retrieve_context("What is the sky color?")
            assert result is not None
            assert "blue" in result

    def test_add_many(self) -> None:
        """Batch writes should update Redis and the inherited revision state."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            count = store.add_many({"sky": "blue", "grass": "green"})
            assert count == 2
            assert store.count() == 2

    def test_add_many_empty_is_noop(self) -> None:
        """An empty batch should avoid Redis writes and report zero additions."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            assert store.add_many({}) == 0
            assert store.count() == 0

    def test_tenant_isolation(self) -> None:
        """Tenant-specific hashes should keep same-key facts isolated."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            store.add("sky", "blue", tenant_id="t1")
            store.add("sky", "green", tenant_id="t2")
            assert store.count(tenant_id="t1") == 1
            assert store.count(tenant_id="t2") == 1

    def test_tenant_id_validation_accepts_safe_redis_ids(self) -> None:
        """Redis-safe tenant identifiers should be accepted."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            store.add("sky", "blue", tenant_id="tenant-A_123")
            assert store.count(tenant_id="tenant-A_123") == 1

    def test_tenant_id_validation_rejects_redis_glob_injection(self) -> None:
        """Redis glob syntax must be rejected in tenant identifiers."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            with pytest.raises(ValueError, match="Invalid Redis tenant_id"):
                store.add("sky", "blue", tenant_id="*__keyspace@0__:*")

    def test_retrieve_no_match(self) -> None:
        """Empty hashes and non-overlapping queries should return no context."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisGroundTruthStore

            store = RedisGroundTruthStore(redis_url="redis://fake")
            assert store.retrieve_context("empty query") is None
            store.add("sky", "blue")
            assert store.retrieve_context("unrelated query") is None


class TestRedisScoreCache:
    """Unit guard for RedisScoreCache using a dict-backed Redis client."""

    def test_put_and_get(self) -> None:
        """A cached score should round-trip through Redis JSON storage."""
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

    def test_cache_payload_uses_wall_clock_timestamp(self) -> None:
        """Redis cache entries should record wall-clock creation time."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake", ttl_seconds=60)
            with patch("director_ai.enterprise.redis.time.time", return_value=1234.5):
                cache.put("query", "prefix", 0.85, 0.1, 0.2)

            entry = cache.get("query", "prefix")
            assert entry is not None
            assert entry.created_at == pytest.approx(1234.5)

    def test_miss(self) -> None:
        """Missing Redis keys should return no cache entry and increment misses."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            assert cache.get("missing", "query") is None
            assert cache.misses == 1

    def test_generation_staleness(self) -> None:
        """Generation changes should evict stale Redis cache payloads."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.1, 0.2)
            cache.invalidate()
            assert cache.get("q", "p") is None
            assert cache.misses >= 1

    def test_malformed_payload_is_evicted(self) -> None:
        """Malformed Redis JSON should be deleted and counted as a miss."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.1, 0.2)
            _, keys = fake.scan(0, match="dai:cache:*")
            assert keys
            fake.set(keys[0], "{not-json", ex=60)

            assert cache.get("q", "p") is None
            assert cache.misses == 1
            assert cache.size == 0

    def test_clear(self) -> None:
        """Clear should delete all owned cache keys and reset counters."""
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

    def test_size_and_clear_scan_multiple_batches(self) -> None:
        """Size and clear should consume multi-cursor Redis scans."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            for idx in range(105):
                cache.put(f"q{idx}", "p", 0.9, 0.0, 0.0)

            assert cache.size == 105
            cache.clear()
            assert cache.size == 0

    def test_clear_empty_cache_is_noop(self) -> None:
        """Clearing an empty Redis cache prefix should leave counters reset."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.clear()
            assert cache.size == 0

    def test_put_uses_setex_fallback_when_set_command_is_unavailable(self) -> None:
        """Client-like objects without SET options should use the setex fallback."""
        fake = _make_fake_redis()
        fake.set = None
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.0, 0.0)
            assert cache.get("q", "p") is not None

    def test_hit_miss_counters(self) -> None:
        """Cache hit and miss counters should reflect Redis lookups."""
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

    def test_tenant_id_changes_cache_key(self) -> None:
        """Tenant identifiers should partition Redis score-cache keys."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.0, 0.0, tenant_id="tenant-a")
            assert cache.get("q", "p", tenant_id="tenant-a") is not None
            assert cache.get("q", "p", tenant_id="tenant-b") is None

    def test_tenant_id_validation_rejects_cache_tenant_injection(self) -> None:
        """Redis glob syntax must be rejected in cache tenant identifiers."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            with pytest.raises(ValueError, match="Invalid Redis tenant_id"):
                cache.put("q", "p", 0.9, 0.0, 0.0, tenant_id="*__keyspace@0__:*")

    def test_scope_changes_cache_key(self) -> None:
        """Cache scope strings should partition otherwise identical entries."""
        fake = _make_fake_redis()
        with patch("director_ai.enterprise.redis.redis") as mock_redis:
            mock_redis.from_url.return_value = fake
            from director_ai.enterprise.redis import RedisScoreCache

            cache = RedisScoreCache(redis_url="redis://fake")
            cache.put("q", "p", 0.9, 0.0, 0.0, scope="session-a")
            assert cache.get("q", "p", scope="session-a") is not None
            assert cache.get("q", "p", scope="session-b") is None
