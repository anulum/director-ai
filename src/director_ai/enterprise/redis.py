# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Director-AI Enterprise — Redis High-Availability State.

Requires: pip install director-ai[enterprise]
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from director_ai.core.cache import ScoreCache
from director_ai.core.knowledge import GroundTruthStore

# redis is optional — declare as Any so the except-branch
# assignment does not conflict with mypy-inferred module type.
redis: Any
try:
    import redis as _redis_mod

    redis = _redis_mod
except ImportError:
    redis = None

logger = logging.getLogger("DirectorAI.Enterprise.Redis")
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_redis_tenant_id(tenant_id: str) -> str:
    if tenant_id == "":
        return tenant_id
    if not _TENANT_ID_RE.fullmatch(tenant_id):
        raise ValueError(
            "Invalid Redis tenant_id: expected 1-64 characters matching [A-Za-z0-9_-]",
        )
    return tenant_id


class RedisGroundTruthStore(GroundTruthStore):
    """Distributed GroundTruthStore using Redis for high availability.

    Facts are stored in per-tenant Redis hashes for isolation.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "dai:facts:",
    ):
        """Connect to Redis and initialize a tenant-scoped fact store."""
        if redis is None:
            raise ImportError(
                "redis package required. Run: pip install director-ai[enterprise]",
            )
        super().__init__()
        self.redis_url = redis_url
        self.prefix = prefix
        self.client = redis.from_url(redis_url, decode_responses=True)
        try:
            self.client.ping()
        except Exception as exc:
            logger.error("Redis connection failed: %s", exc)
            raise

    def _hash_key(self, tenant_id: str = "") -> str:
        tenant_id = _validate_redis_tenant_id(tenant_id)
        return f"{self.prefix}{tenant_id or '_default'}:hash"

    def add(self, key: str, value: str, tenant_id: str = "") -> None:
        """Persist one fact in local memory and the tenant Redis hash."""
        tenant_id = _validate_redis_tenant_id(tenant_id)
        super().add(key, value, tenant_id=tenant_id)
        self.client.hset(self._hash_key(tenant_id), key, value)

    def add_many(self, facts: dict[str, str], tenant_id: str = "") -> int:
        """Batch-add facts. Returns count added."""
        if not facts:
            return 0
        tenant_id = _validate_redis_tenant_id(tenant_id)
        hk = self._hash_key(tenant_id)
        pipe = self.client.pipeline()
        for k, v in facts.items():
            super().add(k, v, tenant_id=tenant_id)
            pipe.hset(hk, k, v)
        pipe.execute()
        return len(facts)

    def retrieve_context(self, query: str, tenant_id: str = "") -> str | None:
        """Return Redis-backed fact context matching query terms for a tenant."""
        tenant_id = _validate_redis_tenant_id(tenant_id)
        facts_dict = self.client.hgetall(self._hash_key(tenant_id))
        if not facts_dict:
            return None

        query_lower = query.lower()
        context = [
            value
            for key, value in facts_dict.items()
            if any(word in query_lower for word in key.lower().split())
        ]
        return "; ".join(context) if context else None

    def count(self, tenant_id: str = "") -> int:
        """Return the number of stored facts for a tenant."""
        tenant_id = _validate_redis_tenant_id(tenant_id)
        return int(self.client.hlen(self._hash_key(tenant_id)))


class RedisScoreCache(ScoreCache):
    """Distributed ScoreCache using Redis.

    Shares coherence scores across all workers in the cluster.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "dai:cache:",
        ttl_seconds: float = 300.0,
    ):
        """Connect to Redis and initialize a distributed score cache."""
        if redis is None:
            raise ImportError(
                "redis wrapper requires the 'redis' package. "
                "Run: pip install director-ai[enterprise]",
            )
        # We call super to keep local state properties if needed, but override behavior
        super().__init__(ttl_seconds=ttl_seconds)
        self.redis_url = redis_url
        self.prefix = prefix
        self.client = redis.from_url(redis_url, decode_responses=True)
        try:
            self.client.ping()
        except Exception as exc:
            logger.error("Redis connection failed: %s", exc)
            raise
        logger.info(
            "Initialized RedisScoreCache at %s (ttl=%ss)", redis_url, ttl_seconds
        )

    def get(
        self,
        query: str,
        prefix: str,
        tenant_id: str = "",
        scope: str = "",
    ):
        """Load a cached score entry, enforcing generation and JSON integrity."""
        # Local import to construct the expected _CacheEntry format transparently
        from director_ai.core.cache import _CacheEntry

        tenant_id = _validate_redis_tenant_id(tenant_id)
        k = self._key(query, prefix, tenant_id, scope)
        redis_key = f"{self.prefix}{k}"

        data = self.client.get(redis_key)
        if not data:
            self.misses += 1
            return None

        try:
            parsed = json.loads(data)
            if parsed.get("generation", 0) != self._generation:
                self.client.delete(redis_key)
                self.misses += 1
                return None

            entry = _CacheEntry(
                score=float(parsed["score"]),
                h_logical=float(parsed["h_logical"]),
                h_factual=float(parsed["h_factual"]),
                created_at=float(parsed["created_at"]),
                generation=int(parsed["generation"]),
            )
            self.hits += 1
            return entry
        except (json.JSONDecodeError, KeyError, TypeError):
            self.client.delete(redis_key)
            self.misses += 1
            return None

    def put(
        self,
        query: str,
        prefix: str,
        score: float,
        h_logical: float,
        h_factual: float,
        tenant_id: str = "",
        scope: str = "",
    ) -> None:
        """Store a coherence score entry under the configured Redis TTL."""
        tenant_id = _validate_redis_tenant_id(tenant_id)
        k = self._key(query, prefix, tenant_id, scope)
        redis_key = f"{self.prefix}{k}"

        payload = json.dumps(
            {
                "score": score,
                "h_logical": h_logical,
                "h_factual": h_factual,
                "created_at": time.monotonic(),
                "generation": self._generation,
            },
        )

        # Set with TTL
        self.client.setex(redis_key, int(self._ttl), payload)

    @property
    def size(self) -> int:
        """Return the count of Redis keys currently owned by this cache prefix."""
        count = 0
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=f"{self.prefix}*", count=100)
            count += len(keys)
            if cursor == 0:
                break
        return count

    def clear(self) -> None:
        """Delete all score-cache Redis keys under this prefix and reset metrics."""
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor, match=f"{self.prefix}*", count=100)
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break
        self.hits = 0
        self.misses = 0
