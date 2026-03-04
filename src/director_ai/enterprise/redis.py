"""
Director-AI Enterprise — Redis High-Availability State.

Requires: pip install director-ai[enterprise]
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

try:
    import redis
except ImportError:
    redis = None  # type: ignore

from director_ai.core.cache import ScoreCache
from director_ai.core.knowledge import GroundTruthStore

logger = logging.getLogger("DirectorAI.Enterprise.Redis")

class RedisGroundTruthStore(GroundTruthStore):
    """Distributed GroundTruthStore using Redis for high availability.
    
    Subclasses GroundTruthStore to intercept add/retrieve_context 
    and route them through a central Redis cluster instead of local RAM.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "dai:facts:"):
        if redis is None:
            raise ImportError("redis wrapper requires the 'redis' package. Run: pip install director-ai[enterprise]")
        super().__init__()
        self.redis_url = redis_url
        self.prefix = prefix
        self.client = redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Initialized RedisGroundTruthStore at {redis_url}")

    async def add(self, key: str, value: str) -> None:
        """Add or update a fact in the distributed Redis store."""
        import asyncio
        # Store locally for backward compatibility, but primarily store in Redis
        await super().add(key, value)
        await asyncio.to_thread(self.client.hset, f"{self.prefix}hash", key, value)
        
    async def retrieve_context(self, query: str) -> str | None:
        """Retrieve relevant facts from the Redis hash."""
        import asyncio
        facts_dict = await asyncio.to_thread(self.client.hgetall, f"{self.prefix}hash")
        if not facts_dict:
            logger.info("RedisGroundTruthStore is empty")
            return None

        query_lower = query.lower()
        context = []

        for key, value in facts_dict.items():
            key_words = key.lower().split()
            if any(word in query_lower for word in key_words):
                context.append(value)

        if context:
            retrieved = "; ".join(context)
            logger.info(f"RAG Retrieval (Redis): Found context '{retrieved}'")
            return retrieved

        return None


class RedisScoreCache(ScoreCache):
    """Distributed ScoreCache using Redis.
    
    Shares coherence scores across all workers in the cluster.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0", prefix: str = "dai:cache:", ttl_seconds: float = 300.0):
        if redis is None:
            raise ImportError("redis wrapper requires the 'redis' package. Run: pip install director-ai[enterprise]")
        # We call super to keep local state properties if needed, but override behavior
        super().__init__(ttl_seconds=ttl_seconds)
        self.redis_url = redis_url
        self.prefix = prefix
        self.client = redis.from_url(redis_url, decode_responses=True)
        logger.info(f"Initialized RedisScoreCache at {redis_url} (ttl={ttl_seconds}s)")

    def get(self, query: str, prefix: str):
        # Local import to construct the expected _CacheEntry format transparently
        from director_ai.core.cache import _CacheEntry
        
        k = self._key(query, prefix)
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

    def put(self, query: str, prefix: str, score: float, h_logical: float, h_factual: float) -> None:
        k = self._key(query, prefix)
        redis_key = f"{self.prefix}{k}"
        
        payload = json.dumps({
            "score": score,
            "h_logical": h_logical,
            "h_factual": h_factual,
            "created_at": time.monotonic(),
            "generation": self._generation
        })
        
        # Set with TTL
        self.client.setex(redis_key, int(self._ttl), payload)

    @property
    def size(self) -> int:
        # Approximate size of keys matching prefix
        return len(self.client.keys(f"{self.prefix}*"))

    def clear(self) -> None:
        keys = self.client.keys(f"{self.prefix}*")
        if keys:
            self.client.delete(*keys)
        self.hits = 0
        self.misses = 0
