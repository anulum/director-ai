# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rate limiting middleware
"""Token-bucket rate limiter for the SaaS API.

Limits requests per API key (from ``request.state.api_key_hash``,
set by ``APIKeyMiddleware``) or per IP if no key is present.

Usage::

    from director_ai.middleware.rate_limit import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        burst=10,
        redis_url="redis://localhost:6379/0",
    )
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Protocol

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger("DirectorAI.RateLimit")

# Paths exempt from rate limiting
_EXEMPT_PATHS = frozenset({"/health", "/healthz", "/ready", "/metrics"})


class _TokenBucket:
    """Simple token-bucket rate limiter."""

    __slots__ = ("capacity", "tokens", "refill_rate", "last_refill")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        """Try to consume one token. Returns True if allowed."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until next token is available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


class RateLimitStore(Protocol):
    """Shared token-bucket storage used by ``RateLimitMiddleware``."""

    def consume(self, client_id: str) -> tuple[bool, float]:
        """Consume one request token and return ``(allowed, retry_after)``."""


class InMemoryRateLimitStore:
    """Per-process token-bucket storage."""

    def __init__(self, *, burst: int, refill_rate: float) -> None:
        self._burst = burst
        self._refill_rate = refill_rate
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(self._burst, self._refill_rate)
        )

    def consume(self, client_id: str) -> tuple[bool, float]:
        """Consume one token from the local process bucket."""
        bucket = self._buckets[client_id]
        if bucket.consume():
            return True, 0.0
        return False, bucket.retry_after


class RedisRateLimitStore:
    """Redis-backed shared token-bucket storage.

    The Lua script keeps the refill and consume operation atomic across workers
    and service instances.
    """

    _SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl_ms = tonumber(ARGV[4])

local state = redis.call("HMGET", key, "tokens", "last")
local tokens = tonumber(state[1])
local last = tonumber(state[2])
if tokens == nil then
  tokens = capacity
  last = now
end

local elapsed = now - last
if elapsed < 0 then
  elapsed = 0
end
tokens = math.min(capacity, tokens + (elapsed * refill_rate))
last = now

local allowed = 0
local retry_after = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  retry_after = (1 - tokens) / refill_rate
end

redis.call("HSET", key, "tokens", tokens, "last", last)
redis.call("PEXPIRE", key, ttl_ms)
return {allowed, retry_after}
"""

    def __init__(
        self,
        redis_url: str,
        *,
        burst: int,
        refill_rate: float,
        prefix: str = "dai:rate:",
    ) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - exercised by import guard
            raise ImportError(
                "Redis-backed rate limiting requires the redis package. "
                "Install with: pip install director-ai[enterprise]",
            ) from exc
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._burst = burst
        self._refill_rate = refill_rate
        self._prefix = prefix
        self._ttl_ms = max(int((burst / refill_rate) * 2000), 1000)

    def consume(self, client_id: str) -> tuple[bool, float]:
        """Consume one token from the shared Redis bucket."""
        key = self._prefix + client_id
        raw: Any = self._client.eval(
            self._SCRIPT,
            1,
            key,
            self._burst,
            self._refill_rate,
            time.time(),
            self._ttl_ms,
        )
        allowed = bool(int(raw[0]))
        retry_after = float(raw[1])
        return allowed, retry_after


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-key token-bucket rate limiter.

    Parameters
    ----------
    app : ASGIApp
        The Starlette/FastAPI application.
    requests_per_minute : int
        Sustained rate limit per key/IP.
    burst : int
        Maximum burst size (bucket capacity). Defaults to
        ``requests_per_minute // 6`` (10-second burst window).
    redis_url : str
        Optional Redis URL for a shared bucket across workers/instances.
    store : RateLimitStore
        Optional injected store for tests or custom deployments. Overrides
        ``redis_url`` and the default in-memory store.
    """

    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 60,
        burst: int | None = None,
        redis_url: str = "",
        redis_prefix: str = "dai:rate:",
        store: RateLimitStore | None = None,
    ) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst or max(requests_per_minute // 6, 1)
        self._refill_rate = requests_per_minute / 60.0
        resolved_store = store
        if resolved_store is None and redis_url:
            resolved_store = RedisRateLimitStore(
                redis_url,
                burst=self._burst,
                refill_rate=self._refill_rate,
                prefix=redis_prefix,
            )
        if resolved_store is None:
            resolved_store = InMemoryRateLimitStore(
                burst=self._burst,
                refill_rate=self._refill_rate,
            )
        self._store: RateLimitStore = resolved_store
        storage_kind = (
            "custom" if store is not None else "redis" if redis_url else "in-memory"
        )
        logger.info(
            "RateLimitMiddleware: %d req/min, burst=%d, store=%s",
            self._rpm,
            self._burst,
            storage_kind,
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Check rate limit before forwarding."""
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Identify client by API key hash or IP
        client_id = getattr(request.state, "api_key_hash", None)
        if client_id is None:
            client_id = request.client.host if request.client else "unknown"

        allowed, retry = self._store.consume(client_id)
        if not allowed:
            logger.warning(
                "Rate limit exceeded for %s (retry_after=%.1fs)",
                client_id[:8],
                retry,
            )
            return JSONResponse(
                {
                    "error": "Rate limit exceeded",
                    "retry_after_seconds": round(retry, 1),
                },
                status_code=429,
                headers={"Retry-After": str(int(retry) + 1)},
            )

        return await call_next(request)
