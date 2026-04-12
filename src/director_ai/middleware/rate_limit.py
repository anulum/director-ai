# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rate limiting middleware
"""In-memory sliding-window rate limiter for the SaaS API.

Limits requests per API key (from ``request.state.api_key_hash``,
set by ``APIKeyMiddleware``) or per IP if no key is present.

Usage::

    from director_ai.middleware.rate_limit import RateLimitMiddleware

    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=60,
        burst=10,
    )
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

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
    """

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst: int | None = None,
    ) -> None:
        super().__init__(app)
        self._rpm = requests_per_minute
        self._burst = burst or max(requests_per_minute // 6, 1)
        self._refill_rate = requests_per_minute / 60.0
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(self._burst, self._refill_rate)
        )
        logger.info(
            "RateLimitMiddleware: %d req/min, burst=%d",
            self._rpm,
            self._burst,
        )

    async def dispatch(self, request: Request, call_next):
        """Check rate limit before forwarding."""
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Identify client by API key hash or IP
        client_id = getattr(request.state, "api_key_hash", None)
        if client_id is None:
            client_id = request.client.host if request.client else "unknown"

        bucket = self._buckets[client_id]
        if not bucket.consume():
            retry = bucket.retry_after
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
