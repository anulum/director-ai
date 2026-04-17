# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — API key authentication middleware
"""Starlette/FastAPI middleware that validates ``Authorization: Bearer <key>``
or ``X-API-Key: <key>`` headers against a set of valid keys.

Keys can be loaded from environment (``DIRECTOR_API_KEYS``), a file,
or passed directly. The ``/health`` endpoint is always exempt.

Usage::

    from director_ai.middleware.api_key import APIKeyMiddleware

    app.add_middleware(
        APIKeyMiddleware,
        keys={"sk-live-abc123", "sk-live-def456"},
    )
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("DirectorAI.APIKey")

# Paths that never require authentication
_EXEMPT_PATHS = frozenset({"/health", "/healthz", "/ready", "/metrics", "/"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate API keys on every request (except health checks).

    Parameters
    ----------
    app : ASGIApp
        The Starlette/FastAPI application.
    keys : set[str] | None
        Valid API keys. If None, reads ``DIRECTOR_API_KEYS`` env var
        (comma-separated).
    keys_file : str
        Path to a file with one key per line.
    on_reject : Callable | None
        Custom rejection handler. Receives the Request; must return a
        Response. Defaults to 401 JSON.
    """

    def __init__(
        self,
        app,
        keys: set[str] | None = None,
        keys_file: str = "",
        on_reject: Callable | None = None,
    ) -> None:
        super().__init__(app)
        self._keys = self._load_keys(keys, keys_file)
        self._on_reject = on_reject
        logger.info("APIKeyMiddleware: %d key(s) loaded", len(self._keys))

    @staticmethod
    def _load_keys(keys: set[str] | None, keys_file: str) -> set[str]:
        """Resolve keys from arguments, env, or file."""
        result: set[str] = set()
        if keys:
            result.update(keys)
        env_keys = os.environ.get("DIRECTOR_API_KEYS", "")
        if env_keys:
            result.update(k.strip() for k in env_keys.split(",") if k.strip())
        if keys_file and os.path.isfile(keys_file):
            with open(keys_file) as f:  # noqa: PTH123
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        result.update(line.split())
        return result

    async def dispatch(self, request: Request, call_next):
        """Check API key before forwarding to the application."""
        # Exempt paths
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        # Extract key from Authorization header or X-API-Key
        key = _extract_key(request)
        if key is None or not self._validate(key):
            if self._on_reject:
                return self._on_reject(request)
            return JSONResponse(
                {"error": "Invalid or missing API key"},
                status_code=401,
            )

        # Attach key hash to request state for downstream usage metering
        request.state.api_key_hash = _hash_key(key)
        return await call_next(request)

    def _validate(self, key: str) -> bool:
        """Constant-time key validation."""
        return any(hmac.compare_digest(key, valid) for valid in self._keys)


def _extract_key(request: Request) -> str | None:
    """Extract API key from request headers."""
    # Try Authorization: Bearer <key>
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # Try X-API-Key header
    return request.headers.get("x-api-key")


def _hash_key(key: str) -> str:
    """HMAC-SHA512 fingerprint for audit logging (never log raw keys).

    Salt is loaded per-installation via
    :func:`director_ai.core.safety.audit_salt.get_audit_salt`; this
    preserves deterministic fingerprints within one deployment while
    preventing a shared rainbow table across all Director-AI installs.
    SHA-512 is chosen to satisfy CodeQL crypto-strength requirements.
    """
    from director_ai.core.safety.audit_salt import get_audit_salt

    return hmac.new(get_audit_salt(), key.encode(), "sha512").hexdigest()[:16]
