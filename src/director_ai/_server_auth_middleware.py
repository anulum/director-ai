# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server Auth Middleware (request IDs, API keys, tenants)

"""HTTP auth middleware: correlation IDs, API-key auth, tenant binding.

Split out of :mod:`director_ai.server` (WCB-8).
:func:`install_auth_middleware` computes the effective auth state
(exempt paths, valid API keys, key→tenant map), exposes it on
``app.state`` for the routers, registers the WebSocket ticket registry,
and installs the HTTP middleware that stamps request IDs, enforces
constant-time API-key checks with tenant binding, and records metrics.
"""

from __future__ import annotations

import contextvars
import hmac
import json as _json_mod
import logging
import time
from typing import TYPE_CHECKING

from .server_support import (
    _extract_request_api_key,
    _normalize_request_id,
    _record_http_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from fastapi import FastAPI, Request, Response

    from .core.config import DirectorConfig

logger = logging.getLogger("DirectorAI.Server")

REQUEST_ID_CTX: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="",
)

_AUTH_EXEMPT_PATHS_BASE = frozenset(
    {"/v1/live", "/v1/health", "/v1/ready", "/v1/source"}
)

__all__ = [
    "REQUEST_ID_CTX",
    "install_auth_middleware",
]


def install_auth_middleware(app: FastAPI, cfg: DirectorConfig) -> None:
    """Wire auth state onto ``app.state`` and install the HTTP middleware."""
    from fastapi.responses import JSONResponse

    # Middleware: correlation IDs + API key auth + metrics

    _auth_exempt = (
        _AUTH_EXEMPT_PATHS_BASE
        if cfg.metrics_require_auth
        else _AUTH_EXEMPT_PATHS_BASE | {"/v1/metrics/prometheus"}
    )

    _api_key_tenant_map: dict[str, str] = {}
    if cfg.api_key_tenant_map:
        _api_key_tenant_map = _json_mod.loads(cfg.api_key_tenant_map)

    # Effective auth keys = explicit api_keys ∪ tenant-map keys. Enforcement
    # must consider both: a map-only config (e.g. the production profile with a
    # key→tenant binding but no separate api_keys list) still requires a valid
    # key. Keying enforcement on cfg.api_keys alone is fail-open.
    _valid_api_keys: list[str] = list(cfg.api_keys)
    for _bound_key in _api_key_tenant_map:
        if _bound_key not in _valid_api_keys:
            _valid_api_keys.append(_bound_key)
    if cfg.production_mode and not _valid_api_keys:
        raise RuntimeError(
            "production_mode requires at least one effective API key "
            "(set api_keys or api_key_tenant_map)"
        )

    # Expose the effective auth state on app.state so request handlers (and the
    # routers split out of this factory) read it from the request instead of
    # closing over create_app locals. The list is shared by reference, so later
    # merges stay visible.
    app.state.valid_api_keys = _valid_api_keys
    app.state.api_key_tenant_map = _api_key_tenant_map

    from .core.runtime.ws_ticket import WebSocketTicketRegistry

    _ws_ticket_registry = WebSocketTicketRegistry(
        ttl_seconds=getattr(cfg, "ws_ticket_ttl_seconds", 30.0),
    )
    app.state.ws_ticket_registry = _ws_ticket_registry

    @app.middleware("http")
    async def _http_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Apply request IDs, API-key auth, tenant binding, and metrics."""
        request_id = _normalize_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        REQUEST_ID_CTX.set(request_id)

        start = time.monotonic()
        api_key_hash = ""
        if _valid_api_keys and request.url.path not in _auth_exempt:
            provided = _extract_request_api_key(request)
            # Constant-time: always compare against ALL keys to prevent
            # timing side-channels that leak key position.
            key_valid = False
            for k in _valid_api_keys:
                if hmac.compare_digest(provided, k):
                    key_valid = True
            if not key_valid:
                logger.warning(
                    "Auth failed from %s on %s",
                    request.client.host if request.client else "unknown",
                    request.url.path,
                )
                # Declared as the base Response so the later call_next() result
                # (a Response) is assignable to the same name.
                response: Response = JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                    headers={"X-Request-ID": request_id},
                )
                _record_http_metrics(
                    request,
                    status_code=response.status_code,
                    started_at=start,
                )
                return response
            import hashlib

            from .core.safety.audit_salt import get_audit_salt

            # Salted truncated SHA-256 fingerprint for audit logs only — NOT
            # used for authentication or password storage. The API key is
            # verified via constant-time HMAC comparison above. Salt is
            # per-installation (VULN-DAI-003) so a leaked log from one
            # deployment cannot be replayed against fingerprints from another.
            api_key_hash = hashlib.sha256(
                get_audit_salt() + provided.encode(),
            ).hexdigest()[:16]

            # Tenant binding: enforce API key → tenant mapping if configured
            if _api_key_tenant_map:
                if provided not in _api_key_tenant_map:
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "API key not bound to any tenant"},
                        headers={"X-Request-ID": request_id},
                    )
                    _record_http_metrics(
                        request,
                        status_code=response.status_code,
                        started_at=start,
                    )
                    return response
                bound_tenant = _api_key_tenant_map[provided]
                claimed_tenant = request.headers.get("X-Tenant-ID", "")
                if claimed_tenant and claimed_tenant != bound_tenant:
                    response = JSONResponse(
                        status_code=403,
                        content={"detail": "API key not authorized for this tenant"},
                        headers={"X-Request-ID": request_id},
                    )
                    _record_http_metrics(
                        request,
                        status_code=response.status_code,
                        started_at=start,
                    )
                    return response
                request.state.tenant_id = bound_tenant
                request.state.kb_write_key_ok = True
                request.state.kb_tenant_binding_ok = True
            else:
                # No key→tenant map: accept header but log for audit.
                # Tenant isolation without key binding is advisory only.
                claimed = request.headers.get("X-Tenant-ID", "")
                if claimed:
                    # api_key_hash is a SHA-256 digest (see above), not the key.
                    # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure
                    logger.debug(
                        "Unbound tenant claim: %s (api_key=%s)", claimed, api_key_hash
                    )
                request.state.tenant_id = claimed
                request.state.kb_write_key_ok = True
                request.state.kb_tenant_binding_ok = False
        else:
            # No API keys configured — tenant from header is untrusted
            request.state.tenant_id = request.headers.get("X-Tenant-ID", "")
            request.state.kb_write_key_ok = False
            request.state.kb_tenant_binding_ok = False

        request.state.api_key_hash = api_key_hash

        # Metrics
        try:
            response = await call_next(request)
        except Exception:
            _record_http_metrics(
                request,
                status_code=500,
                started_at=start,
            )
            raise
        _record_http_metrics(
            request,
            status_code=response.status_code,
            started_at=start,
        )
        response.headers["X-Request-ID"] = request_id
        return response
