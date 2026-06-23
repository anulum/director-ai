# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FastAPI server support helpers

"""Stateless helpers shared by the Director-Class AI FastAPI server.

These functions carry no application state — they read request headers, derive
low-cardinality metric labels, normalise request IDs, and record metrics. They
live apart from ``server.py`` so the ``create_app`` factory keeps to application
wiring. ``server.py`` re-imports them, so ``director_ai.server.<name>`` keeps
resolving for callers and tests.
"""

from __future__ import annotations

import hmac
import time
import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .core.metrics import metrics

if TYPE_CHECKING:
    from fastapi import Request

_REQUEST_ID_MAX_LENGTH = 128
_REQUEST_ID_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
)


def _extract_api_key_from_headers(headers: Mapping[str, str]) -> str:
    """Return the caller API key from X-API-Key or Authorization: Bearer.

    Shared by the HTTP middleware and the WebSocket handshake so both transports
    accept the same auth headers. ``headers`` is any mapping with ``.get`` —
    ``Request.headers`` or ``WebSocket.headers``.
    """
    x_api_key = headers.get("X-API-Key", "").strip()
    if x_api_key:
        return x_api_key

    scheme, _, token = headers.get("Authorization", "").partition(" ")
    if scheme.lower() == "bearer":
        return token.strip()
    return ""


def _extract_request_api_key(request: Request) -> str:
    """Return the caller API key from supported production auth headers."""
    return _extract_api_key_from_headers(request.headers)


def _request_authenticated(request: Request) -> bool:
    """Return True when the caller is authenticated for detail disclosure.

    Auth-exempt probes (`/v1/health`, `/v1/source`) still answer to
    unauthenticated callers, but the detailed payload (version, mode,
    profile, routers, revision health) is only returned to a valid key
    holder. When no API keys are configured there is no auth posture
    (dev server), so detail is returned to keep local debugging usable.
    """
    valid_api_keys = request.app.state.valid_api_keys
    if not valid_api_keys:
        return True
    provided = _extract_request_api_key(request)
    return any(hmac.compare_digest(provided, k) for k in valid_api_keys)


def _record_sector_policy_findings(
    *,
    policy: str,
    report: Any,
    source: str,
) -> None:
    """Record tenant-safe sector-policy finding metrics."""
    for finding in getattr(report, "findings", ()):
        metrics.inc_labeled(
            "sector_policy_findings_total",
            {
                "policy": policy,
                "source": source,
                "code": finding.code,
                "severity": finding.severity,
                "action": finding.action,
            },
        )


def _can_suppress_batcher_metrics(batcher: Any) -> bool:
    """Return true when the batcher supports endpoint-owned metrics."""
    from .core.runtime.batch import BatchProcessor

    return isinstance(batcher, BatchProcessor)


def _http_endpoint_label(request: Request) -> str:
    """Return a low-cardinality route label for HTTP metrics."""
    route = request.scope.get("route")
    route_path = getattr(route, "path", None)
    if isinstance(route_path, str) and route_path:
        return route_path

    try:
        from starlette.routing import Match
    except ImportError:  # pragma: no cover - FastAPI depends on Starlette
        return "__unmatched__"

    partial_path = ""
    for candidate in request.app.routes:
        matches = getattr(candidate, "matches", None)
        if not callable(matches):
            continue
        match, _child_scope = matches(request.scope)
        candidate_path = getattr(candidate, "path", "")
        if not isinstance(candidate_path, str) or not candidate_path:
            continue
        if match is Match.FULL:
            return candidate_path
        if match is Match.PARTIAL and not partial_path:
            partial_path = candidate_path

    return partial_path or "__unmatched__"


def _record_http_metrics(
    request: Request,
    *,
    status_code: int,
    started_at: float,
) -> None:
    """Observe request duration and increment the labelled request counter."""
    elapsed = time.monotonic() - started_at
    metrics.observe("http_request_duration_seconds", elapsed)
    metrics.inc_labeled(
        "http_requests_total",
        {
            "method": request.method,
            "endpoint": _http_endpoint_label(request),
            "status": str(status_code),
        },
    )


def _normalize_request_id(raw: str | None) -> str:
    """Return *raw* if it is a safe, bounded request ID, else a fresh UUID4."""
    if (
        raw
        and len(raw) <= _REQUEST_ID_MAX_LENGTH
        and all(char in _REQUEST_ID_ALLOWED_CHARS for char in raw)
    ):
        return raw
    return str(uuid.uuid4())
