# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Proxy Audit and Fail-Closed Error Plumbing

"""Compliance audit logging and fail-closed error responses for the proxy."""

from __future__ import annotations

import logging
import time as _time
from typing import Any

# FastAPI resolves annotations at runtime (``get_type_hints``), so the response
# types must exist in module globals — not only under ``TYPE_CHECKING``. Import
# them at module level, degrading gracefully when the optional ``[server]``
# extra is absent (the module still imports; using it raises a clear NameError).
try:
    from fastapi import Response
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover — exercised only without the server extra
    pass

_log = logging.getLogger("DirectorAI.Proxy")


def _scorer_error_response(
    audit_log: Any,
    prompt: str,
    text: str,
    *,
    model: str,
    task_type: str,
    t0: float,
) -> Response:
    """Record a non-streaming scorer failure and return a fail-closed 503.

    Mirrors the streaming fail-closed path (KIMI3-H5): a ``scorer.review``
    exception must not surface the unreviewed model output. It is logged,
    recorded as an ``approved=False`` audit entry, and answered with a clear
    503 rather than the bare 500 an uncaught exception would produce. Call only
    from within the review ``except`` block (uses the active exception context).
    """
    _log.exception("scorer.review failed on a non-streaming request; failing closed")
    _audit_log_entry(
        audit_log,
        prompt,
        text,
        model=model,
        score=0.0,
        approved=False,
        confidence=0.0,
        latency_ms=(_time.monotonic() - t0) * 1000,
        task_type=task_type,
    )
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "message": "Scoring unavailable — request halted by Director-AI",
                "type": "scorer_error",
            },
        },
    )


def _audit_log_entry(
    audit_log: Any,
    prompt: str,
    response: str,
    *,
    model: str,
    score: float,
    approved: bool,
    confidence: float,
    latency_ms: float,
    task_type: str = "chat",
) -> None:
    """Log a scored interaction to the compliance audit log (if enabled)."""
    if audit_log is None:
        return
    from director_ai.compliance.audit_log import AuditEntry

    audit_log.log(
        AuditEntry(
            prompt=prompt,
            response=response,
            model=model,
            provider="proxy",
            score=score,
            approved=approved,
            verdict_confidence=confidence,
            task_type=task_type,
            domain="",
            latency_ms=latency_ms,
            timestamp=_time.time(),
        )
    )
