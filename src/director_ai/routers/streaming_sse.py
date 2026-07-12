# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Server-Sent Events streaming route

"""The ``POST /v1/stream/sse`` Server-Sent Events route.

REST counterpart of the ``/v1/stream`` WebSocket: the same two session
shapes — token-level pre-egress oversight (``streaming_oversight``) and
whole-answer processing — delivered as an ``text/event-stream`` response
for clients that cannot (or prefer not to) hold a WebSocket: server-to-
server callers, ``curl``, EventSource-style consumers behind strict
proxies.

Event names mirror the WebSocket message ``type`` values: ``token``,
``halt``, ``complete``, ``result``, and ``error``; each ``data:`` line is
the JSON payload the WebSocket would have sent. Validation problems are
rejected *before* the stream starts as plain HTTP errors (400/413/503) —
only a session that has begun streaming reports failures as an ``error``
event.

Auth and tenant binding ride the standard REST middleware (the path is
not auth-exempt), so unlike the WebSocket no ticket exchange is needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from ..core.metrics import metrics

logger = logging.getLogger("DirectorAI.Server")

_SSE_MAX_PROMPT_LENGTH = 100_000
# Long-lived responses hold a worker slot each; cap them per process like
# the WebSocket connection budget.
_SSE_MAX_CONCURRENT_STREAMS = 64

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import StreamingResponse

    from .._server_helpers import evidence_to_dict as _evidence_to_dict
    from .._server_helpers import halt_evidence_to_dict as _halt_evidence_to_dict

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def _sse_event(event: str, data: dict[str, Any]) -> str:
    """Serialise one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_sse_router() -> APIRouter:
    """Build the SSE streaming route group (/v1/stream/sse)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    sse_lock = asyncio.Lock()
    sse_state = {"active": 0}

    async def _sse_admit() -> bool:
        """Reserve a stream slot under the per-process cap."""
        async with sse_lock:
            if sse_state["active"] >= _SSE_MAX_CONCURRENT_STREAMS:
                metrics.inc_labeled("sse_rejections_total", {"reason": "global_cap"})
                return False
            sse_state["active"] += 1
            metrics.gauge_set("sse_active_streams", float(sse_state["active"]))
            return True

    async def _sse_release() -> None:
        """Release a previously reserved stream slot."""
        async with sse_lock:
            sse_state["active"] = max(0, sse_state["active"] - 1)
            metrics.gauge_set("sse_active_streams", float(sse_state["active"]))

    @router.post("/v1/stream/sse")
    async def stream_sse(request: Request) -> StreamingResponse:
        """Stream one scored session as Server-Sent Events."""
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(400, "invalid JSON body") from exc
        if not isinstance(body, dict):
            raise HTTPException(400, "expected a JSON object")

        prompt = body.get("prompt", "")
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(400, "prompt must be a non-empty string")
        if len(prompt) > _SSE_MAX_PROMPT_LENGTH:
            raise HTTPException(
                413,
                f"prompt exceeds {_SSE_MAX_PROMPT_LENGTH} chars",
            )

        sanitizer = request.app.state._state.get("sanitizer")
        if sanitizer:
            check = sanitizer.check(prompt)
            if check.blocked:
                raise HTTPException(400, f"injection rejected: {check.reason}")

        agent = request.app.state._state.get("agent")
        if not agent:
            raise HTTPException(503, "server not ready")

        if not await _sse_admit():
            raise HTTPException(503, "server at capacity")

        cfg = request.app.state.config
        tenant_id = getattr(request.state, "tenant_id", "") or request.headers.get(
            "X-Tenant-ID",
            "",
        )
        oversight = bool(body.get("streaming_oversight"))

        async def _events() -> AsyncIterator[str]:
            try:
                if oversight:
                    async for frame in _oversight_events(
                        agent,
                        cfg,
                        prompt,
                        tenant_id,
                    ):
                        yield frame
                else:
                    yield await _result_event(agent, prompt, tenant_id)
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.error("SSE streaming failed: %s", exc)
                yield _sse_event("error", {"error": "streaming failed"})
            finally:
                await _sse_release()

        return StreamingResponse(
            _events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


async def _oversight_events(
    agent: Any,
    cfg: Any,
    prompt: str,
    tenant_id: str,
) -> AsyncIterator[str]:
    """Yield token-level pre-egress oversight frames for one session.

    Mirrors the WebSocket oversight branch: every token is scored by the
    streaming kernel *before* it is forwarded, and the token that trips
    the halt is never delivered.
    """
    from ..core import StreamingKernel

    kernel = StreamingKernel(
        hard_limit=cfg.hard_limit,
        window_size=getattr(cfg, "window_size", 5),
        window_threshold=getattr(cfg, "window_threshold", 0.5),
    )
    kernel.reset_state()
    delivered = 0
    halted = False
    last_score = 1.0
    async for token, score in agent.stream(prompt, tenant_id=tenant_id):
        last_score = score
        if kernel.check_halt(score):
            halted = True
            break
        yield _sse_event(
            "token",
            {"token": token, "coherence": round(score, 4)},
        )
        delivered += 1
    yield _sse_event(
        "halt" if halted else "complete",
        {
            "halted": halted,
            "tokens_delivered": delivered,
            "coherence": round(last_score, 4),
            **({"reason": "coherence_halt"} if halted else {}),
        },
    )


async def _result_event(agent: Any, prompt: str, tenant_id: str) -> str:
    """Run one whole-answer session and serialise its result frame."""
    result = await agent.aprocess(prompt, tenant_id=tenant_id)
    return _sse_event(
        "result",
        {
            "output": result.output,
            "coherence": (result.coherence.score if result.coherence else None),
            "halted": result.halted,
            "warning": (result.coherence.warning if result.coherence else False),
            "fallback_used": result.fallback_used,
            "evidence": _evidence_to_dict(
                result.coherence.evidence if result.coherence else None,
            ),
            "halt_evidence": _halt_evidence_to_dict(result.halt_evidence),
        },
    )
