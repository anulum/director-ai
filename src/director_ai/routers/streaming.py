# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WebSocket streaming and ticket routes

"""The /v1/stream WebSocket route and its /v1/stream/ticket helper.

Split out of the ``create_app`` factory. The per-process WebSocket connection
accounting (global + per-IP caps) and the admit/release helpers are factory
locals — one set per app, exactly as before — and the handlers read the
effective auth keys, tenant map, ticket registry, and config from
``app.state``. ``create_streaming_router`` needs no construction-time
dependencies.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import threading
import time
import uuid
from typing import Any

from ..core.metrics import metrics
from ..server_support import (
    _extract_api_key_from_headers,
    _extract_request_api_key,
)

logger = logging.getLogger("DirectorAI.Server")

_WS_MAX_PROMPT_LENGTH = 100_000
_WS_MAX_CONCURRENT = 8
# Denial-of-service controls for the WebSocket streaming endpoint.
_WS_MAX_CONNECTIONS = 256  # global concurrent connections per process
_WS_MAX_CONNECTIONS_PER_IP = 16  # concurrent connections from one client IP
_WS_IDLE_TIMEOUT_S = 300.0  # close a connection idle this long between messages
_WS_MAX_LIFETIME_S = 3600.0  # close a connection older than this
_WS_RATE_WINDOW_S = 10.0  # sliding window for the per-connection message rate
_WS_MAX_MSGS_PER_WINDOW = 60  # messages allowed per window before rate limiting
# Per-session prompt byte budget — the SSE twin (_SSE_MAX_PROMPT_LENGTH in
# streaming_sse.py) has capped prompts since its introduction; the WebSocket
# path gained the same cap in the 2026-07-15 hardening slice (KIMI-E).
_WS_MAX_PROMPT_LENGTH = 100_000
_WS_CONN_CHAR_BUDGET = 5_000_000  # total prompt chars one connection may submit

try:
    from fastapi import (
        APIRouter,
        HTTPException,
        Request,
        WebSocket,
        WebSocketDisconnect,
    )

    from .._server_helpers import evidence_to_dict as _evidence_to_dict
    from .._server_helpers import halt_evidence_to_dict as _halt_evidence_to_dict

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - server extras absent
    _FASTAPI_AVAILABLE = False


def create_streaming_router() -> APIRouter:
    """Build the WebSocket streaming route group (ticket, stream)."""
    if not _FASTAPI_AVAILABLE:  # pragma: no cover - guarded by create_app
        raise ImportError(
            "FastAPI is required for the server. "
            "Install with: pip install director-ai[server]",
        )

    router = APIRouter()

    ws_conn_lock = asyncio.Lock()
    ws_conn_state: dict[str, int] = {"total": 0}
    ws_per_ip: dict[str, int] = {}

    async def _ws_admit(client_ip: str) -> bool:
        """Reserve a connection slot under the global and per-IP caps."""
        async with ws_conn_lock:
            if ws_conn_state["total"] >= _WS_MAX_CONNECTIONS:
                metrics.inc_labeled("ws_rejections_total", {"reason": "global_cap"})
                return False
            if ws_per_ip.get(client_ip, 0) >= _WS_MAX_CONNECTIONS_PER_IP:
                metrics.inc_labeled("ws_rejections_total", {"reason": "per_ip_cap"})
                return False
            ws_conn_state["total"] += 1
            ws_per_ip[client_ip] = ws_per_ip.get(client_ip, 0) + 1
            metrics.gauge_set("ws_active_connections", float(ws_conn_state["total"]))
            return True

    async def _ws_release(client_ip: str) -> None:
        """Release a previously reserved connection slot."""
        async with ws_conn_lock:
            ws_conn_state["total"] = max(0, ws_conn_state["total"] - 1)
            remaining = ws_per_ip.get(client_ip, 0) - 1
            if remaining > 0:
                ws_per_ip[client_ip] = remaining
            else:
                ws_per_ip.pop(client_ip, None)
            metrics.gauge_set("ws_active_connections", float(ws_conn_state["total"]))

    @router.post("/v1/stream/ticket")
    async def stream_ticket(request: Request) -> dict[str, Any]:
        """Issue a short-lived single-use ticket for a browser WebSocket connect.

        Browsers cannot attach auth headers to the WebSocket handshake, so an
        authenticated caller exchanges its key here for a ticket and connects to
        ``/v1/stream?ticket=...``. This endpoint is not auth-exempt, so the
        middleware enforces a valid key before a ticket is minted.
        """
        _valid_api_keys = request.app.state.valid_api_keys
        _ws_ticket_registry = request.app.state.ws_ticket_registry
        if not _valid_api_keys:
            raise HTTPException(
                400,
                "Ticket auth is unnecessary: no API keys are configured",
            )
        api_key = _extract_request_api_key(request)
        tenant_id = getattr(request.state, "tenant_id", "") or request.headers.get(
            "X-Tenant-ID",
            "",
        )
        ticket = _ws_ticket_registry.issue(api_key, tenant_id)
        return {"ticket": ticket, "expires_in": _ws_ticket_registry.ttl_seconds}

    @router.websocket("/v1/stream")
    async def stream(ws: WebSocket) -> None:
        """Handle multiplexed WebSocket agent sessions."""
        _valid_api_keys = ws.app.state.valid_api_keys
        _api_key_tenant_map = ws.app.state.api_key_tenant_map
        _ws_ticket_registry = ws.app.state.ws_ticket_registry
        cfg = ws.app.state.config
        ws_tenant_id = ""
        _ticket_tenant = ""
        if _valid_api_keys:
            provided = _extract_api_key_from_headers(ws.headers)
            if not provided:
                # Browser path: no custom handshake headers are possible, so
                # redeem a short-lived single-use ticket from the query string.
                binding = _ws_ticket_registry.redeem(ws.query_params.get("ticket", ""))
                if binding is not None:
                    provided = binding.api_key
                    _ticket_tenant = binding.tenant_id
            ws_key_valid = False
            for k in _valid_api_keys:
                if hmac.compare_digest(provided, k):
                    ws_key_valid = True
            if not ws_key_valid:
                await ws.close(code=1008, reason="unauthorized")
                return
            if _api_key_tenant_map:
                if provided not in _api_key_tenant_map:
                    await ws.close(code=1008, reason="API key not bound to any tenant")
                    return
                ws_tenant_id = _api_key_tenant_map.get(provided, "")
                claimed = ws.headers.get("X-Tenant-ID", "")
                if claimed and ws_tenant_id and claimed != ws_tenant_id:
                    await ws.close(code=1008, reason="tenant mismatch")
                    return
        if not ws_tenant_id:
            ws_tenant_id = _ticket_tenant or ws.headers.get("X-Tenant-ID", "")

        client_ip = ws.client.host if ws.client else ""
        if not await _ws_admit(client_ip):
            await ws.close(code=1013, reason="server at capacity")
            return
        await ws.accept()

        send_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(_WS_MAX_CONCURRENT)
        active_tasks: dict[str, tuple[asyncio.Task[None], threading.Event]] = {}

        async def _send(payload: dict[str, Any]) -> None:
            """Serialise WebSocket writes through a per-connection lock."""
            async with send_lock:
                await ws.send_json(payload)

        async def _handle_session(session_id: str, data: dict[str, Any]) -> None:
            """Process one WebSocket session payload."""
            prompt = data.get("prompt", "")
            if len(prompt) > _WS_MAX_PROMPT_LENGTH:
                await _send(
                    {
                        "session_id": session_id,
                        "error": (f"prompt exceeds {_WS_MAX_PROMPT_LENGTH} chars"),
                    },
                )
                return

            sanitizer = ws.app.state._state.get("sanitizer")
            if sanitizer:
                check = sanitizer.check(prompt)
                if check.blocked:
                    await _send(
                        {
                            "session_id": session_id,
                            "error": f"injection rejected: {check.reason}",
                        },
                    )
                    return

            agent = ws.app.state._state.get("agent")
            if not agent:
                await _send({"session_id": session_id, "error": "server not ready"})
                return

            if data.get("streaming_oversight"):
                try:
                    from ..core import StreamingKernel

                    # Token-level pre-egress oversight. The WebSocket consumes
                    # ``agent.stream()`` (per-token coherence) and applies its own
                    # kernel as the egress gate: each token is scored *before* it
                    # is forwarded, and the token that trips the halt is never
                    # delivered. This is genuine pre-delivery interception, not
                    # scoring a fully generated answer after the fact.
                    kernel = StreamingKernel(
                        hard_limit=cfg.hard_limit,
                        window_size=getattr(cfg, "window_size", 5),
                        window_threshold=getattr(cfg, "window_threshold", 0.5),
                    )
                    kernel.reset_state()
                    cancel_event = data["_cancel_event"]
                    delivered = 0
                    halted = False
                    last_score = 1.0
                    async for token, score in agent.stream(
                        prompt,
                        tenant_id=ws_tenant_id,
                    ):
                        if cancel_event.is_set():
                            return
                        last_score = score
                        if kernel.check_halt(score):
                            # Pre-egress: suppress the halting token entirely.
                            halted = True
                            break
                        await _send(
                            {
                                "session_id": session_id,
                                "type": "token",
                                "token": token,
                                "coherence": round(score, 4),
                            },
                        )
                        delivered += 1
                    await _send(
                        {
                            "session_id": session_id,
                            "type": "halt" if halted else "complete",
                            "halted": halted,
                            "tokens_delivered": delivered,
                            "coherence": round(last_score, 4),
                            **({"reason": "coherence_halt"} if halted else {}),
                        },
                    )
                except (
                    RuntimeError,
                    ValueError,
                    TypeError,
                    OSError,
                ) as exc:  # pragma: no cover — defensive: websocket transport failure
                    logger.error("WebSocket streaming failed: %s", exc)
                    await _send(
                        {"session_id": session_id, "error": "streaming failed"},
                    )
                return

            try:
                cancel_event = data["_cancel_event"]
                result = await agent.aprocess(
                    prompt,
                    tenant_id=ws_tenant_id,
                    cancel_event=cancel_event,
                )
                if cancel_event.is_set():
                    return
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                logger.error("WebSocket agent.process() failed: %s", exc)
                await _send(
                    {"session_id": session_id, "error": "processing failed"},
                )
                return
            await _send(
                {
                    "session_id": session_id,
                    "type": "result",
                    "output": result.output,
                    "coherence": (result.coherence.score if result.coherence else None),
                    "halted": result.halted,
                    "warning": (
                        result.coherence.warning if result.coherence else False
                    ),
                    "fallback_used": result.fallback_used,
                    "evidence": _evidence_to_dict(
                        result.coherence.evidence if result.coherence else None,
                    ),
                    "halt_evidence": _halt_evidence_to_dict(result.halt_evidence),
                },
            )

        async def _run_session(session_id: str, data: dict[str, Any]) -> None:
            """Run a session under the per-connection concurrency limit."""
            async with semaphore:
                try:
                    await _handle_session(session_id, data)
                finally:
                    active_tasks.pop(session_id, None)

        conn_start = time.monotonic()
        msg_window: list[float] = []
        processed_chars = 0
        try:
            while True:
                if time.monotonic() - conn_start > _WS_MAX_LIFETIME_S:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "lifetime_exceeded"}
                    )
                    await ws.close(code=1001, reason="session lifetime exceeded")
                    break

                try:
                    data = await asyncio.wait_for(
                        ws.receive_json(), timeout=_WS_IDLE_TIMEOUT_S
                    )
                except TimeoutError:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "idle_timeout"}
                    )
                    await ws.close(code=1001, reason="idle timeout")
                    break
                except (ValueError, KeyError) as exc:
                    logger.warning("WebSocket bad JSON: %s", exc)
                    await _send({"error": "invalid JSON"})
                    continue

                now = time.monotonic()
                msg_window.append(now)
                msg_window[:] = [t for t in msg_window if now - t <= _WS_RATE_WINDOW_S]
                if len(msg_window) > _WS_MAX_MSGS_PER_WINDOW:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "rate_limited"}
                    )
                    await _send({"error": "message rate limit exceeded"})
                    continue

                if not isinstance(data, dict):
                    await _send({"error": "expected JSON object"})
                    continue

                # Cancel action
                action = data.get("action", "")
                if action == "cancel":
                    cancel_sid = data.get("session_id", "")
                    active = active_tasks.get(cancel_sid)
                    if active:
                        task, cancel_event = active
                        cancel_event.set()
                        task.cancel()
                        active_tasks.pop(cancel_sid, None)
                    await _send({"session_id": cancel_sid, "type": "cancelled"})
                    continue

                prompt = data.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    await _send({"error": "prompt must be a non-empty string"})
                    continue

                if len(prompt) > _WS_MAX_PROMPT_LENGTH:
                    await _send(
                        {"error": f"prompt exceeds {_WS_MAX_PROMPT_LENGTH} chars"},
                    )
                    continue

                processed_chars += len(prompt)
                if processed_chars > _WS_CONN_CHAR_BUDGET:
                    metrics.inc_labeled(
                        "ws_rejections_total", {"reason": "budget_exceeded"}
                    )
                    await ws.close(code=1009, reason="connection budget exceeded")
                    break

                session_id = data.get("session_id") or str(uuid.uuid4())
                if session_id in active_tasks:
                    await _send(
                        {
                            "session_id": session_id,
                            "error": "session already active",
                        },
                    )
                    continue
                if len(active_tasks) >= _WS_MAX_CONCURRENT:
                    await _send(
                        {
                            "session_id": session_id,
                            "error": "too many active sessions",
                        },
                    )
                    continue
                data["_cancel_event"] = threading.Event()
                task = asyncio.create_task(_run_session(session_id, data))
                active_tasks[session_id] = (task, data["_cancel_event"])

        except WebSocketDisconnect:
            pass
        finally:
            for task, cancel_event in active_tasks.values():
                cancel_event.set()
                task.cancel()
            await _ws_release(client_ip)

    return router
