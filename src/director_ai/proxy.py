# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — OpenAI-Compatible Proxy Server

"""OpenAI-compatible guardrail proxy.

Set ``OPENAI_BASE_URL=http://localhost:8080/v1`` and get transparent
hallucination scoring with zero code changes::

    director-ai proxy --port 8080 --facts kb.txt --threshold 0.6
"""

from __future__ import annotations

import hmac
import json
import logging
import pathlib
import time as _time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import TYPE_CHECKING, Any

from director_ai.core import CoherenceScorer, GroundTruthStore

# FastAPI resolves route-handler annotations at runtime (``get_type_hints``), so
# the types named in handler signatures must exist in module globals — not only
# under ``TYPE_CHECKING``. Import them at module level, degrading gracefully when
# the optional ``[server]`` extra is absent (the proxy still imports; building an
# app raises a clear ImportError instead).
try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:  # pragma: no cover — exercised only without the server extra
    pass

if TYPE_CHECKING:
    import httpx

    from director_ai.core.config import DirectorConfig

_log = logging.getLogger("DirectorAI.Proxy")

STREAM_CHECK_INTERVAL = 8

# Buffered-mode safety cap on the withheld ``pending`` window (KIMI3-H2). A
# stream can withhold many lines without ever producing reviewable content — a
# flood of content-less or non-data lines — which would grow the process memory
# without bound and never trigger a review. When the withheld window exceeds
# either cap the stream fails closed with a halt instead of buffering
# indefinitely; nothing withheld is disclosed.
STREAM_MAX_PENDING_LINES = 512
STREAM_MAX_PENDING_CHARS = 256 * 1024

# Streaming disclosure modes (KIMI2-A). ``immediate`` forwards every chunk as
# it arrives and a mid-stream halt only stops FUTURE tokens — content emitted
# before the halt has already reached the client (early termination with
# partial disclosure). ``buffered`` withholds chunks until they pass a review,
# so a halted stream discloses nothing unreviewed, at a latency cost of up to
# ``STREAM_CHECK_INTERVAL`` chunks.
STREAM_DISCLOSURE_MODES = ("immediate", "buffered")


def _chat_completion_content(data: object) -> str:
    """Extract OpenAI-compatible chat content without exception control flow."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _iter_choice_deltas(chunk: object) -> Iterator[object]:
    """Yield every choice's ``delta`` mapping in wire order.

    A chat completion may carry more than one choice (``n>1``); reviewing
    only ``choices[0]`` lets a sensitive payload on a later choice reach
    the client unreviewed, so every disclosure path walks the full list.
    """
    if not isinstance(chunk, dict):
        return
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if isinstance(choice, dict):
            yield choice.get("delta")


def _delta_content(delta: object) -> str:
    """Text content from a single delta mapping."""
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _delta_tool_call_text(delta: object) -> str:
    """Tool-call name and argument text from a single delta mapping.

    OpenAI tool-call streams carry their payload in
    ``delta.tool_calls[].function.{name,arguments}`` rather than
    ``delta.content``; a response that only calls tools still discloses
    model output (the tool it invokes and the arguments it passes) that
    must reach the review buffer.
    """
    if not isinstance(delta, dict):
        return ""
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if isinstance(name, str) and name:
            parts.append(name)
        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            parts.append(arguments)
    return "".join(parts)


def _stream_delta_content(chunk: object) -> str:
    """Extract OpenAI-compatible stream delta content across every choice."""
    return "".join(_delta_content(delta) for delta in _iter_choice_deltas(chunk))


def _stream_tool_call_content(chunk: object) -> str:
    """Extract tool-call name and argument deltas across every choice."""
    return "".join(_delta_tool_call_text(delta) for delta in _iter_choice_deltas(chunk))


def _stream_chat_content(chunk: object) -> str:
    """Reviewable chat-stream text across ALL choices in wire order.

    For each choice's delta, in choice order, the message content and the
    tool-call name/argument deltas are folded into one reviewed string.
    Walking every choice keeps a multi-choice chunk whose sensitive tool
    call rides on a later choice — or whose first choice is empty — from
    leaving the review buffer empty and bypassing the terminal review.
    """
    parts: list[str] = []
    for delta in _iter_choice_deltas(chunk):
        parts.append(_delta_content(delta))
        parts.append(_delta_tool_call_text(delta))
    return "".join(parts)


def _completion_text(data: object) -> str:
    """Extract legacy ``/v1/completions`` text without exception control flow."""
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _stream_text_content(chunk: object) -> str:
    """Extract a legacy completions stream text delta."""
    if not isinstance(chunk, dict):
        return ""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    text = first.get("text")
    return text if isinstance(text, str) else ""


def _completion_prompt(body: dict[str, Any]) -> str:
    """Extract the legacy ``prompt`` field (string or list of strings)."""
    prompt = body.get("prompt", "")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
        return prompt[0]
    return ""


def create_proxy_app(
    threshold: float = 0.6,
    facts_path: str | None = None,
    facts_root: str | None = None,
    upstream_url: str = "https://api.openai.com",
    on_fail: str = "reject",
    use_nli: bool | None = None,
    api_keys: list[str] | None = None,
    allow_http_upstream: bool = False,
    audit_db: str | None = None,
    config: DirectorConfig | None = None,
    moderations: str = "local",
    stream_disclosure: str = "immediate",
    _transport: Any = None,
) -> FastAPI:
    """Build a FastAPI app that proxies OpenAI requests with scoring.

    Parameters
    ----------
    threshold : float
        Coherence threshold below which responses are flagged.
    facts_path : str | None
        Path to a ``key: value`` facts file (one per line).
    facts_root : str | None
        Allowed root directory for ``facts_path``. When set, the
        resolved ``facts_path`` (with symlinks followed) must lie
        inside ``facts_root``; otherwise :class:`ValueError` is raised.
        Leave ``None`` for CLI/operator use; set in production
        deployments where ``facts_path`` is derived from untrusted
        configuration.
    upstream_url : str
        Base URL of the upstream OpenAI-compatible API.
    on_fail : str
        ``"reject"`` returns 422 on hallucination. ``"warn"`` forwards
        the response with warning headers.
    use_nli : bool | None
        Enable NLI model. ``None`` auto-detects.
    api_keys : list[str] | None
        Required API keys. Clients must send ``X-API-Key`` header.
        ``None`` or empty = no auth (not recommended for production).
    allow_http_upstream : bool
        Allow non-HTTPS upstream URLs. Default ``False`` rejects them.
    audit_db : str | None
        Path to SQLite compliance audit database. None disables audit logging.
    config
        Optional DirectorConfig. When provided, the proxy builds the configured
        store and scorer instead of the minimal in-memory scorer.
    moderations : str
        ``"local"`` serves ``/v1/moderations`` from the shipped
        dependency-free detectors; ``"upstream"`` forwards the request
        to the upstream endpoint verbatim.
    stream_disclosure : str
        ``"immediate"`` (default) forwards every streamed chunk as it
        arrives; a mid-stream halt stops FUTURE tokens only, so content
        emitted before the halt has already reached the client — early
        termination with partial disclosure. ``"buffered"`` withholds
        chunks until they pass a review and discards the unreleased
        buffer on a halt, so a rejected stream discloses nothing
        unreviewed (adds up to ``STREAM_CHECK_INTERVAL`` chunks of
        latency; meaningful with ``on_fail="reject"``).

    """
    from contextlib import asynccontextmanager

    import httpx
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from director_ai._proxy_moderations import (
        MODERATION_MODES,
        register_moderations_route,
    )

    if on_fail not in ("reject", "warn"):
        raise ValueError(f"on_fail must be 'reject' or 'warn', got {on_fail!r}")

    if moderations not in MODERATION_MODES:
        raise ValueError(
            f"moderations must be one of {MODERATION_MODES}, got {moderations!r}",
        )

    if stream_disclosure not in STREAM_DISCLOSURE_MODES:
        raise ValueError(
            f"stream_disclosure must be one of {STREAM_DISCLOSURE_MODES}, "
            f"got {stream_disclosure!r}",
        )

    if upstream_url and not upstream_url.startswith("https://"):
        if not allow_http_upstream:
            raise ValueError(
                f"Non-HTTPS upstream URL: {upstream_url!r}. "
                "Pass allow_http_upstream=True to override.",
            )
        _log.warning("Non-HTTPS upstream: %s", upstream_url)

    if config is None:
        store = GroundTruthStore()
        if facts_path:
            _load_facts(store, facts_path, facts_root=facts_root)
        scorer = CoherenceScorer(
            threshold=threshold,
            ground_truth_store=store,
            use_nli=use_nli,
        )
    else:
        from dataclasses import replace

        cfg = replace(config, coherence_threshold=threshold)
        if use_nli is not None:
            cfg = replace(cfg, use_nli=use_nli)
        store = cfg.build_store()
        if facts_path:
            _load_facts(store, facts_path, facts_root=facts_root)
        scorer = cfg.build_scorer(store=store)

    audit_log = None
    if audit_db:
        from director_ai.compliance.audit_log import AuditLog
        from director_ai.core.redactor import PIIRedactor

        # SEC-2: mask PII before it is sealed into the durable compliance chain
        # when redact_pii is on; a disabled redactor is a passthrough.
        # KIMI2-C: audit_strict_mode fails construction closed (redactor +
        # durable HMAC secret required) instead of warning.
        audit_log = AuditLog(
            audit_db,
            redactor=PIIRedactor(enabled=getattr(config, "redact_pii", False)),
            strict_mode=getattr(config, "audit_strict_mode", False),
        )
        _log.info("Compliance audit log: %s", audit_db)

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if audit_log is not None:
            audit_log.close()

    app = FastAPI(title="Director-AI Proxy", lifespan=_lifespan)
    upstream = upstream_url.rstrip("/")

    if not api_keys:
        _log.warning(
            "Proxy running WITHOUT authentication. Set api_keys for production use."
        )
    else:

        @app.middleware("http")
        async def _auth_middleware(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            if request.url.path == "/health":
                return await call_next(request)
            provided = request.headers.get("X-API-Key", "")
            if not any(hmac.compare_digest(provided, k) for k in api_keys):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Invalid or missing API key",
                            "type": "auth_error",
                        },
                    },
                )
            return await call_next(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "threshold": threshold, "on_fail": on_fail}

    def _client(**kw: Any) -> httpx.AsyncClient:
        if _transport is not None:
            kw["transport"] = _transport
        return httpx.AsyncClient(**kw)

    @app.get("/v1/models")
    async def proxy_models(request: Request) -> JSONResponse:
        async with _client() as client:
            headers = _forward_headers(request)
            resp = await client.get(f"{upstream}/v1/models", headers=headers)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @app.post("/v1/embeddings")
    async def proxy_embeddings(request: Request) -> JSONResponse:
        # Embeddings carry no natural-language claims to verify — plain
        # passthrough so OPENAI_BASE_URL can point every client here.
        body = await request.json()
        async with _client(timeout=120.0) as client:
            resp = await client.post(
                f"{upstream}/v1/embeddings",
                json=body,
                headers=_forward_headers(request),
            )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @app.post("/v1/completions")
    async def proxy_completions(request: Request) -> Response:
        body = await request.json()
        prompt = _completion_prompt(body)
        headers = _forward_headers(request)

        if body.get("stream", False):
            return await _handle_streaming(
                body,
                headers,
                upstream,
                prompt,
                scorer,
                on_fail,
                _transport,
                audit_log=audit_log,
                endpoint="/v1/completions",
                task_type="completion",
                stream_disclosure=stream_disclosure,
            )

        async with _client(timeout=120.0) as client:
            resp = await client.post(
                f"{upstream}/v1/completions",
                json=body,
                headers=headers,
            )

        if resp.status_code != 200:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

        data = resp.json()
        text = _completion_text(data)

        if not text:
            return JSONResponse(content=data)

        t0 = _time.monotonic()
        try:
            approved, cs = scorer.review(prompt, text)
        except Exception:  # noqa: BLE001 - any scorer error fails closed
            return _scorer_error_response(
                audit_log,
                prompt,
                text,
                model=body.get("model", "unknown"),
                task_type="completion",
                t0=t0,
            )
        latency_ms = (_time.monotonic() - t0) * 1000
        extra_headers = {
            "X-Director-Score": f"{cs.score:.4f}",
            "X-Director-Approved": str(approved).lower(),
        }

        _audit_log_entry(
            audit_log,
            prompt,
            text,
            model=body.get("model", "unknown"),
            score=cs.score,
            approved=approved,
            confidence=getattr(cs, "verdict_confidence", 0.0),
            latency_ms=latency_ms,
            task_type="completion",
        )

        if not approved and on_fail == "reject":
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "message": "Hallucination detected by Director-AI",
                        "type": "content_filter",
                        "score": cs.score,
                        "threshold": threshold,
                    },
                },
                headers=extra_headers,
            )

        return JSONResponse(content=data, headers=extra_headers)

    register_moderations_route(
        app,
        mode=moderations,
        upstream=upstream,
        client_factory=_client,
        forward_headers=_forward_headers,
    )

    @app.post("/v1/chat/completions")
    async def proxy_chat(request: Request) -> Response:
        body = await request.json()
        messages = body.get("messages", [])
        prompt = _extract_prompt(messages)
        streaming = body.get("stream", False)
        headers = _forward_headers(request)

        if streaming:
            return await _handle_streaming(
                body,
                headers,
                upstream,
                prompt,
                scorer,
                on_fail,
                _transport,
                audit_log=audit_log,
                stream_disclosure=stream_disclosure,
            )

        async with _client(timeout=120.0) as client:
            resp = await client.post(
                f"{upstream}/v1/chat/completions",
                json=body,
                headers=headers,
            )

        if resp.status_code != 200:
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

        data = resp.json()
        content = _chat_completion_content(data)

        if not content:
            return JSONResponse(content=data)

        t0 = _time.monotonic()
        try:
            approved, cs = scorer.review(prompt, content)
        except Exception:  # noqa: BLE001 - any scorer error fails closed
            return _scorer_error_response(
                audit_log,
                prompt,
                content,
                model=body.get("model", "unknown"),
                task_type="chat",
                t0=t0,
            )
        latency_ms = (_time.monotonic() - t0) * 1000
        extra_headers = {
            "X-Director-Score": f"{cs.score:.4f}",
            "X-Director-Approved": str(approved).lower(),
        }

        _audit_log_entry(
            audit_log,
            prompt,
            content,
            model=body.get("model", "unknown"),
            score=cs.score,
            approved=approved,
            confidence=getattr(cs, "verdict_confidence", 0.0),
            latency_ms=latency_ms,
        )

        if not approved and on_fail == "reject":
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "message": "Hallucination detected by Director-AI",
                        "type": "content_filter",
                        "score": cs.score,
                        "threshold": threshold,
                    },
                },
                headers=extra_headers,
            )

        return JSONResponse(content=data, headers=extra_headers)

    return app


async def _handle_streaming(
    body: dict[str, Any],
    headers: dict[str, str],
    upstream: str,
    prompt: str,
    scorer: Any,
    on_fail: str,
    transport: Any = None,
    audit_log: Any = None,
    endpoint: str = "/v1/chat/completions",
    task_type: str = "chat",
    stream_disclosure: str = "immediate",
) -> StreamingResponse:
    """Proxy an SSE stream with periodic mid-stream reviews.

    ``stream_disclosure="immediate"`` forwards each line as it arrives; a
    halt stops future tokens only (partial disclosure — content already
    emitted has reached the client). ``"buffered"`` holds lines in a
    pending window and releases them only after the accumulated content
    passes a review; a halt discards the unreleased window, and the
    terminal ``[DONE]`` review gates the final release, so a rejected
    stream never discloses unreviewed content. If the upstream drops
    without ``[DONE]``, the unreviewed pending window is discarded
    (fail-closed).
    """
    import httpx
    from fastapi.responses import StreamingResponse

    # Legacy completions stream text deltas at ``choices[0].text`` and a
    # halt chunk must mirror that shape; chat streams use ``delta`` and may
    # carry tool-call deltas that must be reviewed alongside content.
    legacy = endpoint == "/v1/completions"
    extract_content = _stream_text_content if legacy else _stream_chat_content
    halt_choice: dict[str, Any] = {"finish_reason": "content_filter", "index": 0}
    if legacy:
        halt_choice["text"] = ""
    else:
        halt_choice["delta"] = {}
    buffered = stream_disclosure == "buffered"

    async def _stream() -> AsyncIterator[str]:
        buffer: list[str] = []
        pending: list[str] = []  # withheld output lines (buffered mode)
        chunk_count = 0
        model_name = body.get("model", "unknown")
        t0 = _time.monotonic()

        def _pending_overflow() -> bool:
            """Report whether the withheld window has exceeded either cap."""
            return (
                len(pending) > STREAM_MAX_PENDING_LINES
                or sum(len(entry) for entry in pending) > STREAM_MAX_PENDING_CHARS
            )

        def _withhold(text: str) -> bool:
            """Hold *text* in the buffered window; return True on overflow."""
            pending.append(text)
            return _pending_overflow()

        def _fail_closed_frames() -> list[str]:
            """Drop the withheld window and return the halt frames.

            The shared fail-closed action for a pending-window overflow
            (KIMI3-H2) or a scorer exception mid-stream (KIMI3-H5): nothing
            withheld is disclosed — the window is cleared and the client receives
            only the halt marker, with the abort recorded to the audit log like
            any other terminal halt.
            """
            reviewed = "".join(buffer)
            _audit_log_entry(
                audit_log,
                prompt,
                reviewed,
                model=model_name,
                score=0.0,
                approved=False,
                confidence=0.0,
                latency_ms=(_time.monotonic() - t0) * 1000,
                task_type=task_type,
            )
            pending.clear()
            halt = {"choices": [dict(halt_choice)]}
            return [f"data: {json.dumps(halt)}\n", "data: [DONE]\n"]

        async with (
            httpx.AsyncClient(timeout=120.0, transport=transport) as client,
            client.stream(
                "POST",
                f"{upstream}{endpoint}",
                json=body,
                headers=headers,
            ) as resp,
        ):
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    # Non-data lines are withheld too in buffered mode so
                    # the released stream preserves the upstream ordering.
                    if buffered:
                        if _withhold(line + "\n"):
                            for _frame in _fail_closed_frames():
                                yield _frame
                            return
                    else:
                        yield line + "\n"
                    continue

                payload = line[6:]
                if payload.strip() == "[DONE]":
                    text = "".join(buffer)
                    if text:
                        try:
                            approved, _cs = scorer.review(prompt, text)
                        except Exception:  # noqa: BLE001 - any scorer error fails closed
                            _log.exception(
                                "scorer.review failed at stream end; failing closed"
                            )
                            for _frame in _fail_closed_frames():
                                yield _frame
                            return
                        latency_ms = (_time.monotonic() - t0) * 1000
                        _audit_log_entry(
                            audit_log,
                            prompt,
                            text,
                            model=model_name,
                            score=_cs.score,
                            approved=approved,
                            confidence=getattr(_cs, "verdict_confidence", 0.0),
                            latency_ms=latency_ms,
                            task_type=task_type,
                        )
                        if not approved and on_fail == "reject":
                            # Buffered: the withheld window is discarded, so
                            # the client receives only the halt marker.
                            pending.clear()
                            halt = {"choices": [dict(halt_choice)]}
                            yield f"data: {json.dumps(halt)}\n"
                            yield "data: [DONE]\n"
                            return
                    # Final review passed (or nothing to review): release
                    # any withheld tail before the terminal marker.
                    for held in pending:
                        yield held
                    pending.clear()
                    yield line + "\n"
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    if buffered:
                        if _withhold(line + "\n"):
                            for _frame in _fail_closed_frames():
                                yield _frame
                            return
                    else:
                        yield line + "\n"
                    continue

                delta = extract_content(chunk)

                if delta:
                    buffer.append(delta)
                    chunk_count += 1

                    if chunk_count % STREAM_CHECK_INTERVAL == 0:
                        text = "".join(buffer)
                        try:
                            approved, _cs = scorer.review(prompt, text)
                        except Exception:  # noqa: BLE001 - any scorer error fails closed
                            _log.exception(
                                "scorer.review failed mid-stream; failing closed"
                            )
                            for _frame in _fail_closed_frames():
                                yield _frame
                            return
                        if not approved and on_fail == "reject":
                            # Record the mid-stream halt: like the terminal
                            # [DONE] review, a rejection must leave an audit entry.
                            latency_ms = (_time.monotonic() - t0) * 1000
                            _audit_log_entry(
                                audit_log,
                                prompt,
                                text,
                                model=model_name,
                                score=_cs.score,
                                approved=approved,
                                confidence=getattr(_cs, "verdict_confidence", 0.0),
                                latency_ms=latency_ms,
                                task_type=task_type,
                            )
                            pending.clear()
                            halt = {"choices": [dict(halt_choice)]}
                            yield f"data: {json.dumps(halt)}\n"
                            yield "data: [DONE]\n"
                            return
                        if buffered:
                            # Reviewed clean: release the withheld window
                            # (including this line) up to the review point.
                            pending.append(line + "\n")
                            for held in pending:
                                yield held
                            pending.clear()
                            continue

                if buffered:
                    if _withhold(line + "\n"):
                        for _frame in _fail_closed_frames():
                            yield _frame
                        return
                else:
                    yield line + "\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


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


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return str(block.get("text", ""))
            return str(content)
    return ""


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


def _load_facts(
    store: GroundTruthStore,
    path: str,
    *,
    facts_root: str | None = None,
) -> None:
    try:
        resolved = pathlib.Path(path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Facts file not found: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Facts file not found: {path}")
    if facts_root is not None:
        try:
            root_resolved = pathlib.Path(facts_root).resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"facts_root not found: {facts_root}") from exc
        if not root_resolved.is_dir():
            raise ValueError(f"facts_root must be a directory: {facts_root}")
        if not resolved.is_relative_to(root_resolved):
            raise ValueError(
                f"facts_path {resolved} is outside facts_root {root_resolved}"
            )
    with open(resolved, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                store.add(key.strip(), value.strip())
            else:
                store.add(line[:30], line)
