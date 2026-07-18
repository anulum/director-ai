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

This module is the app factory and route orchestration; the supporting
responsibilities live in sibling modules and are re-exported here so the
``director_ai.proxy`` namespace is unchanged: wire-format parsing in
``_proxy_content``, the streaming disclosure handler in
``_proxy_streaming``, audit/fail-closed plumbing in ``_proxy_audit``,
and facts-file loading in ``_proxy_facts``.
"""

from __future__ import annotations

import hmac
import logging
import time as _time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from director_ai._proxy_audit import (
    _audit_log_entry as _audit_log_entry,
)
from director_ai._proxy_audit import (
    _scorer_error_response as _scorer_error_response,
)
from director_ai._proxy_content import (
    _chat_completion_content as _chat_completion_content,
)
from director_ai._proxy_content import (
    _completion_prompt as _completion_prompt,
)
from director_ai._proxy_content import (
    _completion_text as _completion_text,
)
from director_ai._proxy_content import (
    _delta_content as _delta_content,
)
from director_ai._proxy_content import (
    _delta_tool_call_text as _delta_tool_call_text,
)
from director_ai._proxy_content import (
    _extract_prompt as _extract_prompt,
)
from director_ai._proxy_content import (
    _iter_choice_deltas as _iter_choice_deltas,
)
from director_ai._proxy_content import (
    _stream_chat_content as _stream_chat_content,
)
from director_ai._proxy_content import (
    _stream_delta_content as _stream_delta_content,
)
from director_ai._proxy_content import (
    _stream_text_content as _stream_text_content,
)
from director_ai._proxy_content import (
    _stream_tool_call_content as _stream_tool_call_content,
)
from director_ai._proxy_facts import _load_facts as _load_facts
from director_ai._proxy_streaming import (
    STREAM_CHECK_INTERVAL as STREAM_CHECK_INTERVAL,
)
from director_ai._proxy_streaming import (
    STREAM_DISCLOSURE_MODES as STREAM_DISCLOSURE_MODES,
)
from director_ai._proxy_streaming import (
    STREAM_MAX_PENDING_CHARS as STREAM_MAX_PENDING_CHARS,
)
from director_ai._proxy_streaming import (
    STREAM_MAX_PENDING_LINES as STREAM_MAX_PENDING_LINES,
)
from director_ai._proxy_streaming import (
    _handle_streaming as _handle_streaming,
)
from director_ai.core import CoherenceScorer, GroundTruthStore

# FastAPI resolves route-handler annotations at runtime (``get_type_hints``), so
# the types named in handler signatures must exist in module globals — not only
# under ``TYPE_CHECKING``. Import them at module level, degrading gracefully when
# the optional ``[server]`` extra is absent (the proxy still imports; building an
# app raises a clear ImportError instead).
try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover — exercised only without the server extra
    pass

if TYPE_CHECKING:
    import httpx

    from director_ai.core.config import DirectorConfig

_log = logging.getLogger("DirectorAI.Proxy")


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


def _forward_headers(request: Request) -> dict[str, str]:
    headers = {}
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    return headers
