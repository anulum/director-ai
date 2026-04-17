# SPDX-License-Identifier: AGPL-3.0-or-later
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

import contextlib
import hmac
import json
import logging
import pathlib
import time as _time

from director_ai.core import CoherenceScorer, GroundTruthStore

_log = logging.getLogger("DirectorAI.Proxy")

STREAM_CHECK_INTERVAL = 8


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
    _transport=None,
):
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

    """
    from contextlib import asynccontextmanager

    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    if on_fail not in ("reject", "warn"):
        raise ValueError(f"on_fail must be 'reject' or 'warn', got {on_fail!r}")

    if upstream_url and not upstream_url.startswith("https://"):
        if not allow_http_upstream:
            raise ValueError(
                f"Non-HTTPS upstream URL: {upstream_url!r}. "
                "Pass allow_http_upstream=True to override.",
            )
        _log.warning("Non-HTTPS upstream: %s", upstream_url)

    store = GroundTruthStore()
    if facts_path:
        _load_facts(store, facts_path, facts_root=facts_root)

    scorer = CoherenceScorer(
        threshold=threshold,
        ground_truth_store=store,
        use_nli=use_nli,
    )

    audit_log = None
    if audit_db:
        from director_ai.compliance.audit_log import AuditLog

        audit_log = AuditLog(audit_db)
        _log.info("Compliance audit log: %s", audit_db)

    @asynccontextmanager
    async def _lifespan(app):
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
        async def _auth_middleware(request: Request, call_next):
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
    async def health():
        return {"status": "ok", "threshold": threshold, "on_fail": on_fail}

    def _client(**kw):
        if _transport is not None:
            kw["transport"] = _transport
        return httpx.AsyncClient(**kw)

    @app.get("/v1/models")
    async def proxy_models(request: Request):
        async with _client() as client:
            headers = _forward_headers(request)
            resp = await client.get(f"{upstream}/v1/models", headers=headers)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)

    @app.post("/v1/chat/completions")
    async def proxy_chat(request: Request):
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
        content = ""
        with contextlib.suppress(KeyError, IndexError, TypeError):
            content = data["choices"][0]["message"]["content"] or ""

        if not content:
            return JSONResponse(content=data)

        t0 = _time.monotonic()
        approved, cs = scorer.review(prompt, content)
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
    body,
    headers,
    upstream,
    prompt,
    scorer,
    on_fail,
    transport=None,
    audit_log=None,
):
    import httpx
    from fastapi.responses import StreamingResponse

    async def _stream():
        buffer: list[str] = []
        chunk_count = 0
        model_name = body.get("model", "unknown")
        t0 = _time.monotonic()

        async with (
            httpx.AsyncClient(timeout=120.0, transport=transport) as client,
            client.stream(
                "POST",
                f"{upstream}/v1/chat/completions",
                json=body,
                headers=headers,
            ) as resp,
        ):
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    yield line + "\n"
                    continue

                payload = line[6:]
                if payload.strip() == "[DONE]":
                    text = "".join(buffer)
                    if text:
                        approved, _cs = scorer.review(prompt, text)
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
                        )
                        if not approved and on_fail == "reject":
                            halt = {
                                "choices": [
                                    {
                                        "delta": {},
                                        "finish_reason": "content_filter",
                                        "index": 0,
                                    },
                                ],
                            }
                            yield f"data: {json.dumps(halt)}\n"
                            yield "data: [DONE]\n"
                            return
                    yield line + "\n"
                    continue

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    yield line + "\n"
                    continue

                delta = ""
                with contextlib.suppress(KeyError, IndexError, TypeError):
                    delta = chunk["choices"][0]["delta"].get("content", "") or ""

                if delta:
                    buffer.append(delta)
                    chunk_count += 1

                    if chunk_count % STREAM_CHECK_INTERVAL == 0:
                        text = "".join(buffer)
                        approved, _cs = scorer.review(prompt, text)
                        if not approved and on_fail == "reject":
                            halt = {
                                "choices": [
                                    {
                                        "delta": {},
                                        "finish_reason": "content_filter",
                                        "index": 0,
                                    },
                                ],
                            }
                            yield f"data: {json.dumps(halt)}\n"
                            yield "data: [DONE]\n"
                            return

                yield line + "\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _audit_log_entry(
    audit_log,
    prompt: str,
    response: str,
    *,
    model: str,
    score: float,
    approved: bool,
    confidence: float,
    latency_ms: float,
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
            task_type="chat",
            domain="",
            latency_ms=latency_ms,
            timestamp=_time.time(),
        )
    )


def _extract_prompt(messages: list[dict]) -> str:
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


def _forward_headers(request) -> dict[str, str]:
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
