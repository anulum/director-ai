# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — OpenAI-Compatible Proxy Server
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
OpenAI-compatible guardrail proxy.

Set ``OPENAI_BASE_URL=http://localhost:8080/v1`` and get transparent
hallucination scoring with zero code changes::

    director-ai proxy --port 8080 --facts kb.txt --threshold 0.6
"""

import contextlib
import json
import logging
import pathlib

from director_ai.core import CoherenceScorer, GroundTruthStore

_log = logging.getLogger("DirectorAI.Proxy")

STREAM_CHECK_INTERVAL = 8


def create_proxy_app(
    threshold: float = 0.6,
    facts_path: str | None = None,
    upstream_url: str = "https://api.openai.com",
    on_fail: str = "reject",
    use_nli: bool | None = None,
    _transport=None,
):
    """Build a FastAPI app that proxies OpenAI requests with scoring.

    Parameters
    ----------
    threshold : float
        Coherence threshold below which responses are flagged.
    facts_path : str | None
        Path to a ``key: value`` facts file (one per line).
    upstream_url : str
        Base URL of the upstream OpenAI-compatible API.
    on_fail : str
        ``"reject"`` returns 422 on hallucination. ``"warn"`` forwards
        the response with warning headers.
    use_nli : bool | None
        Enable NLI model. ``None`` auto-detects.
    """
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    if on_fail not in ("reject", "warn"):
        raise ValueError(f"on_fail must be 'reject' or 'warn', got {on_fail!r}")

    if upstream_url and not upstream_url.startswith("https://"):
        _log.warning("Upstream URL uses non-HTTPS scheme: %s", upstream_url)

    store = GroundTruthStore()
    if facts_path:
        _load_facts(store, facts_path)

    scorer = CoherenceScorer(
        threshold=threshold, ground_truth_store=store, use_nli=use_nli
    )

    app = FastAPI(title="Director-AI Proxy")
    upstream = upstream_url.rstrip("/")

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
                body, headers, upstream, prompt, scorer, on_fail, _transport
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

        approved, cs = scorer.review(prompt, content)
        extra_headers = {
            "X-Director-Score": f"{cs.score:.4f}",
            "X-Director-Approved": str(approved).lower(),
        }

        if not approved and on_fail == "reject":
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "message": "Hallucination detected by Director-AI",
                        "type": "content_filter",
                        "score": cs.score,
                        "threshold": threshold,
                    }
                },
                headers=extra_headers,
            )

        return JSONResponse(content=data, headers=extra_headers)

    return app


async def _handle_streaming(
    body, headers, upstream, prompt, scorer, on_fail, transport=None
):
    import httpx
    from fastapi.responses import StreamingResponse

    async def _stream():
        buffer: list[str] = []
        chunk_count = 0

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
                        if not approved and on_fail == "reject":
                            halt = {
                                "choices": [
                                    {
                                        "delta": {},
                                        "finish_reason": "content_filter",
                                        "index": 0,
                                    }
                                ]
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
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(halt)}\n"
                            yield "data: [DONE]\n"
                            return

                yield line + "\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


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


def _load_facts(store: GroundTruthStore, path: str) -> None:
    resolved = pathlib.Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Facts file not found: {path}")
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
