# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — proxy real-surface tests
"""Real ASGI-route coverage for the OpenAI-compatible proxy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for proxy surface tests")

import httpx
from httpx import ASGITransport

from director_ai.compliance.audit_log import AuditLog
from director_ai.proxy import create_proxy_app
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _chat_request(*, stream: bool = False) -> dict[str, object]:
    """Return a realistic OpenAI-compatible chat-completion request."""
    payload: dict[str, object] = {
        "model": "local-proxy-model",
        "messages": [
            {"role": "user", "content": "first prompt"},
            {"role": "assistant", "content": "first answer"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "file://ignored.png"}},
                    {"type": "text", "text": "caption prompt"},
                ],
            },
        ],
    }
    if stream:
        payload["stream"] = True
        payload["model"] = "stream-proxy-model"
    return payload


def _audit_entries(db_path: Path) -> list[str]:
    """Return audit-log prompt/response pairs as compact strings."""
    log = AuditLog(db_path)
    try:
        return [f"{entry.prompt}|{entry.response}" for entry in log.query()]
    finally:
        log.close()


def test_proxy_unit_guard_has_real_surface_companion() -> None:
    """The proxy unit guard should be backed by real ASGI route coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS["tests/test_proxy.py"]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_proxy_real_surface.py" in category


@pytest.mark.asyncio
async def test_proxy_health_auth_and_models_route_use_public_asgi_surface() -> None:
    """Health, API-key middleware, and model forwarding should share one app."""
    forwarded_authorization: list[str] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        forwarded_authorization.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={"data": [{"id": "model-a"}]})

    app = create_proxy_app(
        upstream_url="http://upstream.local",
        allow_http_upstream=True,
        api_keys=["proxy-secret"],
        _transport=httpx.MockTransport(_handler),
    )
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://proxy.local",
    ) as client:
        health = await client.get("/health")
        unauthorized = await client.get("/v1/models")
        authorized = await client.get(
            "/v1/models",
            headers={
                "X-API-Key": "proxy-secret",
                "Authorization": "Bearer upstream-token",
            },
        )

    assert health.status_code == 200
    assert cast(dict[str, object], health.json())["status"] == "ok"
    assert unauthorized.status_code == 401
    assert cast(dict[str, object], unauthorized.json())["error"] == {
        "message": "Invalid or missing API key",
        "type": "auth_error",
    }
    assert authorized.status_code == 200
    assert cast(dict[str, object], authorized.json())["data"] == [{"id": "model-a"}]
    assert forwarded_authorization == ["Bearer upstream-token"]


@pytest.mark.asyncio
async def test_proxy_chat_route_scores_and_audits_structured_messages(
    tmp_path: Path,
) -> None:
    """Non-streaming chat should score and audit the extracted user prompt."""
    upstream_payloads: list[dict[str, object]] = []
    audit_db = tmp_path / "proxy-chat-audit.db"

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        upstream_payloads.append(cast(dict[str, object], json.loads(request.content)))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-proxy-real",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The image caption says the sky is blue.",
                        },
                        "finish_reason": "stop",
                    },
                ],
            },
        )

    app = create_proxy_app(
        threshold=0.0,
        upstream_url="http://upstream.local",
        allow_http_upstream=True,
        on_fail="warn",
        use_nli=False,
        audit_db=str(audit_db),
        _transport=httpx.MockTransport(_handler),
    )
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://proxy.local",
    ) as client:
        response = await client.post("/v1/chat/completions", json=_chat_request())

    assert response.status_code == 200
    assert response.headers["x-director-approved"] in {"true", "false"}
    assert float(response.headers["x-director-score"]) >= 0.0
    payload = cast(dict[str, object], response.json())
    assert payload["id"] == "chatcmpl-proxy-real"
    assert upstream_payloads == [_chat_request()]
    assert _audit_entries(audit_db) == [
        "caption prompt|The image caption says the sky is blue."
    ]


@pytest.mark.asyncio
async def test_proxy_streaming_route_scores_sse_and_audits_final_text(
    tmp_path: Path,
) -> None:
    """Streaming chat should forward SSE data and audit the assembled text."""
    audit_db = tmp_path / "proxy-stream-audit.db"

    async def _handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        request_payload = cast(dict[str, object], json.loads(request.content))
        assert request_payload["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"blue "}}]}',
                    'data: {"choices":[{"delta":{"content":"sky"}}]}',
                    "data: [DONE]",
                ]
            )
            + "\n",
        )

    app = create_proxy_app(
        threshold=0.0,
        upstream_url="http://upstream.local",
        allow_http_upstream=True,
        on_fail="warn",
        use_nli=False,
        audit_db=str(audit_db),
        _transport=httpx.MockTransport(_handler),
    )
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://proxy.local",
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_chat_request(stream=True),
        )

    assert response.status_code == 200
    assert 'data: {"choices":[{"delta":{"content":"blue "}}]}' in response.text
    assert 'data: {"choices":[{"delta":{"content":"sky"}}]}' in response.text
    assert "data: [DONE]" in response.text
    assert _audit_entries(audit_db) == ["caption prompt|blue sky"]
