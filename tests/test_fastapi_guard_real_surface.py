# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - FastAPI guard real-surface tests
"""Real ASGI/FastAPI coverage for the DirectorGuard middleware."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from director_ai.integrations.fastapi_guard import DirectorGuard

AsgiMessage = dict[str, Any]
Receive = Callable[[], Awaitable[AsgiMessage]]
Send = Callable[[AsgiMessage], Awaitable[None]]


def _openai_chat_app(content: str) -> FastAPI:
    """Return a FastAPI app with an OpenAI-compatible chat endpoint."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions() -> dict[str, list[dict[str, dict[str, str]]]]:
        return {"choices": [{"message": {"content": content}}]}

    @app.get("/v1/models")
    async def models() -> dict[str, list[str]]:
        return {"data": ["local-model"]}

    return app


@pytest.mark.asyncio
async def test_director_guard_scores_real_openai_chat_completion_surface() -> None:
    """DirectorGuard should score a real ASGI OpenAI-style chat exchange."""
    app = _openai_chat_app("The sky is blue due to Rayleigh scattering.")
    app.add_middleware(
        DirectorGuard,
        facts={"sky": "The sky is blue due to Rayleigh scattering."},
        use_nli=False,
        on_fail="warn",
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Answer from retrieved facts."},
                    {"role": "user", "content": "What colour is the sky?"},
                ],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "choices": [
            {"message": {"content": "The sky is blue due to Rayleigh scattering."}}
        ]
    }
    assert response.headers["x-director-approved"] in {"true", "false"}
    assert 0.0 <= float(response.headers["x-director-score"]) <= 1.0


@pytest.mark.asyncio
async def test_director_guard_rejects_real_policy_mismatch_surface() -> None:
    """DirectorGuard should reject mismatched responses through FastAPI."""
    app = _openai_chat_app("Mars has two moons named Phobos and Deimos.")
    app.add_middleware(
        DirectorGuard,
        facts={"sky": "The sky is blue due to Rayleigh scattering."},
        threshold=0.6,
        use_nli=False,
        on_fail="reject",
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"prompt": "What colour is the sky?"},
        )

    assert response.status_code == 422
    assert response.headers["x-director-approved"] == "false"
    assert response.json()["error"]["type"] == "content_filter"


@pytest.mark.asyncio
async def test_director_guard_passes_real_unscored_routes_through() -> None:
    """DirectorGuard should leave non-POST routes unscored."""
    app = _openai_chat_app("The sky is blue.")
    app.add_middleware(
        DirectorGuard,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        on_fail="reject",
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {"data": ["local-model"]}
    assert "x-director-score" not in response.headers


@pytest.mark.asyncio
async def test_director_guard_replays_real_chunked_asgi_body() -> None:
    """DirectorGuard should replay chunked request bodies to ASGI apps."""
    received: list[AsgiMessage] = []

    async def inner(
        scope: dict[str, Any],
        receive: Receive,
        send: Send,
    ) -> None:
        assert scope["type"] == "http"
        received.append(await receive())
        received.append(await receive())
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.trailers", "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": b'{"response": "The sky is blue."}',
            },
        )

    inbound_messages: list[AsgiMessage] = [
        {"type": "http.response.debug", "body": b"ignored"},
        {
            "type": "http.request",
            "body": b'{"prompt": "What colour',
            "more_body": True,
        },
        {"type": "http.request", "body": b' is the sky?"}', "more_body": False},
        {"type": "http.request", "body": b"after-replay", "more_body": False},
    ]
    inbound = iter(inbound_messages)
    sent: list[AsgiMessage] = []
    guarded = DirectorGuard(
        inner,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        on_fail="warn",
    )

    async def receive() -> AsgiMessage:
        return next(inbound)

    async def send(message: AsgiMessage) -> None:
        sent.append(message)

    await guarded(
        {"type": "http", "method": "POST", "path": "/api/chat"}, receive, send
    )

    assert received == [
        {
            "type": "http.request",
            "body": b'{"prompt": "What colour is the sky?"}',
            "more_body": False,
        },
        {"type": "http.request", "body": b"after-replay", "more_body": False},
    ]
    assert sent[0]["status"] == 200
    assert b"x-director-score" in dict(
        cast(list[tuple[bytes, bytes]], sent[0]["headers"])
    )
    assert sent[1]["body"] == b'{"response": "The sky is blue."}'


@pytest.mark.asyncio
async def test_director_guard_warns_on_detected_injection_without_rejecting() -> None:
    """DirectorGuard should surface injection headers in warn mode."""

    class InjectionResult:
        """Minimal detector result matching the production protocol."""

        injection_risk = 0.91
        injection_detected = True

    class Detector:
        """Protocol-preserving detector used behind the middleware boundary."""

        def detect(
            self,
            *,
            intent: str,
            response: str,
            user_query: str,
            system_prompt: str,
        ) -> InjectionResult:
            assert intent == user_query == "What colour is the sky?"
            assert response == "The sky is blue."
            assert system_prompt == "Answer from retrieved facts."
            return InjectionResult()

    app = _openai_chat_app("The sky is blue.")
    guarded = DirectorGuard(
        app,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        injection_detection=True,
        on_fail="warn",
    )
    guarded._injection_detector = cast(Any, Detector())

    transport = ASGITransport(app=cast(Any, guarded))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "Answer from retrieved facts."},
                    {"role": "user", "content": "What colour is the sky?"},
                ],
            },
        )

    assert response.status_code == 200
    assert response.headers["x-director-injection-detected"] == "true"
    assert response.headers["x-director-injection-risk"] == "0.9100"


@pytest.mark.asyncio
async def test_director_guard_finds_user_prompt_before_non_user_messages() -> None:
    """DirectorGuard should scan OpenAI messages backward to the latest user."""
    app = _openai_chat_app("The sky is blue.")
    app.add_middleware(
        DirectorGuard,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        on_fail="warn",
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What colour is the sky?"},
                    {"role": "assistant", "content": "Previous answer."},
                    ["not-a-message-object"],
                ],
            },
        )

    assert response.status_code == 200
    assert "x-director-score" in response.headers


@pytest.mark.asyncio
async def test_director_guard_scores_choice_fallback_content() -> None:
    """DirectorGuard should fall back when choices contain non-dict entries."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions() -> dict[str, object]:
        return {"choices": [None], "content": "The sky is blue."}

    app.add_middleware(
        DirectorGuard,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        on_fail="warn",
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "assistant", "content": "Previous answer."},
                    ["not-a-message-object"],
                ],
                "prompt": "What colour is the sky?",
            },
        )

    assert response.status_code == 200
    assert response.headers["x-director-approved"] in {"true", "false"}


@pytest.mark.asyncio
async def test_director_guard_scores_choice_message_fallback_text() -> None:
    """DirectorGuard should fall back when choice message is not an object."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat_completions() -> dict[str, object]:
        return {"choices": [{"message": "not-a-dict"}], "text": "The sky is blue."}

    app.add_middleware(
        DirectorGuard,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        on_fail="warn",
    )

    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"prompt": "What colour is the sky?"},
        )

    assert response.status_code == 200
    assert response.headers["x-director-approved"] in {"true", "false"}
