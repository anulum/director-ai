# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - middleware real-surface tests
"""Real ASGI coverage for authentication and rate-limit middleware."""

from __future__ import annotations

from typing import cast

import httpx
import pytest

pytest.importorskip("fastapi", reason="fastapi required for middleware tests")

from fastapi import FastAPI, Request
from httpx import ASGITransport

from director_ai.middleware.api_key import APIKeyMiddleware
from director_ai.middleware.rate_limit import RateLimitMiddleware

_API_KEY = "sk-live-real-surface"


def _middleware_app() -> FastAPI:
    """Return an ASGI app with the production middleware chain mounted."""
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Return the unauthenticated health payload."""
        return {"status": "ok"}

    @app.get("/v1/score")
    async def score(request: Request) -> dict[str, object]:
        """Return request state populated by the auth middleware."""
        api_key_hash = cast(str, request.state.api_key_hash)
        return {"score": 0.5, "api_key_hash": api_key_hash}

    app.add_middleware(RateLimitMiddleware, requests_per_minute=60, burst=2)
    app.add_middleware(APIKeyMiddleware, keys={_API_KEY})
    return app


@pytest.mark.asyncio
async def test_api_key_and_rate_limit_middleware_chain_over_asgi() -> None:
    """Auth should populate request state before rate limiting consumes tokens."""
    app = _middleware_app()
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health_responses = [await client.get("/health") for _ in range(3)]
        missing_key_response = await client.get("/v1/score")
        bearer_response = await client.get(
            "/v1/score",
            headers={"Authorization": f"Bearer {_API_KEY}"},
        )
        x_api_key_response = await client.get(
            "/v1/score",
            headers={"X-API-Key": _API_KEY},
        )
        limited_response = await client.get(
            "/v1/score",
            headers={"X-API-Key": _API_KEY},
        )

    assert [response.status_code for response in health_responses] == [200, 200, 200]
    assert [response.json() for response in health_responses] == [
        {"status": "ok"},
        {"status": "ok"},
        {"status": "ok"},
    ]

    assert missing_key_response.status_code == 401
    assert missing_key_response.json() == {"error": "Invalid or missing API key"}

    assert bearer_response.status_code == 200, bearer_response.text
    bearer_payload = cast(dict[str, object], bearer_response.json())
    assert bearer_payload["score"] == 0.5
    assert len(cast(str, bearer_payload["api_key_hash"])) == 16

    assert x_api_key_response.status_code == 200, x_api_key_response.text
    x_api_key_payload = cast(dict[str, object], x_api_key_response.json())
    assert x_api_key_payload["api_key_hash"] == bearer_payload["api_key_hash"]

    assert limited_response.status_code == 429
    limited_payload = cast(dict[str, object], limited_response.json())
    assert limited_payload["error"] == "Rate limit exceeded"
    assert "Retry-After" in limited_response.headers
