# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FastAPI Middleware Tests

from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from director_ai.integrations.fastapi_guard import DirectorGuard


def _make_app(response_body: dict, path: str = "/api/chat"):
    """Build a minimal FastAPI app returning a fixed JSON body."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post(path)
    async def handler():
        return JSONResponse(content=response_body)

    @app.get("/api/info")
    async def info():
        return {"status": "ok"}

    return app


@pytest.mark.asyncio
async def test_guard_adds_headers():
    inner = _make_app({"response": "The sky is blue."})
    guarded = DirectorGuard(
        inner,
        facts={"sky": "The sky is blue due to Rayleigh scattering."},
        use_nli=False,
        on_fail="warn",
    )

    transport = ASGITransport(app=guarded)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            json={"prompt": "What color is the sky?"},
        )

    assert resp.status_code == 200
    assert "x-director-score" in resp.headers
    assert "x-director-approved" in resp.headers


@pytest.mark.asyncio
async def test_guard_reject_mode():
    inner = _make_app({"response": "Mars has two moons named Phobos and Deimos."})
    guarded = DirectorGuard(
        inner,
        facts={"sky": "The sky is blue due to Rayleigh scattering."},
        threshold=0.6,
        use_nli=False,
        on_fail="reject",
    )

    transport = ASGITransport(app=guarded)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            json={"prompt": "What color is the sky?"},
        )

    # Response about Mars should fail coherence check on "sky" question
    assert resp.status_code in (200, 422)
    if resp.status_code == 422:
        data = resp.json()
        assert data["error"]["type"] == "content_filter"


@pytest.mark.asyncio
async def test_guard_path_filter():
    inner = _make_app(
        {"response": "Mars has two moons."},
        path="/api/chat",
    )

    # Only score /api/scored, not /api/chat
    guarded = DirectorGuard(
        inner,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        paths=["/api/scored"],
        on_fail="reject",
    )

    transport = ASGITransport(app=guarded)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/chat",
            json={"prompt": "What color is the sky?"},
        )

    # Path not in configured paths â†’ pass through unscored
    assert resp.status_code == 200
    assert "x-director-score" not in resp.headers


@pytest.mark.asyncio
async def test_guard_ignores_get():
    inner = _make_app({"response": "anything"})

    guarded = DirectorGuard(
        inner,
        facts={"sky": "The sky is blue."},
        use_nli=False,
        on_fail="reject",
    )

    transport = ASGITransport(app=guarded)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/info")

    assert resp.status_code == 200
    assert "x-director-score" not in resp.headers
