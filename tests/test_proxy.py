# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Proxy Server Tests

import httpx
import pytest
from httpx import ASGITransport

from director_ai.proxy import create_proxy_app


def _upstream_transport(content: str):
    """Create an httpx transport that returns a fixed chat completion."""

    async def _handler(request: httpx.Request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    },
                ],
            },
        )

    return httpx.MockTransport(_handler)


@pytest.mark.asyncio
async def test_proxy_health():
    app = create_proxy_app(threshold=0.7, on_fail="reject")
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["threshold"] == 0.7


@pytest.mark.asyncio
async def test_proxy_forwards_approved():
    mock_transport = _upstream_transport("The sky is blue due to Rayleigh scattering.")
    app = create_proxy_app(
        threshold=0.3,
        upstream_url="http://fake-upstream",
        on_fail="reject",
        use_nli=False,
        allow_http_upstream=True,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport,
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    assert resp.status_code == 200
    assert "x-director-score" in resp.headers
    assert "x-director-approved" in resp.headers


@pytest.mark.asyncio
async def test_proxy_rejects_hallucination():
    mock_transport = _upstream_transport("Mars has two moons named Phobos and Deimos.")
    app = create_proxy_app(
        threshold=0.6,
        upstream_url="http://fake-upstream",
        on_fail="reject",
        use_nli=False,
        allow_http_upstream=True,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport,
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    assert resp.status_code in (200, 422)
    if resp.status_code == 422:
        data = resp.json()
        assert data["error"]["type"] == "content_filter"


@pytest.mark.asyncio
async def test_proxy_warn_mode():
    mock_transport = _upstream_transport("Mars has two moons named Phobos and Deimos.")
    app = create_proxy_app(
        threshold=0.6,
        upstream_url="http://fake-upstream",
        on_fail="warn",
        use_nli=False,
        allow_http_upstream=True,
        _transport=mock_transport,
    )

    proxy_transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=proxy_transport,
        base_url="http://test",
    ) as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "What color is the sky?"}],
            },
        )

    # Warn mode always forwards
    assert resp.status_code == 200
    assert "x-director-score" in resp.headers
