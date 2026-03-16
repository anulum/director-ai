# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Proxy Server Tests (streaming, auth, warn mode)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import httpx
import pytest

from director_ai.proxy import create_proxy_app


def _ok_response(content="The sky is blue."):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "model": "test",
        },
    )


def _make_app(**kw):
    defaults = {
        "threshold": 0.1,
        "upstream_url": "https://api.openai.com",
        "allow_http_upstream": True,
        "_transport": httpx.MockTransport(lambda req: _ok_response()),
    }
    defaults.update(kw)
    return create_proxy_app(**defaults)


class TestProxyAuth:
    @pytest.mark.asyncio
    async def test_no_auth_required(self):
        app = _make_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as c:
            resp = await c.post(
                "http://test/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "sky color"}]},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_auth_rejects_bad_key(self):
        app = _make_app(api_keys=["secret-key"])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as c:
            resp = await c.post(
                "http://test/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "test"}]},
                headers={"X-API-Key": "wrong-key"},
            )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_auth_accepts_valid_key(self):
        app = _make_app(api_keys=["secret-key"])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as c:
            resp = await c.post(
                "http://test/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "sky color"}]},
                headers={"X-API-Key": "secret-key"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_exempt_from_auth(self):
        app = _make_app(api_keys=["secret-key"])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as c:
            resp = await c.get("http://test/health")
        assert resp.status_code == 200


class TestProxyWarnMode:
    @pytest.mark.asyncio
    async def test_warn_forwards_with_headers(self):
        app = _make_app(threshold=0.99, on_fail="warn")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as c:
            resp = await c.post(
                "http://test/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "sky color"}]},
            )
        assert resp.status_code == 200
        assert "x-director-score" in resp.headers
        assert resp.headers.get("x-director-approved") == "false"

    @pytest.mark.asyncio
    async def test_reject_returns_422(self):
        app = _make_app(threshold=0.99, on_fail="reject")
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app)) as c:
            resp = await c.post(
                "http://test/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "sky color"}]},
            )
        assert resp.status_code == 422
        assert "hallucination" in resp.json()["error"]["message"].lower()


class TestProxyValidation:
    def test_http_upstream_rejected(self):
        with pytest.raises(ValueError, match="Non-HTTPS"):
            create_proxy_app(upstream_url="http://localhost:8000")

    def test_http_upstream_allowed(self):
        app = create_proxy_app(
            upstream_url="http://localhost:8000",
            allow_http_upstream=True,
            _transport=httpx.MockTransport(lambda req: _ok_response()),
        )
        assert app is not None

    def test_invalid_on_fail(self):
        with pytest.raises(ValueError, match="on_fail"):
            create_proxy_app(on_fail="explode")
