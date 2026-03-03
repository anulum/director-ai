# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — WebSocket Multiplexed Streaming Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.config import DirectorConfig


@pytest.fixture
def ws_app():
    """Create a test app with mocked agent."""
    from director_ai.server import create_app

    cfg = DirectorConfig(
        use_nli=False,
        llm_provider="mock",
        tenant_routing=True,
    )
    app = create_app(config=cfg)
    return app


@pytest.fixture
def client(ws_app):
    """HTTPX test client with ASGI transport."""
    try:
        from starlette.testclient import TestClient

        with TestClient(ws_app) as c:
            yield c
    except ImportError:
        pytest.skip("starlette testclient not available")


class TestWSMuxProtocol:
    def test_session_id_echoed_in_response(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "Hello world", "session_id": "sid-001"})
            resp = ws.receive_json()
            assert resp.get("session_id") == "sid-001"

    def test_auto_session_id_when_absent(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "Hello"})
            resp = ws.receive_json()
            assert "session_id" in resp

    def test_cancel_returns_cancelled(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"action": "cancel", "session_id": "nonexistent"})
            resp = ws.receive_json()
            assert resp.get("type") == "cancelled"
            assert resp.get("session_id") == "nonexistent"

    def test_invalid_json_error(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_text("not json at all {{{")
            resp = ws.receive_json()
            assert "error" in resp

    def test_empty_prompt_error(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": ""})
            resp = ws.receive_json()
            assert "error" in resp
            assert "non-empty" in resp["error"]

    def test_backward_compat_no_session_id(self, client):
        """Messages without session_id still produce a valid result."""
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "test backward compat"})
            resp = ws.receive_json()
            has_result = resp.get("type") == "result"
            has_sid = "session_id" in resp
            assert has_result or has_sid


class TestTenantVectorFactEndpoint:
    def test_add_vector_fact(self, client):
        resp = client.post(
            "/v1/tenants/acme/vector-facts",
            json={"key": "hq", "value": "Acme HQ is in NYC"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "acme"
        assert data["count"] >= 1

    def test_vector_fact_without_tenant_routing(self):
        """Endpoint returns 404 when tenant routing is disabled."""
        from director_ai.server import create_app

        cfg = DirectorConfig(use_nli=False, tenant_routing=False)
        app = create_app(config=cfg)
        try:
            from starlette.testclient import TestClient

            with TestClient(app) as client:
                resp = client.post(
                    "/v1/tenants/acme/vector-facts",
                    json={"key": "hq", "value": "Acme HQ"},
                )
                assert resp.status_code == 404
        except ImportError:
            pytest.skip("starlette testclient not available")
