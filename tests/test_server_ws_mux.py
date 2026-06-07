# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WebSocket Multiplexed Streaming Tests
"""Multi-angle tests for WebSocket streaming pipeline."""

import json

import pytest

pytest.importorskip("fastapi", reason="server extras not installed")

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


@pytest.fixture
def ws_app():
    """Create a test app with mocked agent."""
    cfg = DirectorConfig(
        use_nli=False,
        llm_provider="mock",
        tenant_routing=True,
    )
    return create_app(config=cfg)


@pytest.fixture
def client(ws_app):
    with TestClient(ws_app) as c:
        yield c


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

    def test_tenant_map_rejects_valid_but_unbound_key(self):
        cfg = DirectorConfig(
            api_keys=["bound-key", "orphan-key"],
            api_key_tenant_map=json.dumps({"bound-key": "tenant-a"}),
            use_nli=False,
            llm_provider="mock",
            tenant_routing=True,
        )

        with (
            TestClient(create_app(config=cfg)) as client,
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect(
                "/v1/stream",
                headers={
                    "X-API-Key": "orphan-key",
                    "X-Tenant-ID": "tenant-b",
                },
            ) as ws,
        ):
            ws.send_json({"prompt": "tenant claim should not be accepted"})

        assert exc_info.value.code == 1008


class TestWSAuthHeaders:
    """WebSocket handshake accepts the same auth headers as HTTP."""

    @staticmethod
    def _auth_app():
        cfg = DirectorConfig(
            api_keys=["ws-key-1"],
            use_nli=False,
            llm_provider="mock",
        )
        return create_app(config=cfg)

    def test_ws_accepts_bearer_token(self):
        with (
            TestClient(self._auth_app()) as client,
            client.websocket_connect(
                "/v1/stream",
                headers={"Authorization": "Bearer ws-key-1"},
            ) as ws,
        ):
            ws.send_json({"prompt": "hello", "session_id": "sid-b"})
            resp = ws.receive_json()
            assert resp.get("session_id") == "sid-b"

    def test_ws_accepts_x_api_key(self):
        with (
            TestClient(self._auth_app()) as client,
            client.websocket_connect(
                "/v1/stream",
                headers={"X-API-Key": "ws-key-1"},
            ) as ws,
        ):
            ws.send_json({"prompt": "hello", "session_id": "sid-x"})
            resp = ws.receive_json()
            assert resp.get("session_id") == "sid-x"

    def test_ws_rejects_missing_auth(self):
        with (
            TestClient(self._auth_app()) as client,
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "hello"})
        assert exc.value.code == 1008

    def test_ws_rejects_bad_bearer(self):
        with (
            TestClient(self._auth_app()) as client,
            pytest.raises(WebSocketDisconnect) as exc,
            client.websocket_connect(
                "/v1/stream",
                headers={"Authorization": "Bearer wrong-key"},
            ) as ws,
        ):
            ws.send_json({"prompt": "hello"})
        assert exc.value.code == 1008


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
        cfg = DirectorConfig(use_nli=False, tenant_routing=False)
        app = create_app(config=cfg)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/tenants/acme/vector-facts",
                json={"key": "hq", "value": "Acme HQ"},
            )
            assert resp.status_code == 404
