# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle deep coverage for server pipeline."""

from __future__ import annotations

import asyncio
import threading

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from starlette.websockets import WebSocketDisconnect

from director_ai.core.config import DirectorConfig


@pytest.fixture
def stats_client():
    from starlette.testclient import TestClient

    cfg = DirectorConfig(use_nli=False, stats_backend="sqlite")
    from director_ai.server import create_app

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def audit_client(tmp_path):
    from starlette.testclient import TestClient

    cfg = DirectorConfig(
        use_nli=False,
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    from director_ai.server import create_app

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def nli_client():
    from starlette.testclient import TestClient

    cfg = DirectorConfig(use_nli=True)
    from director_ai.server import create_app

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


class TestRateLimitWithSlowapi:
    def test_rate_limit_configured(self):
        """Exercise rate limit setup when slowapi IS available."""
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False, rate_limit_rpm=120)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.get("/v1/health")
            assert resp.status_code == 200


class TestStatsEndpoints:
    def test_stats_sqlite(self, stats_client):
        # Do a review to populate stats
        stats_client.post(
            "/v1/review",
            json={"prompt": "sky?", "response": "The sky is blue."},
        )
        resp = stats_client.get("/v1/stats")
        assert resp.status_code == 200

    def test_stats_hourly_sqlite(self, stats_client):
        resp = stats_client.get("/v1/stats/hourly")
        assert resp.status_code == 200


class TestAuditEndpoints:
    def test_review_with_audit(self, audit_client):
        resp = audit_client.post(
            "/v1/review",
            json={"prompt": "sky?", "response": "The sky is blue."},
        )
        assert resp.status_code == 200


class TestProcessAudit:
    def test_process_with_audit(self, audit_client):
        resp = audit_client.post("/v1/process", json={"prompt": "What is 2+2?"})
        assert resp.status_code == 200


class TestDeleteSession:
    def test_delete_existing_session(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c:
            c.post(
                "/v1/review",
                json={
                    "prompt": "q",
                    "response": "a",
                    "session_id": "del-me",
                },
            )
            resp = c.delete("/v1/sessions/del-me")
            assert resp.status_code == 200
            assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_session(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.delete("/v1/sessions/nope")
            assert resp.status_code == 404


class TestApiKeyAuth:
    def test_auth_required(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False, api_keys=["test-key-abc"])
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c:
            # No key â†’ 401
            resp = c.post(
                "/v1/review",
                json={"prompt": "q", "response": "a"},
            )
            assert resp.status_code == 401

            # Exempt paths still work
            resp = c.get("/v1/health")
            assert resp.status_code == 200

            # With key â†’ 200
            resp = c.post(
                "/v1/review",
                json={"prompt": "q", "response": "a"},
                headers={"X-API-Key": "test-key-abc"},
            )
            assert resp.status_code == 200


class TestWsAuthAndErrors:
    def test_ws_auth_rejected(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False, api_keys=["ws-key"])
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with (
            TestClient(app) as c,
            pytest.raises(WebSocketDisconnect),
            c.websocket_connect("/v1/stream") as ws,
        ):
            ws.receive_json()

    def test_ws_non_dict(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c, c.websocket_connect("/v1/stream") as ws:
            ws.send_json([1, 2, 3])
            data = ws.receive_json()
            assert "error" in data

    def test_ws_streaming_oversight_returns_frame(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c, c.websocket_connect("/v1/stream") as ws:
            ws.send_json(
                {
                    "session_id": "stream-oversight-regression",
                    "prompt": "What is 2+2?",
                    "streaming_oversight": True,
                }
            )
            data = ws.receive_json()

        assert data["session_id"] == "stream-oversight-regression"
        assert data["type"] in {"result", "halt"}
        assert "output" in data
        assert "coherence" in data
        assert "error" not in data

    def test_ws_cancel_sets_processing_cancel_event(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False)
        from director_ai.server import create_app

        class SlowAgent:
            def __init__(self):
                self.started = threading.Event()
                self.cancel_event = None

            async def aprocess(self, prompt, tenant_id="", cancel_event=None):
                self.cancel_event = cancel_event
                self.started.set()
                while cancel_event is not None and not cancel_event.is_set():
                    await asyncio.sleep(0.01)
                raise RuntimeError("processing cancelled")

        agent = SlowAgent()
        app = create_app(config=cfg)
        with TestClient(app) as c:
            app.state._state["agent"] = agent
            with c.websocket_connect("/v1/stream") as ws:
                ws.send_json({"session_id": "cancel-me", "prompt": "slow work"})
                assert agent.started.wait(1.0)
                ws.send_json({"action": "cancel", "session_id": "cancel-me"})
                data = ws.receive_json()

        assert data == {"session_id": "cancel-me", "type": "cancelled"}
        assert agent.cancel_event is not None
        assert agent.cancel_event.is_set()

    def test_ws_rejects_requests_above_active_limit(self, monkeypatch):
        from starlette.testclient import TestClient

        import director_ai.server as server_mod

        monkeypatch.setattr(server_mod, "_WS_MAX_CONCURRENT", 1)
        cfg = DirectorConfig(use_nli=False)

        class SlowAgent:
            def __init__(self):
                self.started = threading.Event()

            async def aprocess(self, prompt, tenant_id="", cancel_event=None):
                self.started.set()
                while cancel_event is not None and not cancel_event.is_set():
                    await asyncio.sleep(0.01)
                raise RuntimeError("processing cancelled")

        agent = SlowAgent()
        app = server_mod.create_app(config=cfg)
        with TestClient(app) as c:
            app.state._state["agent"] = agent
            with c.websocket_connect("/v1/stream") as ws:
                ws.send_json({"session_id": "first", "prompt": "slow work"})
                assert agent.started.wait(1.0)
                ws.send_json({"session_id": "second", "prompt": "queued work"})
                data = ws.receive_json()
                ws.send_json({"action": "cancel", "session_id": "first"})
                ws.receive_json()

        assert data["session_id"] == "second"
        assert data["error"] == "too many active sessions"


class TestSourceEndpoint:
    def test_source_disabled(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False, source_endpoint_enabled=False)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.get("/v1/source")
            assert resp.status_code == 404

    def test_source_enabled(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False, source_endpoint_enabled=True)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c:
            resp = c.get("/v1/source")
            assert resp.status_code == 200
            assert "AGPL" in resp.json()["license"]
