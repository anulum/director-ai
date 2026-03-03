"""Deep coverage for server.py — rate limiting, stats, audit, WS error, halt paths."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

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
        resp = audit_client.post(
            "/v1/process", json={"prompt": "What is 2+2?"}
        )
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
            # No key → 401
            resp = c.post(
                "/v1/review",
                json={"prompt": "q", "response": "a"},
            )
            assert resp.status_code == 401

            # Exempt paths still work
            resp = c.get("/v1/health")
            assert resp.status_code == 200

            # With key → 200
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
        with TestClient(app) as c:
            with pytest.raises(Exception):
                with c.websocket_connect("/v1/stream") as ws:
                    ws.receive_json()

    def test_ws_non_dict(self):
        from starlette.testclient import TestClient

        cfg = DirectorConfig(use_nli=False)
        from director_ai.server import create_app

        app = create_app(config=cfg)
        with TestClient(app) as c:
            with c.websocket_connect("/v1/stream") as ws:
                ws.send_json([1, 2, 3])
                data = ws.receive_json()
                assert "error" in data


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
