# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Coverage tests for server.py â€” FastAPI endpoints."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


@pytest.fixture
def client():
    from starlette.testclient import TestClient

    cfg = DirectorConfig(use_nli=False)
    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_client():
    from starlette.testclient import TestClient

    cfg = DirectorConfig(use_nli=False, api_keys=["test-key-123"])
    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_source(self, client):
        resp = client.get("/v1/source")
        assert resp.status_code == 200
        assert resp.json()["license"] == "AGPL-3.0-or-later"


class TestReview:
    def test_review_approved(self, client):
        resp = client.post(
            "/v1/review",
            json={
                "prompt": "What color is the sky?",
                "response": "The sky is blue.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "approved" in data
        assert "coherence" in data

    def test_review_with_session(self, client):
        resp = client.post(
            "/v1/review",
            json={
                "prompt": "What color is the sky?",
                "response": "The sky is blue.",
                "session_id": "test-session-1",
            },
        )
        assert resp.status_code == 200


class TestProcess:
    def test_process(self, client):
        resp = client.post("/v1/process", json={"prompt": "What is 2+2?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        assert "halted" in data


class TestBatch:
    def test_batch(self, client):
        resp = client.post(
            "/v1/batch",
            json={"prompts": ["What is 2+2?", "What color is the sky?"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2


class TestTenants:
    def test_tenants_disabled(self, client):
        resp = client.get("/v1/tenants")
        assert resp.status_code == 404


class TestSessions:
    def test_session_not_found(self, client):
        resp = client.get("/v1/sessions/nonexistent")
        assert resp.status_code == 404

    def test_session_delete_not_found(self, client):
        resp = client.delete("/v1/sessions/nonexistent")
        assert resp.status_code == 404

    def test_session_create_and_get(self, client):
        client.post(
            "/v1/review",
            json={
                "prompt": "Test",
                "response": "Answer",
                "session_id": "s1",
            },
        )
        resp = client.get("/v1/sessions/s1")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "s1"

    def test_session_delete(self, client):
        client.post(
            "/v1/review",
            json={
                "prompt": "Test",
                "response": "Answer",
                "session_id": "del-me",
            },
        )
        resp = client.delete("/v1/sessions/del-me")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestMetrics:
    def test_metrics(self, client):
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200

    def test_prometheus(self, client):
        resp = client.get("/v1/metrics/prometheus")
        assert resp.status_code == 200


class TestConfig:
    def test_config(self, client):
        resp = client.get("/v1/config")
        assert resp.status_code == 200
        assert "config" in resp.json()


class TestStats:
    def test_stats(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200

    def test_stats_hourly(self, client):
        resp = client.get("/v1/stats/hourly")
        assert resp.status_code == 200


class TestDashboard:
    def test_dashboard(self, client):
        resp = client.get("/v1/dashboard")
        assert resp.status_code == 200
        assert "Director-AI Dashboard" in resp.text


class TestAuth:
    def test_no_key_rejected(self, auth_client):
        resp = auth_client.post(
            "/v1/review",
            json={
                "prompt": "Test",
                "response": "Answer",
            },
        )
        assert resp.status_code == 401

    def test_valid_key(self, auth_client):
        resp = auth_client.post(
            "/v1/review",
            json={"prompt": "Test", "response": "Answer"},
            headers={"X-API-Key": "test-key-123"},
        )
        assert resp.status_code == 200

    def test_health_exempt(self, auth_client):
        resp = auth_client.get("/v1/health")
        assert resp.status_code == 200


class TestWebSocket:
    def test_ws_standard_mode(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "What is 2+2?"})
            data = ws.receive_json()
            assert "type" in data or "output" in data

    def test_ws_empty_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": ""})
            data = ws.receive_json()
            assert "error" in data

    def test_ws_non_dict(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json("just a string")
            data = ws.receive_json()
            assert "error" in data
