# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Server Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

try:
    from fastapi.testclient import TestClient

    from director_ai.core.config import DirectorConfig
    from director_ai.server import create_app

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")


@pytest.fixture
def client():
    """TestClient for the Director AI server (with lifespan)."""
    config = DirectorConfig(use_nli=False, metrics_enabled=True)
    app = create_app(config)
    with TestClient(app) as c:
        yield c


class TestHealth:
    """Health endpoint tests."""

    def test_health_ok(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime_seconds" in data

    def test_health_has_profile(self, client):
        resp = client.get("/v1/health")
        data = resp.json()
        assert "profile" in data


class TestReview:
    """Review endpoint tests."""

    def test_review_valid(self, client):
        resp = client.post(
            "/v1/review",
            json={"prompt": "What color is the sky?", "response": "Blue"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "approved" in data
        assert "coherence" in data
        assert "h_logical" in data
        assert "h_factual" in data

    def test_review_missing_fields(self, client):
        resp = client.post("/v1/review", json={"prompt": "Hello"})
        assert resp.status_code == 422  # Validation error


class TestProcess:
    """Process endpoint tests."""

    def test_process_valid(self, client):
        resp = client.post(
            "/v1/process",
            json={"prompt": "What is the meaning of life?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "output" in data
        assert "halted" in data
        assert "candidates_evaluated" in data

    def test_process_missing_prompt(self, client):
        resp = client.post("/v1/process", json={})
        assert resp.status_code == 422


class TestBatch:
    """Batch endpoint tests."""

    def test_batch_valid(self, client):
        resp = client.post(
            "/v1/batch",
            json={"prompts": ["Q1", "Q2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["succeeded"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_prompts(self, client):
        resp = client.post("/v1/batch", json={"prompts": []})
        assert resp.status_code == 422  # min_length=1


class TestMetrics:
    """Metrics endpoint tests."""

    def test_metrics_json(self, client):
        # Trigger a review first
        client.post(
            "/v1/review",
            json={"prompt": "Q", "response": "A"},
        )
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "counters" in data
        assert "histograms" in data
        assert "gauges" in data

    def test_metrics_prometheus(self, client):
        resp = client.get("/v1/metrics/prometheus")
        assert resp.status_code == 200
        assert "director_ai_" in resp.text


class TestConfig:
    """Config endpoint tests."""

    def test_config_endpoint(self, client):
        resp = client.get("/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert "config" in data
        assert "coherence_threshold" in data["config"]


class TestStats:
    """Stats endpoint tests."""

    def test_stats_returns_summary(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "approved" in data
        assert "rejected" in data
        assert isinstance(data["total"], int)

    def test_stats_after_review(self, client):
        client.post(
            "/v1/review",
            json={"prompt": "Q", "response": "A"},
        )
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_stats_hourly(self, client):
        resp = client.get("/v1/stats/hourly")
        assert resp.status_code == 200

    def test_stats_hourly_custom_days(self, client):
        resp = client.get("/v1/stats/hourly?days=1")
        assert resp.status_code == 200


class TestDashboard:
    """Dashboard endpoint tests."""

    def test_dashboard_html(self, client):
        resp = client.get("/v1/dashboard")
        assert resp.status_code == 200
        assert "Director-AI Dashboard" in resp.text
        assert "Total Reviews" in resp.text

    def test_dashboard_after_review(self, client):
        client.post(
            "/v1/review",
            json={"prompt": "Q", "response": "A"},
        )
        resp = client.get("/v1/dashboard")
        assert resp.status_code == 200
        assert "Approval Rate" in resp.text


class TestWebSocket:
    """WebSocket /v1/stream endpoint tests."""

    def test_ws_valid_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "What is 2+2?"})
            data = ws.receive_json()
            assert data["type"] == "result"
            assert "output" in data
            assert "halted" in data

    def test_ws_empty_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": ""})
            data = ws.receive_json()
            assert "error" in data

    def test_ws_missing_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"foo": "bar"})
            data = ws.receive_json()
            assert "error" in data

    def test_ws_multiple_messages(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "Q1"})
            d1 = ws.receive_json()
            ws.send_json({"prompt": "Q2"})
            d2 = ws.receive_json()
            assert d1["type"] == "result"
            assert d2["type"] == "result"

    def test_ws_prompt_exceeds_max_length(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": "x" * 100_001})
            data = ws.receive_json()
            assert "error" in data
            assert "100000" in data["error"]

    def test_ws_non_string_prompt(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json({"prompt": 12345})
            data = ws.receive_json()
            assert "error" in data

    def test_ws_non_dict_payload(self, client):
        with client.websocket_connect("/v1/stream") as ws:
            ws.send_json([1, 2, 3])
            data = ws.receive_json()
            assert "error" in data


class TestWebSocketAgentError:
    """WebSocket error handling when agent.process() raises."""

    def test_ws_agent_error_returns_error_json(self, client):
        from unittest.mock import patch

        with (
            patch(
                "director_ai.core.agent.CoherenceAgent.process",
                side_effect=RuntimeError("GPU OOM"),
            ),
            client.websocket_connect("/v1/stream") as ws,
        ):
            ws.send_json({"prompt": "trigger failure"})
            data = ws.receive_json()
            assert "error" in data
            assert "GPU OOM" in data["error"]

    def test_ws_agent_error_does_not_kill_connection(self, client):
        from unittest.mock import patch

        call_count = 0

        def _fail_once(prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("transient error")
            from director_ai.core.agent import CoherenceAgent

            return CoherenceAgent.process(client.app, prompt)

        with client.websocket_connect("/v1/stream") as ws:
            with patch(
                "director_ai.core.agent.CoherenceAgent.process",
                side_effect=ValueError("transient"),
            ):
                ws.send_json({"prompt": "fail"})
                d1 = ws.receive_json()
                assert "error" in d1

            # Connection survives — next message works
            ws.send_json({"prompt": "ok"})
            d2 = ws.receive_json()
            assert d2["type"] == "result"
