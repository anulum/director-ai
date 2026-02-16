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
