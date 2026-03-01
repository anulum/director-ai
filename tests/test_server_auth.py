# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Server Auth, Rate Limit, Correlation ID Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
from __future__ import annotations

import pytest

from director_ai.core.config import DirectorConfig

try:
    from fastapi.testclient import TestClient

    from director_ai.server import create_app

    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _SERVER_AVAILABLE, reason="fastapi not installed")


def _auth_app():
    cfg = DirectorConfig(api_keys=["test-key-123"], llm_provider="mock")
    return create_app(cfg)


def _noauth_app():
    cfg = DirectorConfig(api_keys=[], llm_provider="mock")
    return create_app(cfg)


def test_auth_rejects_missing_key():
    with TestClient(_auth_app()) as client:
        r = client.get("/v1/config")
    assert r.status_code == 401


def test_auth_rejects_wrong_key():
    with TestClient(_auth_app()) as client:
        r = client.get("/v1/config", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_auth_accepts_correct_key():
    with TestClient(_auth_app()) as client:
        r = client.get("/v1/config", headers={"X-API-Key": "test-key-123"})
    assert r.status_code == 200


def test_health_exempt_from_auth():
    with TestClient(_auth_app()) as client:
        r = client.get("/v1/health")
    assert r.status_code == 200


def test_prometheus_exempt_from_auth():
    with TestClient(_auth_app()) as client:
        r = client.get("/v1/metrics/prometheus")
    assert r.status_code == 200


def test_no_auth_when_keys_empty():
    with TestClient(_noauth_app()) as client:
        r = client.get("/v1/config")
    assert r.status_code == 200


def test_correlation_id_generated():
    with TestClient(_noauth_app()) as client:
        r = client.get("/v1/health")
    assert "X-Request-ID" in r.headers
    assert len(r.headers["X-Request-ID"]) > 0


def test_correlation_id_echoed():
    with TestClient(_noauth_app()) as client:
        r = client.get(
            "/v1/health",
            headers={"X-Request-ID": "my-trace-42"},
        )
    assert r.headers["X-Request-ID"] == "my-trace-42"


def test_api_keys_redacted_in_config():
    with TestClient(_auth_app()) as client:
        r = client.get(
            "/v1/config",
            headers={"X-API-Key": "test-key-123"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["config"]["api_keys"] == "***"
