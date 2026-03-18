# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Server Auth, Rate Limit, Correlation ID Tests

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


def _auth_app(metrics_require_auth=True):
    cfg = DirectorConfig(
        api_keys=["test-key-123"],
        llm_provider="mock",
        metrics_require_auth=metrics_require_auth,
    )
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


def test_prometheus_exempt_when_metrics_auth_disabled():
    with TestClient(_auth_app(metrics_require_auth=False)) as client:
        r = client.get("/v1/metrics/prometheus")
    assert r.status_code == 200


def test_prometheus_requires_auth_by_default():
    with TestClient(_auth_app()) as client:
        r = client.get("/v1/metrics/prometheus")
    assert r.status_code == 401


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


def test_rate_limiter_has_default_limits():
    """Limiter is constructed with default_limits when rate_limit_rpm > 0."""
    try:
        from slowapi import Limiter  # noqa: F401
    except ImportError:
        pytest.skip("slowapi not installed")

    cfg = DirectorConfig(rate_limit_rpm=60, llm_provider="mock")
    app = create_app(cfg)
    limiter = app.state.limiter
    assert limiter is not None
    assert limiter._default_limits is not None
    assert len(limiter._default_limits) > 0


def test_prompt_too_long_returns_422():
    with TestClient(_noauth_app()) as client:
        r = client.post(
            "/v1/review",
            json={"prompt": "x" * 100_001, "response": "ok"},
        )
    assert r.status_code == 422


def test_response_too_long_returns_422():
    with TestClient(_noauth_app()) as client:
        r = client.post(
            "/v1/review",
            json={"prompt": "ok", "response": "x" * 500_001},
        )
    assert r.status_code == 422


def test_valid_body_sizes_accepted():
    with TestClient(_noauth_app()) as client:
        r = client.post(
            "/v1/review",
            json={"prompt": "What is 2+2?", "response": "4"},
        )
    assert r.status_code == 200


# â”€â”€ Stats backend tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_prometheus_stats_returns_summary():
    """Default prometheus backend returns summary dict from /v1/stats."""
    cfg = DirectorConfig(api_keys=[], llm_provider="mock", stats_backend="prometheus")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total" in data
    assert "approved" in data


def test_prometheus_hourly_graceful():
    """Prometheus backend returns empty hourly breakdown with note."""
    cfg = DirectorConfig(api_keys=[], llm_provider="mock", stats_backend="prometheus")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/stats/hourly")
    assert r.status_code == 200
    data = r.json()
    assert data["data"] == []
    assert "sqlite" in data["note"]


def test_dashboard_renders_with_prometheus():
    cfg = DirectorConfig(api_keys=[], llm_provider="mock", stats_backend="prometheus")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/dashboard")
    assert r.status_code == 200
    assert "Dashboard" in r.text


# â”€â”€ AGPL Â§13 source endpoint tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_source_returns_200():
    with TestClient(_noauth_app()) as client:
        r = client.get("/v1/source")
    assert r.status_code == 200
    data = r.json()
    assert data["license"] == "AGPL-3.0-or-later"
    assert "repository_url" in data
    assert data["agpl_section"] == "13"


def test_source_exempt_from_auth():
    with TestClient(_auth_app()) as client:
        r = client.get("/v1/source")
    assert r.status_code == 200


def test_source_disabled_returns_404():
    cfg = DirectorConfig(
        api_keys=[],
        llm_provider="mock",
        source_endpoint_enabled=False,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/source")
    assert r.status_code == 404


def test_source_custom_repo_url():
    cfg = DirectorConfig(
        api_keys=[],
        llm_provider="mock",
        source_repository_url="https://git.example.com/fork",
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/source")
    assert r.status_code == 200
    assert r.json()["repository_url"] == "https://git.example.com/fork"


def test_metrics_auth_required_returns_401():
    """When metrics_require_auth=True, /v1/metrics/prometheus needs API key."""
    cfg = DirectorConfig(
        api_keys=["secret"],
        llm_provider="mock",
        metrics_require_auth=True,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/metrics/prometheus")
    assert r.status_code == 401


def test_metrics_auth_required_with_key_returns_200():
    cfg = DirectorConfig(
        api_keys=["secret"],
        llm_provider="mock",
        metrics_require_auth=True,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/metrics/prometheus", headers={"X-API-Key": "secret"})
    assert r.status_code == 200


def test_rate_limit_strict_raises_without_slowapi(monkeypatch):
    """rate_limit_strict=True + missing slowapi raises ImportError."""
    import director_ai.server as srv

    monkeypatch.setattr(srv, "_SLOWAPI_AVAILABLE", False)
    cfg = DirectorConfig(rate_limit_rpm=60, rate_limit_strict=True, llm_provider="mock")
    with pytest.raises(ImportError, match="rate_limit_strict"):
        create_app(cfg)


def test_server_provider_wiring_openai(monkeypatch):
    """Non-local provider config sets env key for CoherenceAgent."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-openai")
    cfg = DirectorConfig(llm_provider="openai", llm_api_key="from-config")
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/health")
    assert r.status_code == 200


def test_sqlite_stats_works(tmp_path):
    db_path = str(tmp_path / "test_stats.db")
    cfg = DirectorConfig(
        api_keys=[],
        llm_provider="mock",
        stats_backend="sqlite",
        stats_db_path=db_path,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/v1/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 0
