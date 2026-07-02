# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - public endpoint exposure real-surface tests
"""Real FastAPI coverage for the public endpoint exposure contract."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("fastapi", reason="fastapi required for endpoint tests")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.core.metrics import metrics
from director_ai.server import create_app

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements/public_endpoint_exposure_policy.toml"
_API_KEY = "public-endpoint-real-key"
_TENANT_ID = "public-endpoint-tenant"


def _load_policy() -> dict[str, object]:
    """Return the checked-in public endpoint exposure policy."""
    return tomllib.loads(POLICY_PATH.read_text(encoding="utf-8"))


def _endpoint_policy(path: str) -> dict[str, object]:
    """Return the policy entry for a public endpoint path."""
    endpoints = cast(list[dict[str, object]], _load_policy()["endpoints"])
    for endpoint in endpoints:
        if endpoint["path"] == path:
            return endpoint
    raise AssertionError(f"missing endpoint policy for {path}")


def _endpoint_paths(default_auth: str) -> set[str]:
    """Return policy endpoint paths with the requested default auth class."""
    endpoints = cast(list[dict[str, object]], _load_policy()["endpoints"])
    return {
        cast(str, endpoint["path"])
        for endpoint in endpoints
        if endpoint["default_auth"] == default_auth
    }


def _policy_server_app(
    *,
    metrics_require_auth: bool = True,
    source_endpoint_enabled: bool = True,
) -> FastAPI:
    """Return the production server app with credentials configured."""
    config = DirectorConfig(
        api_key_tenant_map=json.dumps({_API_KEY: _TENANT_ID}),
        llm_provider="mock",
        metrics_require_auth=metrics_require_auth,
        profile="fast",
        rate_limit_rpm=0,
        source_endpoint_enabled=source_endpoint_enabled,
        use_nli=False,
    )
    return create_app(config)


def _auth_headers() -> dict[str, str]:
    """Return tenant-bound request headers accepted by the real middleware."""
    return {
        "X-API-Key": _API_KEY,
        "X-Tenant-ID": _TENANT_ID,
        "X-Request-ID": "trace-public-endpoint-real",
    }


def test_public_endpoint_policy_matches_real_authenticated_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credentialed apps should expose only the documented public probes."""
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    metrics.reset()
    app = _policy_server_app(metrics_require_auth=True)

    assert _endpoint_paths("exempt") == {
        "/v1/live",
        "/v1/health",
        "/v1/ready",
        "/v1/source",
    }
    assert _endpoint_paths("protected_when_credentials_configured") == {
        "/v1/metrics",
        "/v1/metrics/prometheus",
    }
    assert "licence class only" in cast(
        str,
        _endpoint_policy("/v1/health")["allowed_payload"],
    )

    with TestClient(app) as client:
        live_response = client.get("/v1/live")
        health_response = client.get("/v1/health")
        ready_response = client.get("/v1/ready")
        source_response = client.get("/v1/source")
        metrics_response = client.get("/v1/metrics")
        prometheus_response = client.get("/v1/metrics/prometheus")
        metrics_with_key = client.get("/v1/metrics", headers=_auth_headers())
        prometheus_with_key = client.get(
            "/v1/metrics/prometheus",
            headers=_auth_headers(),
        )

    assert live_response.status_code == 200
    assert live_response.json() == {"ok": True}

    assert health_response.status_code == 200
    health_payload = cast(dict[str, object], health_response.json())
    assert health_payload["status"] == "ok"
    assert health_payload["license"] == "open-core"
    assert "version" not in health_payload
    assert "profile" not in health_payload
    assert "model_revisions" not in health_payload

    assert ready_response.status_code in {200, 503}
    assert ready_response.status_code not in {401, 403}

    assert source_response.status_code == 200
    source_payload = cast(dict[str, object], source_response.json())
    assert source_payload["repository_url"] == "https://github.com/anulum/director-ai"
    assert str(source_payload["instructions"]).startswith("git clone ")

    assert metrics_response.status_code == 401
    assert metrics_response.json() == {"detail": "Invalid or missing API key"}
    assert prometheus_response.status_code == 401
    assert prometheus_response.json() == {"detail": "Invalid or missing API key"}

    assert metrics_with_key.status_code == 200
    assert "counters" in cast(dict[str, object], metrics_with_key.json())
    assert prometheus_with_key.status_code == 200
    assert "director_ai_" in prometheus_with_key.text


def test_public_endpoint_operator_controls_are_real_server_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator toggles should change only the documented public surfaces."""
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    metrics.reset()
    app = _policy_server_app(
        metrics_require_auth=False,
        source_endpoint_enabled=False,
    )

    with TestClient(app) as client:
        source_response = client.get("/v1/source")
        prometheus_response = client.get("/v1/metrics/prometheus")
        metrics_response = client.get("/v1/metrics")
        config_response = client.get("/v1/config")

    assert source_response.status_code == 404
    assert source_response.json() == {"detail": "Source endpoint disabled"}

    assert prometheus_response.status_code == 200
    assert "director_ai_" in prometheus_response.text

    assert metrics_response.status_code == 401
    assert config_response.status_code == 401
