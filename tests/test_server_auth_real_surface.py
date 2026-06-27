# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - server auth real-surface tests
"""Real server-surface coverage for HTTP and WebSocket authentication."""

from __future__ import annotations

import json
from typing import cast

import pytest

pytest.importorskip("fastapi", reason="fastapi required for server auth tests")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.core.metrics import metrics
from director_ai.server import create_app

_API_KEY = "server-auth-real-key"
_TENANT_ID = "tenant-alpha"


def _tenant_bound_server_app() -> FastAPI:
    """Return the production server app with tenant-bound auth enabled."""
    config = DirectorConfig(
        api_key_tenant_map=json.dumps({_API_KEY: _TENANT_ID}),
        llm_provider="mock",
        metrics_require_auth=True,
        profile="fast",
        rate_limit_rpm=0,
        use_nli=False,
    )
    return create_app(config)


def _auth_headers(*, request_id: str = "trace-real-auth-1") -> dict[str, str]:
    """Return headers for the tenant-bound API key."""
    return {
        "X-API-Key": _API_KEY,
        "X-Tenant-ID": _TENANT_ID,
        "X-Request-ID": request_id,
    }


def test_server_auth_http_and_ticket_contract_over_real_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full server should enforce auth while preserving public probes."""
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    metrics.reset()
    app = _tenant_bound_server_app()

    with TestClient(app) as client:
        health_response = client.get("/v1/health")
        missing_key_response = client.get("/v1/config")
        wrong_tenant_response = client.get(
            "/v1/config",
            headers={
                "X-API-Key": _API_KEY,
                "X-Tenant-ID": "tenant-beta",
                "X-Request-ID": "trace-wrong-tenant",
            },
        )
        authorized_response = client.get(
            "/v1/config",
            headers=_auth_headers(request_id="trace-authorized"),
        )
        prometheus_without_key = client.get("/v1/metrics/prometheus")
        prometheus_with_key = client.get(
            "/v1/metrics/prometheus",
            headers=_auth_headers(request_id="trace-metrics"),
        )
        telemetry_response = client.get(
            "/v1/metrics",
            headers=_auth_headers(request_id="trace-telemetry"),
        )
        ticket_response = client.post(
            "/v1/stream/ticket",
            headers=_auth_headers(request_id="trace-ticket"),
        )

        assert ticket_response.status_code == 200, ticket_response.text
        ticket_payload = cast(dict[str, object], ticket_response.json())
        ticket = cast(str, ticket_payload["ticket"])
        assert ticket

        with client.websocket_connect(f"/v1/stream?ticket={ticket}") as websocket:
            websocket.send_json({"prompt": "What is 2+2?"})
            websocket_payload = cast(dict[str, object], websocket.receive_json())

    assert health_response.status_code == 200
    health_payload = cast(dict[str, object], health_response.json())
    assert health_payload["status"] == "ok"
    assert "license" in health_payload
    assert "version" not in health_payload
    assert "model_revisions" not in health_payload

    assert missing_key_response.status_code == 401
    assert missing_key_response.json() == {"detail": "Invalid or missing API key"}
    assert missing_key_response.headers["X-Request-ID"]

    assert wrong_tenant_response.status_code == 403
    assert wrong_tenant_response.json() == {
        "detail": "API key not authorized for this tenant"
    }
    assert wrong_tenant_response.headers["X-Request-ID"] == "trace-wrong-tenant"

    assert authorized_response.status_code == 200, authorized_response.text
    assert authorized_response.headers["X-Request-ID"] == "trace-authorized"
    authorized_payload = cast(dict[str, object], authorized_response.json())
    config_payload = cast(dict[str, object], authorized_payload["config"])
    assert config_payload["api_keys"] == []
    assert config_payload["api_key_tenant_map"] == "***"

    assert prometheus_without_key.status_code == 401
    assert prometheus_with_key.status_code == 200
    assert "director_ai_" in prometheus_with_key.text
    assert prometheus_with_key.headers["X-Request-ID"] == "trace-metrics"
    assert telemetry_response.status_code == 200
    telemetry_payload = cast(dict[str, object], telemetry_response.json())
    telemetry_counters = cast(dict[str, object], telemetry_payload["counters"])
    http_total = cast(dict[str, object], telemetry_counters["http_requests_total"])
    route_labels = cast(dict[str, float], http_total["multi_labels"])
    assert 'endpoint="/v1/config",method="GET",status="401"' in route_labels
    assert 'endpoint="/v1/config",method="GET",status="403"' in route_labels

    expires_in = cast(float, ticket_payload["expires_in"])
    assert expires_in > 0
    assert websocket_payload["type"] == "result"
    assert "output" in websocket_payload
    assert "halted" in websocket_payload
