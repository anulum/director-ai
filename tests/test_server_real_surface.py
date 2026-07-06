# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - server real-surface tests
"""Real server-surface coverage for the core HTTP and WebSocket routes."""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("fastapi", reason="fastapi required for server tests")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.core.metrics import metrics
from director_ai.server import create_app


def _core_server_app() -> FastAPI:
    """Return the production server app in its fast local profile."""
    config = DirectorConfig.from_profile("fast")
    config.mode = "general"
    config.hybrid_retrieval = False
    config.reranker_enabled = False
    return create_app(config)


def test_core_server_routes_round_trip_over_real_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Core HTTP and WebSocket routes should work through the mounted app."""
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    metrics.reset()
    app = _core_server_app()

    with TestClient(app) as client:
        health_response = client.get("/v1/health")
        ready_response = client.get("/v1/ready")
        review_response = client.post(
            "/v1/review",
            json={"prompt": "What is 2+2?", "response": "2+2 is 4."},
        )
        process_response = client.post(
            "/v1/process",
            json={"prompt": "Write one sentence about water."},
        )
        batch_response = client.post(
            "/v1/batch",
            json={
                "task": "review",
                "prompts": ["What is water?", "What is air?"],
                "responses": ["Water is H2O.", "Air is a mixture of gases."],
            },
        )
        scorer_models_response = client.get("/v1/scorer/models")
        stats_response = client.get("/v1/stats")
        hourly_response = client.get("/v1/stats/hourly?days=1")
        dashboard_response = client.get("/v1/dashboard")
        metrics_response = client.get("/v1/metrics")
        prometheus_response = client.get("/v1/metrics/prometheus")

        with client.websocket_connect("/v1/stream") as websocket:
            websocket.send_json({"prompt": "What is the capital of Slovakia?"})
            websocket_payload = cast(dict[str, object], websocket.receive_json())

    assert health_response.status_code == 200
    health_payload = cast(dict[str, object], health_response.json())
    assert health_payload["status"] == "ok"
    assert health_payload["profile"] == "fast"
    assert cast(dict[str, object], health_payload["routers"])["knowledge"] == "mounted"

    assert ready_response.status_code == 200
    ready_payload = cast(dict[str, object], ready_response.json())
    assert ready_payload["ready"] is True
    assert ready_payload["reason"] == ""

    assert review_response.status_code == 200, review_response.text
    review_payload = cast(dict[str, object], review_response.json())
    assert "approved" in review_payload
    assert "coherence" in review_payload
    assert "h_logical" in review_payload
    assert "h_factual" in review_payload

    assert process_response.status_code == 200, process_response.text
    process_payload = cast(dict[str, object], process_response.json())
    assert "output" in process_payload
    assert "halted" in process_payload
    assert "candidates_evaluated" in process_payload

    assert batch_response.status_code == 200, batch_response.text
    batch_payload = cast(dict[str, object], batch_response.json())
    assert batch_payload["total"] == 2
    assert (
        cast(int, batch_payload["succeeded"]) + cast(int, batch_payload["failed"]) == 2
    )

    assert scorer_models_response.status_code == 200
    scorer_models_payload = cast(dict[str, object], scorer_models_response.json())
    assert "current" in scorer_models_payload
    assert "models" in scorer_models_payload

    assert stats_response.status_code == 200
    stats_payload = cast(dict[str, object], stats_response.json())
    assert isinstance(stats_payload["total"], int)

    assert hourly_response.status_code == 200
    assert "data" in cast(dict[str, object], hourly_response.json())

    assert dashboard_response.status_code == 200
    assert "Director-AI Dashboard" in dashboard_response.text

    assert metrics_response.status_code == 200
    metrics_payload = cast(dict[str, object], metrics_response.json())
    assert "counters" in metrics_payload
    assert prometheus_response.status_code == 200
    assert "director_ai_" in prometheus_response.text

    assert websocket_payload["type"] == "result"
    assert "output" in websocket_payload
    assert "halted" in websocket_payload


def test_server_rate_limiter_is_wired_over_real_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rate limiting should mount the production SlowAPI middleware."""
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    config = DirectorConfig.from_profile("fast")
    config.hybrid_retrieval = False
    config.llm_provider = "mock"
    config.rate_limit_rpm = 60
    config.reranker_enabled = False
    config.use_nli = False
    app = create_app(config)

    middleware_names = {
        cast(type[object], middleware.cls).__name__
        for middleware in app.user_middleware
    }

    with TestClient(app) as client:
        config_response = client.get("/v1/config")

    assert config_response.status_code == 200
    assert "SlowAPIMiddleware" in middleware_names
    assert app.state.limiter is not None
    assert app.state.limiter._default_limits is not None


def test_server_rejects_excessive_cors_origins_over_create_app() -> None:
    """CORS origin limits should fail closed through app construction."""
    origins = ",".join(f"https://console-{idx}.example" for idx in range(101))
    config = DirectorConfig(
        api_keys=[],
        cors_origins=origins,
        hybrid_retrieval=False,
        llm_provider="mock",
        reranker_enabled=False,
        use_nli=False,
    )

    with pytest.raises(ValueError, match="Too many CORS origins"):
        create_app(config)
