# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — metrics real-surface tests
"""Real server-surface coverage for Director-AI metrics export."""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("fastapi", reason="fastapi required for metrics surface tests")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from director_ai.core.config import DirectorConfig
from director_ai.core.metrics import metrics
from director_ai.server import create_app
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _metrics_app() -> FastAPI:
    """Return the production server with lightweight local scoring enabled."""
    config = DirectorConfig.from_profile("fast")
    config.hybrid_retrieval = False
    config.llm_provider = "mock"
    config.metrics_require_auth = False
    config.rate_limit_rpm = 0
    config.reranker_enabled = False
    config.use_nli = False
    return create_app(config)


def _counter_total(payload: dict[str, object], name: str) -> float:
    """Return a counter total from a structured metrics payload."""
    counters = cast(dict[str, object], payload["counters"])
    counter = cast(dict[str, object], counters[name])
    return cast(float, counter["total"])


def _histogram_count(payload: dict[str, object], name: str) -> int:
    """Return a histogram count from a structured metrics payload."""
    histograms = cast(dict[str, object], payload["histograms"])
    histogram = cast(dict[str, object], histograms[name])
    return cast(int, histogram["count"])


def test_metrics_unit_guard_has_real_surface_companion() -> None:
    """The metrics unit guard should be backed by real HTTP metrics coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_metrics.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_metrics_real_surface.py" in category


def test_review_route_exports_real_metrics_over_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real review request should update JSON and Prometheus metric exports."""
    monkeypatch.setenv("DIRECTOR_FORCE_CPU", "1")
    metrics.reset()
    app = _metrics_app()

    with TestClient(app) as client:
        review_response = client.post(
            "/v1/review",
            json={
                "prompt": "Name one capital city.",
                "response": "Bratislava is the capital of Slovakia.",
            },
        )
        metrics_response = client.get("/v1/metrics")
        prometheus_response = client.get("/v1/metrics/prometheus")

    assert review_response.status_code == 200, review_response.text
    assert metrics_response.status_code == 200, metrics_response.text
    assert prometheus_response.status_code == 200, prometheus_response.text

    metrics_payload = cast(dict[str, object], metrics_response.json())
    assert _counter_total(metrics_payload, "reviews_total") >= 1.0
    assert _histogram_count(metrics_payload, "review_duration_seconds") >= 1
    assert _histogram_count(metrics_payload, "coherence_score") >= 1

    prometheus_text = prometheus_response.text
    assert "director_ai_reviews_total" in prometheus_text
    assert "director_ai_review_duration_seconds_count" in prometheus_text
    assert "director_ai_http_request_duration_seconds_count" in prometheus_text
