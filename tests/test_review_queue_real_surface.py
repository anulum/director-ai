# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real public-surface coverage for the review queue."""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("fastapi", reason="fastapi required for server route tests")

from fastapi.testclient import TestClient

import director_ai
from director_ai.core import ReviewQueue
from director_ai.core.config import DirectorConfig
from director_ai.core.metrics import metrics
from director_ai.server import create_app
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _review_queue_client() -> TestClient:
    """Build a real FastAPI app with continuous review batching enabled."""
    config = DirectorConfig(
        mode="general",
        scorer_backend="lite",
        use_nli=True,
        coherence_threshold=0.0,
        hard_limit=0.0,
        soft_limit=0.0,
        adaptive_threshold_enabled=False,
        review_queue_enabled=True,
        review_queue_max_batch=1,
        review_queue_flush_timeout_ms=5000.0,
        hybrid_retrieval=False,
        reranker_enabled=False,
        retrieval_abstention_threshold=0.0,
    )
    return TestClient(create_app(config))


def test_review_queue_unit_guard_declares_this_companion() -> None:
    """The legacy review queue unit guard should declare this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_review_queue.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_review_queue_real_surface.py" in reason


def test_review_queue_public_exports_resolve_to_runtime_type() -> None:
    """Package and core exports should expose the production queue class."""
    assert director_ai.ReviewQueue is ReviewQueue


def test_public_review_route_flushes_real_review_queue_metrics() -> None:
    """The review route should flush the production queue via public HTTP."""
    metrics.reset()

    with _review_queue_client() as client:
        review_response = client.post(
            "/v1/review",
            headers={"X-Tenant-ID": "tenant-alpha"},
            json={
                "prompt": "Which deployment guard is required?",
                "response": "Every deployment requires a signed safety review.",
            },
        )
        metrics_response = client.get("/v1/metrics")
        prometheus_response = client.get("/v1/metrics/prometheus")

    assert review_response.status_code == 200, review_response.text
    payload = cast(dict[str, object], review_response.json())
    assert isinstance(payload["approved"], bool)
    assert isinstance(payload["coherence"], float)

    assert metrics_response.status_code == 200, metrics_response.text
    metrics_payload = cast(
        dict[str, dict[str, dict[str, float]]], metrics_response.json()
    )
    queue_histogram = metrics_payload["histograms"]["review_queue_batch_size"]
    assert queue_histogram["count"] == 1
    assert queue_histogram["total"] == pytest.approx(1.0)
    assert queue_histogram["mean"] == pytest.approx(1.0)

    assert prometheus_response.status_code == 200, prometheus_response.text
    prometheus_text = prometheus_response.text
    assert "director_ai_review_queue_batch_size_count 1" in prometheus_text
    assert "director_ai_review_queue_batch_size_sum 1.0" in prometheus_text
