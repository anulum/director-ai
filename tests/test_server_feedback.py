# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Server tests for production feedback and online calibration endpoints."""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    from director_ai.core.calibration.feedback_store import FeedbackStore
    from director_ai.core.config import DirectorConfig
    from director_ai.server import create_app

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _FASTAPI_AVAILABLE, reason="FastAPI not installed")


def _feedback_client(tmp_path):
    cfg = DirectorConfig(
        mode="general",
        use_nli=False,
        feedback_db_path=str(tmp_path / "feedback.db"),
        sanitize_inputs=False,
    )
    app = create_app(cfg)
    return TestClient(app)


class TestServerFeedback:
    def test_feedback_requires_configured_store(self):
        cfg = DirectorConfig(mode="general", use_nli=False, sanitize_inputs=False)
        app = create_app(cfg)
        with TestClient(app) as client:
            resp = client.post(
                "/v1/feedback",
                json={
                    "prompt": "q",
                    "response": "a",
                    "guardrail_approved": True,
                    "human_approved": False,
                    "guardrail_score": 0.8,
                },
            )
        assert resp.status_code == 503

    def test_records_feedback_and_reports_disagreement(self, tmp_path):
        with _feedback_client(tmp_path) as client:
            resp = client.post(
                "/v1/feedback",
                json={
                    "prompt": "What is the capital of France?",
                    "response": "Berlin.",
                    "guardrail_approved": True,
                    "human_approved": False,
                    "guardrail_score": 0.82,
                    "domain": "qa",
                    "review_id": "rev-123",
                },
                headers={"X-Tenant-ID": "tenant-a"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["disagreement"] is True
        assert data["correction_count"] == 1
        assert data["tenant_id"] == "tenant-a"
        assert data["review_id"] == "rev-123"

        store = FeedbackStore(tmp_path / "feedback.db")
        corrections = store.get_corrections()
        assert len(corrections) == 1
        assert corrections[0].review_id == "rev-123"
        assert corrections[0].tenant_id == "tenant-a"
        store.close()

    def test_feedback_validation_rejects_bad_score(self, tmp_path):
        with _feedback_client(tmp_path) as client:
            resp = client.post(
                "/v1/feedback",
                json={
                    "prompt": "q",
                    "response": "a",
                    "guardrail_approved": True,
                    "human_approved": True,
                    "guardrail_score": 1.5,
                },
            )
        assert resp.status_code == 422

    def test_calibration_endpoint_returns_metrics(self, tmp_path):
        with _feedback_client(tmp_path) as client:
            for i in range(20):
                client.post(
                    "/v1/feedback",
                    json={
                        "prompt": f"q{i}",
                        "response": f"a{i}",
                        "guardrail_approved": i % 2 == 0,
                        "human_approved": i % 3 == 0,
                        "guardrail_score": 0.8 if i % 2 == 0 else 0.2,
                        "domain": "qa",
                    },
                )
            resp = client.get("/v1/feedback/calibration?domain=qa&min_corrections=10")

        assert resp.status_code == 200
        data = resp.json()
        assert data["correction_count"] == 20
        assert 0.0 <= data["current_accuracy"] <= 1.0
        assert "optimal_threshold" in data

    def test_calibration_rejects_invalid_minimum(self, tmp_path):
        with _feedback_client(tmp_path) as client:
            resp = client.get("/v1/feedback/calibration?min_corrections=0")
        assert resp.status_code == 400
