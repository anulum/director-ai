# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real HTTP and SQLite coverage for feedback calibration storage."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport

from director_ai.core.calibration.feedback_store import FeedbackStore
from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


@pytest.mark.asyncio
async def test_feedback_route_persists_tenant_calibration_rows(
    tmp_path: Path,
) -> None:
    """Feedback HTTP writes should persist tenant-safe calibration rows."""
    db_path = tmp_path / "feedback.db"

    async with _feedback_client(db_path) as client:
        accepted = await _post_feedback(
            client,
            prompt="What is the capital of France?",
            response="Berlin.",
            guardrail_approved=True,
            human_approved=False,
            guardrail_score=0.82,
            domain="qa",
            review_id="rev-1",
            tenant_id="tenant-a",
        )
        corrected = await _post_feedback(
            client,
            prompt="What is the boiling point of water?",
            response="100 degrees Celsius.",
            guardrail_approved=True,
            human_approved=True,
            guardrail_score=0.93,
            domain="qa",
            review_id="rev-2",
            tenant_id="tenant-a",
        )

    assert accepted == {
        "accepted": True,
        "correction_count": 1,
        "disagreement": True,
        "tenant_id": "tenant-a",
        "review_id": "rev-1",
    }
    assert corrected["correction_count"] == 2
    assert corrected["disagreement"] is False

    store = FeedbackStore(db_path)
    try:
        rows = store.export_calibration_rows(domain="qa", include_text=False)
    finally:
        store.close()

    assert [row["review_id"] for row in rows] == ["rev-2", "rev-1"]
    assert {row["tenant_id"] for row in rows} == {"tenant-a"}
    assert [row["disagreement"] for row in rows] == [False, True]
    assert all(
        row["schema_version"] == "director-ai.calibration-feedback.v1" for row in rows
    )
    assert all(row["prompt"] == "" and row["response"] == "" for row in rows)


@pytest.mark.asyncio
async def test_feedback_calibration_endpoint_uses_persisted_http_feedback(
    tmp_path: Path,
) -> None:
    """Calibration metrics should be derived from feedback route records."""
    db_path = tmp_path / "feedback.db"

    async with _feedback_client(db_path) as client:
        for index in range(20):
            await _post_feedback(
                client,
                prompt=f"question {index}",
                response=f"answer {index}",
                guardrail_approved=index % 2 == 0,
                human_approved=index % 3 == 0,
                guardrail_score=0.85 if index % 2 == 0 else 0.15,
                domain="qa",
                review_id=f"rev-{index}",
                tenant_id="tenant-cal",
            )
        response = await client.get(
            "/v1/feedback/calibration?domain=qa&min_corrections=10"
        )

    assert response.status_code == 200, response.text
    payload = cast(dict[str, Any], response.json())
    assert payload["correction_count"] == 20
    assert payload["optimal_threshold"] is not None
    for key in ("current_accuracy", "tpr", "tnr", "fpr", "fnr", "fpr_ci", "fnr_ci"):
        assert 0.0 <= payload[key] <= 1.0


@pytest.mark.asyncio
async def test_feedback_routes_fail_closed_without_configured_store() -> None:
    """Feedback routes should reject writes and calibration without a store."""
    cfg = DirectorConfig(mode="general", use_nli=False, sanitize_inputs=False)
    app = create_app(cfg)

    async with _asgi_client(app) as client:
        feedback = await client.post(
            "/v1/feedback",
            json={
                "prompt": "question",
                "response": "answer",
                "guardrail_approved": True,
                "human_approved": False,
                "guardrail_score": 0.7,
            },
        )
        invalid_minimum = await client.get("/v1/feedback/calibration?min_corrections=0")
        calibration = await client.get("/v1/feedback/calibration")

    assert feedback.status_code == 503
    assert feedback.json() == {"detail": "Feedback store not configured"}
    assert invalid_minimum.status_code == 400
    assert invalid_minimum.json() == {
        "detail": "min_corrections must be between 1 and 100000"
    }
    assert calibration.status_code == 503
    assert calibration.json() == {"detail": "Feedback store not configured"}


def _feedback_app(db_path: Path) -> FastAPI:
    cfg = DirectorConfig(
        mode="general",
        use_nli=False,
        feedback_db_path=str(db_path),
        sanitize_inputs=False,
    )
    return create_app(cfg)


@asynccontextmanager
async def _feedback_client(db_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    async with _asgi_client(_feedback_app(db_path)) as client:
        yield client


@asynccontextmanager
async def _asgi_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://director.local",
        ) as client:
            yield client


async def _post_feedback(
    client: httpx.AsyncClient,
    *,
    prompt: str,
    response: str,
    guardrail_approved: bool,
    human_approved: bool,
    guardrail_score: float,
    domain: str,
    review_id: str,
    tenant_id: str,
) -> dict[str, Any]:
    http_response = await client.post(
        "/v1/feedback",
        json={
            "prompt": prompt,
            "response": response,
            "guardrail_approved": guardrail_approved,
            "human_approved": human_approved,
            "guardrail_score": guardrail_score,
            "domain": domain,
            "review_id": review_id,
        },
        headers={"X-Tenant-ID": tenant_id},
    )
    assert http_response.status_code == 200, http_response.text
    payload = http_response.json()
    assert isinstance(payload, dict)
    return payload
