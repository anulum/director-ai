# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Fine-tune API real-surface tests
"""Real ASGI coverage for the public fine-tuning API router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import httpx
import pytest

pytest.importorskip("fastapi", reason="fastapi required for fine-tune API tests")

from fastapi import FastAPI
from httpx import ASGITransport

from director_ai.finetune_api import create_finetune_router


def _finetune_app(models_dir: Path) -> FastAPI:
    """Return a FastAPI app with the public fine-tune router mounted."""
    app = FastAPI()
    app.include_router(
        create_finetune_router(models_dir=models_dir),
        prefix="/v1/finetune",
    )
    return app


def _balanced_training_jsonl(sample_pairs_per_label: int = 250) -> bytes:
    """Return valid balanced JSONL bytes for the production validator."""
    rows: list[dict[str, object]] = []
    for index in range(sample_pairs_per_label):
        rows.append(
            {
                "premise": f"Safety review {index} confirms claim alpha.",
                "hypothesis": f"Claim alpha is supported for case {index}.",
                "label": 1,
            }
        )
        rows.append(
            {
                "premise": f"Safety review {index} confirms claim beta.",
                "hypothesis": f"Claim beta contradicts case {index}.",
                "label": 0,
            }
        )
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows).encode("utf-8")


def _managed_training_request() -> dict[str, object]:
    """Return a portable dry-run managed-training payload."""
    return {
        "backend": "portable",
        "dry_run": True,
        "display_name": "director-ai-finetune-api-real-surface",
        "dataset_uri": "s3://director-ai-tests/finetune/train.jsonl",
        "output_uri": "s3://director-ai-tests/finetune/output",
        "container_image_uri": "ghcr.io/anulum/director-ai-trainer:3.17.0",
        "base_model": "factcg-deberta-v3-large",
        "epochs": 1,
        "batch_size": 16,
        "timeout_minutes": 30,
    }


@pytest.mark.asyncio
async def test_validate_endpoint_accepts_real_balanced_jsonl_upload(
    tmp_path: Path,
) -> None:
    """The public validate endpoint should accept a real multipart JSONL upload."""
    app = _finetune_app(tmp_path / "models")
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/finetune/validate",
            files={
                "file": (
                    "train.jsonl",
                    _balanced_training_jsonl(),
                    "application/x-ndjson",
                )
            },
        )

    assert response.status_code == 200, response.text
    payload = cast(dict[str, object], response.json())
    assert payload["is_valid"] is True
    assert payload["total_samples"] == 500
    assert payload["label_distribution"] == {"0": 250, "1": 250}
    assert payload["duplicate_count"] == 0
    assert cast(float, payload["estimated_cost_usd"]) > 0.0
    assert payload["errors"] == []


@pytest.mark.asyncio
async def test_start_endpoint_rejects_invalid_upload_before_training(
    tmp_path: Path,
) -> None:
    """The public start endpoint should reject invalid data before spawning work."""
    app = _finetune_app(tmp_path / "models")
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/finetune/start",
            files={
                "file": (
                    "broken.jsonl",
                    b"not json\nalso not json\n",
                    "application/x-ndjson",
                )
            },
        )

    assert response.status_code == 422, response.text
    payload = cast(dict[str, object], response.json())
    detail = cast(dict[str, object], payload["detail"])
    assert detail["message"] == "Data validation failed"
    assert "No valid samples found" in cast(list[str], detail["errors"])


@pytest.mark.asyncio
async def test_managed_dry_run_lifecycle_uses_real_router_state(
    tmp_path: Path,
) -> None:
    """Managed dry-run submission, listing, and status should round-trip over HTTP."""
    app = _finetune_app(tmp_path / "models")
    transport = ASGITransport(app=app)
    tenant_headers = {"X-Tenant-ID": "tenant.alpha"}

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        submit_response = await client.post(
            "/v1/finetune/managed/submit",
            json=_managed_training_request(),
            headers=tenant_headers,
        )
        submit_payload = cast(dict[str, object], submit_response.json())
        job_id = cast(str, submit_payload["job_id"])

        list_response = await client.get(
            "/v1/finetune/managed/jobs",
            headers=tenant_headers,
        )
        status_response = await client.post(
            "/v1/finetune/managed/status",
            json={"backend": "portable", "job_id": job_id},
            headers=tenant_headers,
        )
        other_tenant_response = await client.post(
            "/v1/finetune/managed/status",
            json={"backend": "portable", "job_id": job_id},
            headers={"X-Tenant-ID": "tenant.beta"},
        )

    assert submit_response.status_code == 200, submit_response.text
    assert submit_payload["backend"] == "portable"
    assert submit_payload["dry_run"] is True
    assert submit_payload["state"] == "dry_run"
    assert submit_payload["tenant_id"] == "tenant.alpha"
    request_payload = cast(dict[str, object], submit_payload["request"])
    assert "director-ai-finetune-api-real-surface" in json.dumps(request_payload)

    assert list_response.status_code == 200, list_response.text
    list_payload = cast(dict[str, object], list_response.json())
    assert list_payload["tenant_id"] == "tenant.alpha"
    assert list_payload["count"] == 1
    jobs = cast(list[dict[str, object]], list_payload["jobs"])
    assert jobs[0]["job_id"] == job_id

    assert status_response.status_code == 200, status_response.text
    status_payload = cast(dict[str, object], status_response.json())
    assert status_payload == {
        "backend": "portable",
        "job_id": job_id,
        "state": "dry_run",
        "metrics": {},
        "artifact_uri": "",
        "error": "",
    }
    assert other_tenant_response.status_code == 404


@pytest.mark.asyncio
async def test_managed_models_endpoint_exposes_real_registry(
    tmp_path: Path,
) -> None:
    """The managed models endpoint should expose the production model registry."""
    app = _finetune_app(tmp_path / "models")
    transport = ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/v1/finetune/managed/models",
            params={"include_experimental": "false"},
        )

    assert response.status_code == 200, response.text
    payload = cast(dict[str, object], response.json())
    models = cast(list[dict[str, object]], payload["models"])
    factcg_records = [
        model for model in models if model["alias"] == "factcg-deberta-v3-large"
    ]
    assert len(factcg_records) == 1
    assert factcg_records[0]["status"] == "stable"
