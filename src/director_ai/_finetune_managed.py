# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed-training REST endpoints

"""Managed-training lane of the fine-tuning REST API.

Registers the ``/managed/*`` endpoints — submit, list, status, cancel,
model registry, and pre-activation benchmarking — against a router owned
by :func:`director_ai.finetune_api.create_finetune_router`. The local
training lane (upload, worker thread, activation) stays in the facade.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from director_ai.finetune_jobs import ManagedTrainingRecord, _ManagedJobStore

from ._finetune_schemas import _FASTAPI_AVAILABLE

if _FASTAPI_AVAILABLE:
    from fastapi import APIRouter, HTTPException, Request

    from ._finetune_schemas import (
        ManagedModelBenchmarkRequest,
        ManagedTrainingLookupRequest,
        ManagedTrainingRequest,
    )

__all__ = ["_managed_record_to_dict", "register_managed_routes"]


def _managed_record_to_dict(record: ManagedTrainingRecord) -> dict[str, Any]:
    """Serialise a managed training record for REST responses."""
    return {
        "job_id": record.job_id,
        "backend": record.backend,
        "state": record.state,
        "tenant_id": record.tenant_id,
        "dry_run": record.dry_run,
        "submitted_at": record.submitted_at,
        "display_name": record.display_name,
        "output_uri": record.output_uri,
        "console_uri": record.console_uri,
        "error": record.error,
    }


def register_managed_routes(
    router: APIRouter,
    managed_store: _ManagedJobStore,
    tenant_from_request: Callable[[Request], str],
) -> None:
    """Register the managed-training endpoints on the fine-tuning router."""

    @router.post("/managed/submit")
    async def submit_managed_training(
        req: ManagedTrainingRequest, request: Request
    ) -> dict[str, Any]:
        """Submit or dry-run a managed training job."""
        from director_ai.core.training.jobs import (
            TrainingHardware,
            TrainingJobSpec,
            build_internal_suite_spec,
            submit_training_job,
        )

        hardware = TrainingHardware(
            machine_type=req.machine_type,
            accelerator_type=req.accelerator_type,
            accelerator_count=req.accelerator_count,
            boot_disk_gb=req.boot_disk_gb,
        )
        if req.suite:
            spec = build_internal_suite_spec(
                suite=req.suite,
                dataset_uri=req.dataset_uri,
                output_uri=req.output_uri,
                project=req.project,
                region=req.region,
                container_image_uri=req.container_image_uri,
                hardware=hardware,
            )
        else:
            spec = TrainingJobSpec(
                display_name=req.display_name,
                caller="product",
                dataset_uri=req.dataset_uri,
                output_uri=req.output_uri,
                eval_uri=req.eval_uri,
                project=req.project,
                region=req.region,
                base_model=req.base_model,
                allow_experimental_model=req.allow_experimental_model,
                epochs=req.epochs,
                batch_size=req.batch_size,
                learning_rate=req.learning_rate,
                timeout_minutes=req.timeout_minutes,
                container_image_uri=req.container_image_uri,
                service_account=req.service_account,
                network=req.network,
                hardware=hardware,
            )
        try:
            submission = submit_training_job(
                spec,
                backend=req.backend,
                dry_run=req.dry_run,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        tenant_id = tenant_from_request(request)
        managed_store.add(
            ManagedTrainingRecord(
                job_id=submission.job_id,
                backend=submission.backend,
                state=submission.state,
                tenant_id=tenant_id,
                dry_run=submission.dry_run,
                submitted_at=submission.submitted_at,
                display_name=spec.display_name,
                output_uri=spec.output_uri,
                console_uri=submission.console_uri,
            )
        )
        return {
            "backend": submission.backend,
            "job_id": submission.job_id,
            "state": submission.state,
            "dry_run": submission.dry_run,
            "tenant_id": tenant_id,
            "submitted_at": submission.submitted_at,
            "console_uri": submission.console_uri,
            "request": submission.request,
        }

    @router.get("/managed/jobs")
    async def list_managed_training_jobs(request: Request) -> dict[str, Any]:
        """List managed training submissions for the current tenant."""
        tenant_id = tenant_from_request(request)
        records = managed_store.list_for_tenant(tenant_id)
        return {
            "tenant_id": tenant_id,
            "count": len(records),
            "jobs": [_managed_record_to_dict(record) for record in records],
        }

    @router.post("/managed/status")
    async def get_managed_training_status(
        req: ManagedTrainingLookupRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Return backend status for a managed training job."""
        from director_ai.core.training.jobs import get_training_backend

        tenant_id = tenant_from_request(request)
        record = managed_store.get(tenant_id, req.job_id)
        if record is None:
            raise HTTPException(404, "Managed training job not found")
        if record.backend != req.backend:
            raise HTTPException(409, f"Job was submitted to backend {record.backend!r}")
        if record.dry_run:
            return {
                "backend": record.backend,
                "job_id": record.job_id,
                "state": record.state,
                "metrics": {},
                "artifact_uri": "",
                "error": record.error,
            }
        try:
            status = get_training_backend(req.backend).status(req.job_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, f"Training backend status failed: {exc}") from exc

        managed_store.update_state(
            tenant_id,
            req.job_id,
            status.state,
            error=status.error,
        )
        return {
            "backend": status.backend,
            "job_id": status.job_id,
            "state": status.state,
            "metrics": status.metrics,
            "artifact_uri": status.artifact_uri,
            "error": status.error,
        }

    @router.post("/managed/cancel")
    async def cancel_managed_training(
        req: ManagedTrainingLookupRequest,
        request: Request,
    ) -> dict[str, Any]:
        """Cancel a managed training job owned by the current tenant."""
        from director_ai.core.training.jobs import get_training_backend

        tenant_id = tenant_from_request(request)
        record = managed_store.get(tenant_id, req.job_id)
        if record is None:
            raise HTTPException(404, "Managed training job not found")
        if record.backend != req.backend:
            raise HTTPException(409, f"Job was submitted to backend {record.backend!r}")
        if record.dry_run:
            raise HTTPException(
                409, "Dry-run managed training jobs cannot be cancelled"
            )
        try:
            status = get_training_backend(req.backend).cancel(req.job_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, f"Training backend cancel failed: {exc}") from exc

        managed_store.update_state(
            tenant_id,
            req.job_id,
            status.state,
            error=status.error,
        )
        return {
            "backend": status.backend,
            "job_id": status.job_id,
            "state": status.state,
            "metrics": status.metrics,
            "artifact_uri": status.artifact_uri,
            "error": status.error,
        }

    @router.get("/managed/models")
    async def list_managed_training_models(
        include_experimental: bool = False,
    ) -> dict[str, Any]:
        """List selectable managed fine-tune base models."""
        from director_ai.core.training.model_registry import (
            finetune_model_registry_to_dict,
        )

        return {
            "models": finetune_model_registry_to_dict(
                include_experimental=include_experimental,
            ),
        }

    @router.post("/managed/benchmark-models")
    async def benchmark_managed_training_models(
        req: ManagedModelBenchmarkRequest,
    ) -> dict[str, Any]:
        """Benchmark trained model artifacts before activation."""
        from director_ai.core.training.finetune_benchmark import (
            benchmark_model_candidates,
        )

        try:
            report = benchmark_model_candidates(
                req.model_artifacts,
                general_path=req.general_path,
                eval_path=req.eval_path,
                batch_size=req.batch_size,
                allow_experimental=req.allow_experimental_model,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        payload: dict[str, Any] = report.to_dict()
        return payload
