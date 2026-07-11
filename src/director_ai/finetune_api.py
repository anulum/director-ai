# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning REST API

"""Server-side fine-tuning API (Phase C).

Endpoints::

    POST /v1/finetune/validate      — validate JSONL before training
    POST /v1/finetune/start         — start fine-tuning job
    GET  /v1/finetune/{job_id}      — job status + progress
    GET  /v1/finetune/{job_id}/result — regression report + metrics
    POST /v1/finetune/{job_id}/activate — activate fine-tuned model
    POST /v1/finetune/{job_id}/rollback — revert to baseline
    GET  /v1/finetune/models        — list all fine-tuned models
    DELETE /v1/finetune/{job_id}    — delete model + artifacts

Mount via::

    from director_ai.finetune_api import create_finetune_router
    app.include_router(create_finetune_router(), prefix="/v1/finetune")

This module owns the local training lane (upload validation, worker
spawn, activation/rollback/deletion) and the router factory; the sibling
responsibilities are composed from dedicated modules — request/response
contracts in :mod:`._finetune_schemas`, the background training worker in
:mod:`._finetune_worker`, and the ``/managed/*`` endpoints in
:mod:`._finetune_managed`.

Job records persist to ``<models_dir>/finetune_jobs.sqlite3`` (see
:mod:`director_ai.finetune_jobs`), so job state and model-activation
designations survive restarts and are shared across server workers.
"""

from __future__ import annotations

import logging
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

# Job records and stores live in ``finetune_jobs``; the private names are
# re-exported (redundant aliases) because tests and the server exercise them
# through this module's historical surface.
from director_ai.finetune_jobs import _MAX_CONCURRENT_JOBS as _MAX_CONCURRENT_JOBS
from director_ai.finetune_jobs import FinetuneJob as FinetuneJob
from director_ai.finetune_jobs import ManagedTrainingRecord as ManagedTrainingRecord
from director_ai.finetune_jobs import _JobStore as _JobStore
from director_ai.finetune_jobs import _ManagedJobStore as _ManagedJobStore

from ._finetune_managed import _managed_record_to_dict as _managed_record_to_dict
from ._finetune_managed import register_managed_routes as register_managed_routes
from ._finetune_worker import _run_training_worker as _run_training_worker

__all__ = [
    "FinetuneJob",
    "ManagedTrainingRecord",
    "create_finetune_router",
]

logger = logging.getLogger("DirectorAI.FinetuneAPI")

try:
    from fastapi import APIRouter, HTTPException, Request, UploadFile

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

if _FASTAPI_AVAILABLE:
    from ._finetune_schemas import JobStatus as JobStatus
    from ._finetune_schemas import ModelInfo as ModelInfo
    from ._finetune_schemas import StartRequest as StartRequest
    from ._finetune_schemas import ValidateRequest as ValidateRequest

_DEFAULT_MODELS_DIR = Path("./director-models")
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
_JOB_DB_FILENAME = "finetune_jobs.sqlite3"
_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


# ── Router factory ───────────────────────────────────────────────────


async def _read_upload_with_limit(file: UploadFile) -> bytes:
    """Stream-read upload, rejecting before exceeding _MAX_UPLOAD_BYTES."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(64 * 1024):
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                413,
                f"Upload too large (>{_MAX_UPLOAD_BYTES} bytes)",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _tenant_from_request(request: Request) -> str:
    """Return a validated tenant id from request headers."""
    tenant_id = str(request.headers.get("X-Tenant-ID", ""))
    if not tenant_id:
        return ""
    if not _SAFE_TENANT_RE.fullmatch(tenant_id):
        raise HTTPException(
            400,
            "X-Tenant-ID must be 1-128 chars: letters, numbers, dot, "
            "underscore, colon, dash",
        )
    return tenant_id


def create_finetune_router(models_dir: Path | None = None) -> APIRouter:
    """Create the fine-tuning API router.

    Parameters
    ----------
    models_dir : directory for storing fine-tuned models

    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError("pip install director-ai[server]")

    if models_dir is None:
        models_dir = _DEFAULT_MODELS_DIR
    models_dir = Path(models_dir).resolve()
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.warning("Cannot create models dir %s (read-only filesystem)", models_dir)

    upload_dir = models_dir / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Jobs persist next to the model artefacts, so records — including the
    # ``activated`` designations that protect models from deletion — survive
    # restarts and are shared across workers (BUG-2). The stores fall back to
    # memory-only when the database cannot be created (read-only filesystem).
    job_db_path = models_dir / _JOB_DB_FILENAME
    store = _JobStore(job_db_path)
    managed_store = _ManagedJobStore(job_db_path)
    router = APIRouter(tags=["finetune"])

    @router.post("/validate")
    async def validate_data(
        file: UploadFile, req: ValidateRequest | None = None
    ) -> dict[str, Any]:
        """Validate uploaded JSONL data before training."""
        if req is None:
            # Pydantic v2's mypy plugin wants explicit kwargs for
            # optional fields; the defaults below mirror the model
            # definition.
            req = ValidateRequest(epochs=3, batch_size=16)

        content = await _read_upload_with_limit(file)
        data_path = upload_dir / f"validate_{uuid.uuid4().hex[:8]}.jsonl"
        try:
            data_path.write_bytes(content)

            from director_ai.core.training.finetune_validator import (
                validate_finetune_data,
            )

            report = validate_finetune_data(str(data_path), epochs=req.epochs)
            return {
                "is_valid": report.is_valid,
                "total_samples": report.total_samples,
                "label_distribution": report.label_distribution,
                "class_balance_ratio": report.class_balance_ratio,
                "duplicate_count": report.duplicate_count,
                "estimated_train_time_min": report.estimated_train_time_min,
                "estimated_cost_usd": report.estimated_cost_usd,
                "warnings": report.warnings,
                "errors": report.errors,
            }
        finally:
            data_path.unlink(missing_ok=True)

    @router.post("/start")
    async def start_training(
        file: UploadFile, req: StartRequest | None = None
    ) -> dict[str, Any]:
        """Upload data and start a fine-tuning job."""
        if req is None:
            req = StartRequest(
                base_model="factcg-deberta-v3-large",
                allow_experimental_model=False,
                epochs=3,
                batch_size=16,
                learning_rate=2e-5,
                mix_general_data=False,
                general_data_ratio=0.2,
                early_stopping_patience=0,
                class_weighted_loss=False,
                auto_benchmark=True,
            )

        content = await _read_upload_with_limit(file)
        job_id_prefix = uuid.uuid4().hex[:8]
        data_path = upload_dir / f"data_{job_id_prefix}.jsonl"
        data_path.write_bytes(content)

        from director_ai.core.training.finetune_validator import validate_finetune_data
        from director_ai.core.training.model_registry import resolve_finetune_model

        report = validate_finetune_data(str(data_path))
        if not report.is_valid:
            data_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Data validation failed",
                    "errors": report.errors,
                    "warnings": report.warnings,
                },
            )
        try:
            resolve_finetune_model(
                req.base_model,
                allow_experimental=req.allow_experimental_model,
            )
        except ValueError as exc:
            data_path.unlink(missing_ok=True)
            raise HTTPException(422, str(exc)) from exc

        try:
            job = store.create(req.model_dump())
        except ValueError as exc:
            data_path.unlink(missing_ok=True)
            raise HTTPException(429, str(exc)) from exc

        job.validation_report = {
            "total_samples": report.total_samples,
            "label_distribution": report.label_distribution,
            "estimated_train_time_min": report.estimated_train_time_min,
            "estimated_cost_usd": report.estimated_cost_usd,
        }
        store.save(job)

        thread = threading.Thread(
            target=_run_training_worker,
            args=(job, data_path, models_dir, store),
            daemon=True,
        )
        thread.start()

        return {
            "job_id": job.job_id,
            "state": job.state,
            "estimated_time_min": report.estimated_train_time_min,
            "estimated_cost_usd": report.estimated_cost_usd,
            "total_samples": report.total_samples,
        }

    register_managed_routes(router, managed_store, _tenant_from_request)

    @router.get("/{job_id}")
    async def get_job_status(job_id: str) -> dict[str, Any]:
        """Get job status and progress."""
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        status: dict[str, Any] = JobStatus(
            job_id=job.job_id,
            state=job.state,
            progress=job.progress,
            current_step=job.current_step,
            total_steps=job.total_steps,
            error=job.error,
        ).model_dump()
        return status

    @router.get("/{job_id}/result")
    async def get_job_result(job_id: str) -> dict[str, Any]:
        """Get training results and regression report."""
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if job.state not in ("completed", "failed"):
            raise HTTPException(409, f"Job {job_id} is still {job.state}")
        return {
            "job_id": job.job_id,
            "state": job.state,
            "metrics": job.metrics,
            "regression_report": job.regression_report,
            "model_path": job.model_path,
            "error": job.error,
        }

    @router.post("/{job_id}/activate")
    async def activate_model(job_id: str, request: Request) -> dict[str, Any]:
        """Activate a completed fine-tune as the live default scorer.

        Mounted in the Director-AI server, activation hot-swaps the running
        scorer to this ``model_path`` so subsequent reviews use the fine-tuned
        model with no restart (the swap is serialised and the ground-truth store
        reused). The designation is also recorded (surfaced by ``list_models``)
        and protects the model from deletion until it is rolled back. Mounted
        standalone (no server scorer to swap), activation records the
        designation only: set the server's ``nli_model`` to this ``model_path``
        and restart to serve it.
        """
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if job.state != "completed":
            raise HTTPException(
                409,
                f"Job {job_id} is not completed (state={job.state})",
            )
        job.activated = True
        store.save(job)
        server_state = getattr(request.app.state, "_state", None)
        activator = (
            server_state.get("scorer_activator")
            if isinstance(server_state, dict)
            else None
        )
        hot_swapped = False
        if activator is not None and job.model_path:
            await activator(job.model_path)
            hot_swapped = True
        logger.info(
            "Model %s marked active: %s (hot_swapped=%s)",
            job_id,
            job.model_path,
            hot_swapped,
        )
        detail = (
            "Model activated and hot-swapped into the running scorer; "
            "subsequent reviews use it."
            if hot_swapped
            else (
                "Model marked active and protected from deletion. To serve it, "
                "set the server's nli_model to this model_path and restart; the "
                "running scorer is not hot-swapped."
            )
        )
        return {
            "job_id": job_id,
            "activated": True,
            "model_path": job.model_path,
            "hot_swapped": hot_swapped,
            "detail": detail,
        }

    @router.post("/{job_id}/rollback")
    async def rollback_model(job_id: str) -> dict[str, Any]:
        """Deactivate fine-tuned model, revert to baseline."""
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        job.activated = False
        store.save(job)
        logger.info("Model %s rolled back", job_id)
        return {"job_id": job_id, "activated": False}

    @router.get("/", name="list_models")
    async def list_models() -> dict[str, Any]:
        """List all fine-tuning jobs and models."""
        jobs = store.list_all()
        return {
            "models": [
                ModelInfo(
                    job_id=j.job_id,
                    model_path=j.model_path,
                    activated=j.activated,
                    created_at=j.created_at,
                    metrics=j.metrics,
                    regression_report=j.regression_report,
                ).model_dump()
                for j in jobs
            ],
        }

    @router.delete("/{job_id}")
    async def delete_model(job_id: str) -> dict[str, Any]:
        """Delete a fine-tuned model and its artifacts."""
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if job.activated:
            raise HTTPException(
                409,
                "Cannot delete an activated model — rollback first",
            )

        if job.model_path:
            target = Path(job.model_path).resolve()
            if target.is_relative_to(models_dir) and target.exists():
                shutil.rmtree(target, ignore_errors=True)

        store.delete(job_id)
        logger.info("Job %s deleted", job_id)
        return {"deleted": True, "job_id": job_id}

    return router
