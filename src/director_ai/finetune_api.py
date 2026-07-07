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
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("DirectorAI.FinetuneAPI")

try:
    from fastapi import APIRouter, HTTPException, Request, UploadFile
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

_DEFAULT_MODELS_DIR = Path("./director-models")
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
_MAX_CONCURRENT_JOBS = 4
_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


# ── Job state ────────────────────────────────────────────────────────


@dataclass
class FinetuneJob:
    """In-process state for one local fine-tuning job."""

    job_id: str
    state: str = (
        "pending"  # pending, validating, training, benchmarking, completed, failed
    )
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    validation_report: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    regression_report: dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    error: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0
    activated: bool = False


class _JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self) -> None:
        self._jobs: dict[str, FinetuneJob] = {}
        self._lock = threading.Lock()

    def create(self, config: dict[str, Any]) -> FinetuneJob:
        """Create a job unless the in-process concurrency cap is reached."""
        with self._lock:
            active = sum(
                1
                for j in self._jobs.values()
                if j.state in ("training", "validating", "benchmarking")
            )
            if active >= _MAX_CONCURRENT_JOBS:
                raise ValueError(
                    f"Too many concurrent jobs ({active}/{_MAX_CONCURRENT_JOBS})",
                )
        job = FinetuneJob(
            job_id=uuid.uuid4().hex,
            config=config,
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> FinetuneJob | None:
        """Return a job by id, or ``None`` when it is unknown."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[FinetuneJob]:
        """Return a snapshot of all known jobs."""
        with self._lock:
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        """Delete a job record and report whether it existed."""
        with self._lock:
            return self._jobs.pop(job_id, None) is not None


@dataclass
class ManagedTrainingRecord:
    """Ledger entry for one managed training submission."""

    job_id: str
    backend: str
    state: str
    tenant_id: str
    dry_run: bool
    submitted_at: float
    display_name: str
    output_uri: str
    console_uri: str = ""
    error: str = ""


class _ManagedJobStore:
    """Thread-safe in-process ledger for managed training submissions."""

    def __init__(self) -> None:
        self._jobs: dict[str, ManagedTrainingRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: ManagedTrainingRecord) -> None:
        """Store or replace a managed-training record by job id."""
        with self._lock:
            self._jobs[record.job_id] = record

    def get(self, tenant_id: str, job_id: str) -> ManagedTrainingRecord | None:
        """Return a tenant-owned managed-training record."""
        with self._lock:
            record = self._jobs.get(job_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    def list_for_tenant(self, tenant_id: str) -> list[ManagedTrainingRecord]:
        """Return all managed-training records visible to one tenant."""
        with self._lock:
            return [
                record
                for record in self._jobs.values()
                if record.tenant_id == tenant_id
            ]

    def update_state(
        self,
        tenant_id: str,
        job_id: str,
        state: str,
        *,
        error: str = "",
    ) -> ManagedTrainingRecord | None:
        """Update a tenant-owned record state and optional error."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.tenant_id != tenant_id:
                return None
            record.state = state
            record.error = error
            return record


# ── Pydantic models ──────────────────────────────────────────────────

if _FASTAPI_AVAILABLE:

    class ValidateRequest(BaseModel):
        """Validation parameters for uploaded fine-tuning JSONL."""

        epochs: int = Field(3, ge=1, le=20)
        batch_size: int = Field(16, ge=1, le=128)

    class StartRequest(BaseModel):
        """Configuration for starting a local fine-tuning job."""

        base_model: str = "factcg-deberta-v3-large"
        allow_experimental_model: bool = False
        epochs: int = Field(3, ge=1, le=20)
        batch_size: int = Field(16, ge=1, le=128)
        learning_rate: float = Field(2e-5, gt=0, le=1e-3)
        mix_general_data: bool = False
        general_data_ratio: float = Field(0.2, ge=0.0, le=0.5)
        early_stopping_patience: int = Field(0, ge=0, le=20)
        class_weighted_loss: bool = False
        auto_benchmark: bool = True
        auto_onnx_export: bool = False

    class JobStatus(BaseModel):
        """Public job progress response."""

        job_id: str
        state: str
        progress: float
        current_step: int
        total_steps: int
        error: str = ""

    class ModelInfo(BaseModel):
        """Stored fine-tuned model metadata."""

        job_id: str
        model_path: str
        activated: bool
        created_at: float
        metrics: dict[str, Any] = {}
        regression_report: dict[str, Any] = {}

    class ManagedTrainingRequest(BaseModel):
        """Managed-training submission request."""

        backend: str = "vertex"
        dry_run: bool = True
        display_name: str = "director-ai-managed-training"
        dataset_uri: str
        output_uri: str
        eval_uri: str | None = None
        project: str | None = None
        region: str = "us-central1"
        container_image_uri: str
        base_model: str = "factcg-deberta-v3-large"
        allow_experimental_model: bool = False
        machine_type: str = "g2-standard-8"
        accelerator_type: str = "NVIDIA_L4"
        accelerator_count: int = Field(1, ge=0, le=8)
        boot_disk_gb: int = Field(100, ge=50, le=4096)
        epochs: int = Field(3, ge=1, le=20)
        batch_size: int = Field(16, ge=1, le=128)
        learning_rate: float = Field(2e-5, gt=0, le=1e-3)
        timeout_minutes: int = Field(180, ge=1, le=1440)
        service_account: str | None = None
        network: str | None = None
        suite: str = ""

    class ManagedTrainingLookupRequest(BaseModel):
        """Managed-training job lookup request."""

        backend: str = "vertex"
        job_id: str = Field(..., min_length=1, max_length=500)

    class ManagedModelBenchmarkRequest(BaseModel):
        """Managed-model benchmark request."""

        model_artifacts: dict[str, str]
        general_path: str | None = None
        eval_path: str | None = None
        batch_size: int | None = Field(None, ge=1, le=128)
        allow_experimental_model: bool = False


# ── Training worker ──────────────────────────────────────────────────


def _run_training_worker(job: FinetuneJob, data_path: Path, models_dir: Path) -> None:
    """Background thread that runs the fine-tuning pipeline."""
    train_path: Path | None = None
    eval_path: Path | None = None
    try:
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli
        from director_ai.core.training.model_registry import resolve_finetune_model

        job.state = "training"
        cfg = job.config
        model_profile = resolve_finetune_model(
            cfg.get("base_model", "factcg-deberta-v3-large"),
            allow_experimental=cfg.get("allow_experimental_model", False),
        )

        output_dir = str(models_dir / job.job_id)
        config = FinetuneConfig(
            base_model=model_profile.model_id,
            output_dir=output_dir,
            epochs=cfg.get("epochs", 3),
            batch_size=cfg.get("batch_size", 16),
            learning_rate=cfg.get("learning_rate", 2e-5),
            mix_general_data=cfg.get("mix_general_data", False),
            general_data_ratio=cfg.get("general_data_ratio", 0.2),
            early_stopping_patience=cfg.get("early_stopping_patience", 0),
            class_weighted_loss=cfg.get("class_weighted_loss", False),
            auto_benchmark=cfg.get("auto_benchmark", True),
            auto_onnx_export=cfg.get("auto_onnx_export", False),
        )

        import random

        from director_ai.core.training.finetune import _load_jsonl

        rows = _load_jsonl(data_path)
        rng = random.Random(42)
        rng.shuffle(rows)
        n_eval = max(1, int(len(rows) * 0.1))
        eval_rows = rows[:n_eval]
        train_rows = rows[n_eval:]

        train_path = data_path.parent / f"{job.job_id}_train.jsonl"
        eval_path = data_path.parent / f"{job.job_id}_eval.jsonl"
        for p, r in [(train_path, train_rows), (eval_path, eval_rows)]:
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in r)

        job.total_steps = len(train_rows) // config.batch_size * config.epochs

        result = finetune_nli(str(train_path), eval_path=str(eval_path), config=config)

        # Set result fields before state so readers see consistent data
        job.model_path = result.output_dir
        job.metrics = result.eval_metrics
        job.regression_report = result.regression_report
        job.completed_at = time.time()
        job.progress = 1.0
        job.state = "completed"

        logger.info(
            "Job %s completed: bal_acc=%.1f%%",
            job.job_id,
            result.best_balanced_accuracy * 100,
        )

    except Exception as exc:
        job.error = str(exc)
        job.state = "failed"
        logger.error("Job %s failed: %s", job.job_id, exc)
    finally:
        paths_to_clean: tuple[Path | None, ...] = (data_path, train_path, eval_path)
        for _p in paths_to_clean:
            if _p is not None:
                _p.unlink(missing_ok=True)


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

    store = _JobStore()
    managed_store = _ManagedJobStore()
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

        thread = threading.Thread(
            target=_run_training_worker,
            args=(job, data_path, models_dir),
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
        tenant_id = _tenant_from_request(request)
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
        tenant_id = _tenant_from_request(request)
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

        tenant_id = _tenant_from_request(request)
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

        tenant_id = _tenant_from_request(request)
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
