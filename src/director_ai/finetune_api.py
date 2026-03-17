# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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

import contextlib
import json
import logging
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("DirectorAI.FinetuneAPI")

try:
    from fastapi import APIRouter, HTTPException, UploadFile
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

_DEFAULT_MODELS_DIR = Path("./director-models")
_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB
_MAX_CONCURRENT_JOBS = 4


# ── Job state ────────────────────────────────────────────────────────


@dataclass
class FinetuneJob:
    job_id: str
    state: str = (
        "pending"  # pending, validating, training, benchmarking, completed, failed
    )
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    config: dict = field(default_factory=dict)
    validation_report: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    regression_report: dict = field(default_factory=dict)
    model_path: str = ""
    error: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0
    activated: bool = False


class _JobStore:
    """Thread-safe in-memory job store."""

    def __init__(self):
        self._jobs: dict[str, FinetuneJob] = {}
        self._lock = threading.Lock()

    def create(self, config: dict) -> FinetuneJob:
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
        with self._lock:
            return self._jobs.get(job_id)

    def list_all(self) -> list[FinetuneJob]:
        with self._lock:
            return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None


# ── Pydantic models ──────────────────────────────────────────────────

if _FASTAPI_AVAILABLE:

    class ValidateRequest(BaseModel):
        epochs: int = Field(3, ge=1, le=20)
        batch_size: int = Field(16, ge=1, le=128)

    class StartRequest(BaseModel):
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
        job_id: str
        state: str
        progress: float
        current_step: int
        total_steps: int
        error: str = ""

    class ModelInfo(BaseModel):
        job_id: str
        model_path: str
        activated: bool
        created_at: float
        metrics: dict = {}
        regression_report: dict = {}


# ── Training worker ──────────────────────────────────────────────────


def _run_training_worker(job: FinetuneJob, data_path: Path, models_dir: Path):
    """Background thread that runs the fine-tuning pipeline."""
    train_path: Path | None = None
    eval_path: Path | None = None
    try:
        from director_ai.core.training.finetune import FinetuneConfig, finetune_nli

        job.state = "training"
        cfg = job.config

        output_dir = str(models_dir / job.job_id)
        config = FinetuneConfig(
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
    with contextlib.suppress(PermissionError):
        upload_dir.mkdir(parents=True, exist_ok=True)

    store = _JobStore()
    router = APIRouter(tags=["finetune"])

    @router.post("/validate")
    async def validate_data(file: UploadFile, req: ValidateRequest | None = None):
        """Validate uploaded JSONL data before training."""
        if req is None:
            req = ValidateRequest()  # type: ignore[call-arg]

        content = await _read_upload_with_limit(file)
        data_path = upload_dir / f"validate_{uuid.uuid4().hex[:8]}.jsonl"
        try:
            data_path.write_bytes(content)

            from director_ai.core.training.finetune_validator import validate_finetune_data

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
    async def start_training(file: UploadFile, req: StartRequest | None = None):
        """Upload data and start a fine-tuning job."""
        if req is None:
            req = StartRequest()  # type: ignore[call-arg]

        content = await _read_upload_with_limit(file)
        job_id_prefix = uuid.uuid4().hex[:8]
        data_path = upload_dir / f"data_{job_id_prefix}.jsonl"
        data_path.write_bytes(content)

        from director_ai.core.training.finetune_validator import validate_finetune_data

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

    @router.get("/{job_id}")
    async def get_job_status(job_id: str):
        """Get job status and progress."""
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        return JobStatus(
            job_id=job.job_id,
            state=job.state,
            progress=job.progress,
            current_step=job.current_step,
            total_steps=job.total_steps,
            error=job.error,
        ).model_dump()

    @router.get("/{job_id}/result")
    async def get_job_result(job_id: str):
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
    async def activate_model(job_id: str):
        """Activate a fine-tuned model as the default scorer."""
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        if job.state != "completed":
            raise HTTPException(
                409,
                f"Job {job_id} is not completed (state={job.state})",
            )
        job.activated = True
        logger.info("Model %s activated: %s", job_id, job.model_path)
        return {"job_id": job_id, "activated": True, "model_path": job.model_path}

    @router.post("/{job_id}/rollback")
    async def rollback_model(job_id: str):
        """Deactivate fine-tuned model, revert to baseline."""
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, f"Job {job_id} not found")
        job.activated = False
        logger.info("Model %s rolled back", job_id)
        return {"job_id": job_id, "activated": False}

    @router.get("/", name="list_models")
    async def list_models():
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
    async def delete_model(job_id: str):
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
