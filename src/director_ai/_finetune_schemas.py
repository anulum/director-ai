# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning API request/response contracts

"""Pydantic request/response contracts for the fine-tuning REST API.

The models are defined only when FastAPI (and therefore Pydantic v2) is
installed, mirroring the optional ``[server]`` extra; consumers check
``_FASTAPI_AVAILABLE`` before referencing them. The endpoint handlers
composing these contracts live in :mod:`director_ai.finetune_api` (local
lane) and :mod:`director_ai._finetune_managed` (managed lane).
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, Field

    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

__all__ = [
    "_FASTAPI_AVAILABLE",
    "JobStatus",
    "ManagedModelBenchmarkRequest",
    "ManagedTrainingLookupRequest",
    "ManagedTrainingRequest",
    "ModelInfo",
    "StartRequest",
    "ValidateRequest",
]


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
