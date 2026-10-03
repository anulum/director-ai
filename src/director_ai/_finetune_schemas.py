# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning API request/response contracts

"""Pydantic request/response contracts for the fine-tuning REST API.

The models are defined when Pydantic is installed. The historical
``_FASTAPI_AVAILABLE`` flag records that import; consumers check it before
referencing a model. The endpoint handlers
composing these contracts live in :mod:`director_ai.finetune_api` (local
lane) and :mod:`director_ai._finetune_managed` (managed lane).
"""

from __future__ import annotations

from typing import Any

try:
    from pydantic import BaseModel, ConfigDict, Field

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
        """Validation parameters for uploaded fine-tuning JSONL.

        Attributes
        ----------
        epochs : int
            Training epochs; defaults to 3 and accepts 1 through 20.
        batch_size : int
            Batch size; defaults to 16 and accepts 1 through 128.
        """

        epochs: int = Field(default=3, ge=1, le=20)
        batch_size: int = Field(default=16, ge=1, le=128)

    class StartRequest(BaseModel):
        """Configuration for starting a local fine-tuning job.

        Attributes
        ----------
        base_model : str
            Stable scorer alias selected by default.
        allow_experimental_model : bool
            Explicit opt-in for experimental model selection; false by default.
        epochs : int
            Training epochs; defaults to 3 and accepts 1 through 20.
        batch_size : int
            Batch size; defaults to 16 and accepts 1 through 128.
        learning_rate : float
            Positive learning rate, at most 0.001; defaults to 0.00002.
        mix_general_data : bool
            Whether to mix general examples into training; false by default.
        general_data_ratio : float
            General-example proportion from 0 to 0.5; defaults to 0.2.
        early_stopping_patience : int
            Early-stop patience from 0 through 20; defaults to zero.
        class_weighted_loss : bool
            Whether to weight loss by class frequency; false by default.
        auto_benchmark : bool
            Whether to request automatic local benchmarking; true by default.
        auto_onnx_export : bool
            Whether to request ONNX export; false by default.
        """

        base_model: str = "factcg-deberta-v3-large"
        allow_experimental_model: bool = False
        epochs: int = Field(default=3, ge=1, le=20)
        batch_size: int = Field(default=16, ge=1, le=128)
        learning_rate: float = Field(default=2e-5, gt=0, le=1e-3)
        mix_general_data: bool = False
        general_data_ratio: float = Field(default=0.2, ge=0.0, le=0.5)
        early_stopping_patience: int = Field(default=0, ge=0, le=20)
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
        """Managed-training submission request.

        Attributes
        ----------
        backend : str
            Managed backend identifier; defaults to vertex.
        dry_run : bool
            Prepare the request without submitting training; true by default.
        display_name : str
            Human-readable managed training job name.
        dataset_uri : str
            Required managed training dataset location.
        output_uri : str
            Required location for managed job outputs.
        eval_uri : str | None
            Optional managed evaluation dataset location.
        project : str | None
            Optional managed provider project identifier.
        region : str
            Managed provider region; defaults to us-central1.
        container_image_uri : str
            Required container image used by the managed job.
        base_model : str
            Stable scorer alias selected by default.
        allow_experimental_model : bool
            Explicit opt-in for experimental model selection; false by default.
        machine_type : str
            Managed machine class; defaults to g2-standard-8.
        accelerator_type : str
            Managed accelerator class; defaults to NVIDIA_L4.
        accelerator_count : int
            Accelerator count from 0 through 8; defaults to one.
        boot_disk_gb : int
            Boot disk size from 50 through 4096 GiB; defaults to 100.
        epochs : int
            Training epochs; defaults to 3 and accepts 1 through 20.
        batch_size : int
            Batch size; defaults to 16 and accepts 1 through 128.
        learning_rate : float
            Positive learning rate, at most 0.001; defaults to 0.00002.
        timeout_minutes : int
            Timeout from 1 through 1440 minutes; defaults to 180.
        service_account : str | None
            Optional managed execution service-account identifier.
        network : str | None
            Optional managed network identifier.
        suite : str
            Optional benchmark suite selector.
        """

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
        accelerator_count: int = Field(default=1, ge=0, le=8)
        boot_disk_gb: int = Field(default=100, ge=50, le=4096)
        epochs: int = Field(default=3, ge=1, le=20)
        batch_size: int = Field(default=16, ge=1, le=128)
        learning_rate: float = Field(default=2e-5, gt=0, le=1e-3)
        timeout_minutes: int = Field(default=180, ge=1, le=1440)
        service_account: str | None = None
        network: str | None = None
        suite: str = ""

    class ManagedTrainingLookupRequest(BaseModel):
        """Managed-training job lookup request."""

        backend: str = "vertex"
        job_id: str = Field(..., min_length=1, max_length=500)

    class ManagedModelBenchmarkRequest(BaseModel):
        """Managed-model benchmark request.

        Attributes
        ----------
        model_jobs : dict[str, str]
            Model aliases mapped to managed job identifiers.
        general_dataset_id : str
            Identifier of the general benchmark dataset.
        eval_dataset_id : str | None
            Optional domain evaluation dataset identifier.
        batch_size : int | None
            Optional override from 1 through 128; defaults to None.
        allow_experimental_model : bool
            Explicit opt-in for experimental model selection; false by default.
        """

        model_config = ConfigDict(extra="forbid")
        model_jobs: dict[str, str]
        general_dataset_id: str
        eval_dataset_id: str | None = None
        batch_size: int | None = Field(default=None, ge=1, le=128)
        allow_experimental_model: bool = False
