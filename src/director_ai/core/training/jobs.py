# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed Training Jobs

"""Managed training job specifications and backends.

The same interface serves customer-triggered fine-tuning jobs and internal
GPU validation jobs. Cloud backends can dry-run to produce an auditable job
request without provisioning paid compute.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .model_registry import (
    DEFAULT_FINE_TUNE_MODEL_ALIAS,
    TrainingModelProfile,
    resolve_finetune_model,
)

_LOCAL_BACKEND = "local"
_VERTEX_BACKEND = "vertex"
_VALID_BACKENDS = (_LOCAL_BACKEND, _VERTEX_BACKEND)
_VALID_CALLERS = ("product", "internal")
_VALID_TASKS = ("finetune-nli", "suite")
_DEFAULT_CONTAINER_IMAGE = "python:3.12-slim"
_DEFAULT_VERTEX_REGION = "us-central1"
_MAX_TIMEOUT_MINUTES = 24 * 60


@dataclass(frozen=True)
class TrainingHardware:
    """Compute profile for managed training."""

    machine_type: str = "g2-standard-8"
    accelerator_type: str = "NVIDIA_L4"
    accelerator_count: int = 1
    boot_disk_gb: int = 100
    boot_disk_type: str = "pd-ssd"

    def validate(self) -> None:
        if not self.machine_type:
            raise ValueError("machine_type is required")
        if self.accelerator_count < 0:
            raise ValueError("accelerator_count must be >= 0")
        if self.accelerator_count and not self.accelerator_type:
            raise ValueError("accelerator_type is required when GPUs are requested")
        if self.boot_disk_gb < 50:
            raise ValueError("boot_disk_gb must be at least 50")


@dataclass(frozen=True)
class TrainingJobSpec:
    """Portable managed training job request."""

    display_name: str
    task_type: str = "finetune-nli"
    caller: str = "product"
    dataset_uri: str = ""
    output_uri: str = ""
    eval_uri: str | None = None
    project: str | None = None
    region: str = _DEFAULT_VERTEX_REGION
    base_model: str = DEFAULT_FINE_TUNE_MODEL_ALIAS
    allow_experimental_model: bool = False
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    timeout_minutes: int = 180
    container_image_uri: str = _DEFAULT_CONTAINER_IMAGE
    service_account: str | None = None
    network: str | None = None
    hardware: TrainingHardware = field(default_factory=TrainingHardware)
    labels: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)

    def validate(self, backend: str) -> None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}")
        if self.caller not in _VALID_CALLERS:
            raise ValueError(f"caller must be one of {_VALID_CALLERS}")
        if self.task_type not in _VALID_TASKS:
            raise ValueError(f"task_type must be one of {_VALID_TASKS}")
        if not self.display_name:
            raise ValueError("display_name is required")
        if not self.dataset_uri:
            raise ValueError("dataset_uri is required")
        if not self.output_uri:
            raise ValueError("output_uri is required")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.task_type == "finetune-nli":
            self.resolved_model_profile()
        if self.timeout_minutes < 1 or self.timeout_minutes > _MAX_TIMEOUT_MINUTES:
            raise ValueError(
                f"timeout_minutes must be between 1 and {_MAX_TIMEOUT_MINUTES}"
            )
        self.hardware.validate()
        if backend == _VERTEX_BACKEND:
            if not self.project:
                raise ValueError("project is required for vertex backend")
            _require_gcs_uri("dataset_uri", self.dataset_uri)
            _require_gcs_uri("output_uri", self.output_uri)
            if self.eval_uri:
                _require_gcs_uri("eval_uri", self.eval_uri)
            if self.container_image_uri == _DEFAULT_CONTAINER_IMAGE:
                raise ValueError("container_image_uri must be a training image")

    @property
    def dataset_hash(self) -> str:
        return hashlib.sha256(self.dataset_uri.encode("utf-8")).hexdigest()[:16]

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.to_redacted_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_redacted_dict(self) -> dict[str, Any]:
        env = {
            key: ("<redacted>" if _looks_secret(key) else value)
            for key, value in self.env.items()
        }
        return {
            "display_name": self.display_name,
            "task_type": self.task_type,
            "caller": self.caller,
            "dataset_uri": self.dataset_uri,
            "output_uri": self.output_uri,
            "eval_uri": self.eval_uri,
            "project": self.project,
            "region": self.region,
            "base_model": self.base_model,
            "resolved_base_model": self.resolved_model_profile().model_id
            if self.task_type == "finetune-nli"
            else self.base_model,
            "allow_experimental_model": self.allow_experimental_model,
            "model_profile": self.resolved_model_profile().to_dict()
            if self.task_type == "finetune-nli"
            else {},
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "timeout_minutes": self.timeout_minutes,
            "container_image_uri": self.container_image_uri,
            "service_account": self.service_account,
            "network": self.network,
            "hardware": {
                "machine_type": self.hardware.machine_type,
                "accelerator_type": self.hardware.accelerator_type,
                "accelerator_count": self.hardware.accelerator_count,
                "boot_disk_gb": self.hardware.boot_disk_gb,
                "boot_disk_type": self.hardware.boot_disk_type,
            },
            "labels": dict(self.labels),
            "env": env,
            "args": list(self.args),
        }

    def resolved_model_profile(self) -> TrainingModelProfile:
        """Return the validated base-model profile for this training job."""

        return resolve_finetune_model(
            self.base_model,
            allow_experimental=self.allow_experimental_model,
        )


@dataclass(frozen=True)
class TrainingJobSubmission:
    """Submission result for a managed training job."""

    backend: str
    job_id: str
    state: str
    dry_run: bool
    request: dict[str, Any]
    submitted_at: float
    console_uri: str = ""


@dataclass(frozen=True)
class TrainingJobStatus:
    """Backend-neutral managed training status."""

    backend: str
    job_id: str
    state: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_uri: str = ""
    error: str = ""


class TrainingJobBackend(Protocol):
    """Interface implemented by training job backends."""

    name: str

    def submit(self, spec: TrainingJobSpec, *, dry_run: bool) -> TrainingJobSubmission:
        """Submit *spec* or return the request when ``dry_run`` is true."""

    def status(self, job_id: str) -> TrainingJobStatus:
        """Return status for an existing job."""

    def cancel(self, job_id: str) -> TrainingJobStatus:
        """Cancel an existing job."""


class LocalTrainingBackend:
    """Local backend that reuses the existing fine-tuning CLI path."""

    name = _LOCAL_BACKEND

    def submit(self, spec: TrainingJobSpec, *, dry_run: bool) -> TrainingJobSubmission:
        spec.validate(self.name)
        command = _local_command(spec)
        request = {
            "command": command,
            "cwd": str(Path.cwd()),
            "spec": spec.to_redacted_dict(),
            "dataset_hash": spec.dataset_hash,
            "config_hash": spec.config_hash,
        }
        job_id = f"local-{spec.config_hash}"
        if not dry_run:
            _run_local_job(spec)
        return TrainingJobSubmission(
            backend=self.name,
            job_id=job_id,
            state="dry_run" if dry_run else "completed",
            dry_run=dry_run,
            request=request,
            submitted_at=time.time(),
        )

    def status(self, job_id: str) -> TrainingJobStatus:
        return TrainingJobStatus(backend=self.name, job_id=job_id, state="unknown")

    def cancel(self, job_id: str) -> TrainingJobStatus:
        return TrainingJobStatus(
            backend=self.name,
            job_id=job_id,
            state="unsupported",
            error="local managed jobs run synchronously",
        )


class VertexTrainingBackend:
    """Vertex custom training backend."""

    name = _VERTEX_BACKEND

    def submit(self, spec: TrainingJobSpec, *, dry_run: bool) -> TrainingJobSubmission:
        spec.validate(self.name)
        request = build_vertex_custom_job_request(spec)
        job_id = (
            f"projects/{spec.project}/locations/{spec.region}/customJobs/"
            f"dry-run-{spec.config_hash}"
        )
        console_uri = (
            "https://console.cloud.google.com/vertex-ai/training/custom-jobs"
            f"?project={spec.project}"
        )
        if not dry_run:
            aiplatform = importlib.import_module("google.cloud.aiplatform")
            aiplatform.init(
                project=spec.project,
                location=spec.region,
                staging_bucket=spec.output_uri,
            )
            job = aiplatform.CustomJob(
                display_name=spec.display_name,
                worker_pool_specs=request["job_spec"]["worker_pool_specs"],
                base_output_dir=spec.output_uri,
            )
            job.submit(
                service_account=spec.service_account,
                network=spec.network,
                sync=False,
            )
            job_id = str(getattr(job, "resource_name", job_id))
            console_uri = str(getattr(job, "gca_resource", "")) or console_uri
        return TrainingJobSubmission(
            backend=self.name,
            job_id=job_id,
            state="dry_run" if dry_run else "submitted",
            dry_run=dry_run,
            request=request,
            submitted_at=time.time(),
            console_uri=console_uri,
        )

    def status(self, job_id: str) -> TrainingJobStatus:
        aiplatform = importlib.import_module("google.cloud.aiplatform")
        job = aiplatform.CustomJob.get(job_id)
        state = str(getattr(job, "state", "unknown"))
        return TrainingJobStatus(backend=self.name, job_id=job_id, state=state)

    def cancel(self, job_id: str) -> TrainingJobStatus:
        aiplatform = importlib.import_module("google.cloud.aiplatform")
        job = aiplatform.CustomJob.get(job_id)
        job.cancel()
        return TrainingJobStatus(backend=self.name, job_id=job_id, state="cancelled")


def submit_training_job(
    spec: TrainingJobSpec,
    *,
    backend: str,
    dry_run: bool = True,
) -> TrainingJobSubmission:
    """Submit a managed training job through a named backend."""

    return get_training_backend(backend).submit(spec, dry_run=dry_run)


def get_training_backend(name: str) -> TrainingJobBackend:
    """Return the managed training backend named *name*."""

    if name == _LOCAL_BACKEND:
        return LocalTrainingBackend()
    if name == _VERTEX_BACKEND:
        return VertexTrainingBackend()
    raise ValueError(f"backend must be one of {_VALID_BACKENDS}")


def build_internal_suite_spec(
    *,
    suite: str,
    dataset_uri: str,
    output_uri: str,
    project: str | None = None,
    region: str = _DEFAULT_VERTEX_REGION,
    container_image_uri: str = _DEFAULT_CONTAINER_IMAGE,
    hardware: TrainingHardware | None = None,
) -> TrainingJobSpec:
    """Create an internal validation-suite job spec.

    Internal jobs exercise the same managed backend as product jobs, but their
    command runs a test suite instead of fine-tuning on customer data.
    """

    if not suite:
        raise ValueError("suite is required")
    labels = _normalise_labels({"director_ai_caller": "internal", "suite": suite})
    return TrainingJobSpec(
        display_name=f"director-ai-{suite}",
        task_type="suite",
        caller="internal",
        dataset_uri=dataset_uri,
        output_uri=output_uri,
        project=project,
        region=region,
        container_image_uri=container_image_uri,
        hardware=hardware or TrainingHardware(),
        labels=labels,
        args=["python", "-m", "pytest", f"tests/{suite}.py", "-q"],
    )


def build_vertex_custom_job_request(spec: TrainingJobSpec) -> dict[str, Any]:
    """Build the Vertex custom job request for *spec* without submitting it."""

    spec.validate(_VERTEX_BACKEND)
    labels = _normalise_labels(
        {
            "director_ai_caller": spec.caller,
            "director_ai_task": spec.task_type,
            "director_ai_model": spec.resolved_model_profile().alias
            if spec.task_type == "finetune-nli"
            else "suite",
            **spec.labels,
        }
    )
    container_args = spec.args or _container_finetune_args(spec)
    container_env = [{"name": key, "value": value} for key, value in spec.env.items()]
    machine_spec: dict[str, Any] = {"machine_type": spec.hardware.machine_type}
    if spec.hardware.accelerator_count:
        machine_spec["accelerator_type"] = spec.hardware.accelerator_type
        machine_spec["accelerator_count"] = spec.hardware.accelerator_count

    return {
        "display_name": spec.display_name,
        "labels": labels,
        "job_spec": {
            "base_output_directory": {"output_uri_prefix": spec.output_uri},
            "scheduling": {"timeout": f"{spec.timeout_minutes * 60}s"},
            "service_account": spec.service_account,
            "network": spec.network,
            "worker_pool_specs": [
                {
                    "replica_count": 1,
                    "machine_spec": machine_spec,
                    "disk_spec": {
                        "boot_disk_type": spec.hardware.boot_disk_type,
                        "boot_disk_size_gb": spec.hardware.boot_disk_gb,
                    },
                    "container_spec": {
                        "image_uri": spec.container_image_uri,
                        "args": container_args,
                        "env": container_env,
                    },
                }
            ],
        },
    }


def submission_to_json(submission: TrainingJobSubmission) -> str:
    """Serialise a submission result for CLI/API output."""

    return json.dumps(
        {
            "backend": submission.backend,
            "job_id": submission.job_id,
            "state": submission.state,
            "dry_run": submission.dry_run,
            "submitted_at": submission.submitted_at,
            "console_uri": submission.console_uri,
            "request": submission.request,
        },
        indent=2,
        sort_keys=True,
    )


def _container_finetune_args(spec: TrainingJobSpec) -> list[str]:
    model_profile = spec.resolved_model_profile()
    args = [
        "director-ai",
        "finetune",
        spec.dataset_uri,
        "--output",
        spec.output_uri,
        "--epochs",
        str(spec.epochs),
        "--batch-size",
        str(spec.batch_size),
        "--lr",
        str(spec.learning_rate),
        "--base-model",
        model_profile.model_id,
    ]
    if spec.eval_uri:
        args.extend(["--eval", spec.eval_uri])
    return args


def _local_command(spec: TrainingJobSpec) -> list[str]:
    if spec.task_type == "suite":
        return list(spec.args)
    return _container_finetune_args(spec)


def _run_local_job(spec: TrainingJobSpec) -> None:
    if spec.task_type == "suite":
        import pytest

        pytest_args = (
            spec.args[3:] if spec.args[:3] == ["python", "-m", "pytest"] else spec.args
        )
        exit_code = pytest.main(pytest_args)
        if exit_code:
            raise RuntimeError(f"local suite job failed with exit code {exit_code}")
        return

    from .finetune import FinetuneConfig, finetune_nli

    config = FinetuneConfig(
        base_model=spec.resolved_model_profile().model_id,
        output_dir=spec.output_uri,
        epochs=spec.epochs,
        batch_size=spec.batch_size,
        learning_rate=spec.learning_rate,
    )
    finetune_nli(spec.dataset_uri, eval_path=spec.eval_uri, config=config)


def _require_gcs_uri(name: str, uri: str) -> None:
    if not uri.startswith("gs://"):
        raise ValueError(f"{name} must be a gs:// URI for vertex backend")


def _looks_secret(key: str) -> bool:
    normalised = key.lower()
    return any(part in normalised for part in ("token", "secret", "password", "key"))


def _normalise_labels(labels: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in labels.items():
        clean_key = key.lower().replace("_", "-")[:63]
        clean_value = str(value).lower().replace("_", "-")[:63]
        if clean_key and clean_value:
            result[clean_key] = clean_value
    return result


def shell_join(command: list[str]) -> str:
    """Return a display-only shell representation of *command*."""

    return " ".join(shlex.quote(part) for part in command)
