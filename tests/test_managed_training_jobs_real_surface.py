# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Managed training real-surface tests
"""Real subprocess coverage for managed training job public contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypeAlias, cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JsonObject: TypeAlias = dict[str, object]


def _subprocess_env(*, extra_pythonpath: Path | None = None) -> dict[str, str]:
    """Return an environment that imports the checkout production package."""
    env = os.environ.copy()
    path_parts = [str(PROJECT_ROOT / "src")]
    if extra_pythonpath is not None:
        path_parts.insert(0, str(extra_pythonpath))
    existing = env.get("PYTHONPATH")
    if existing:
        path_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    return env


def _run_director_cli(
    *args: str,
    extra_pythonpath: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the production CLI module in a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "director_ai.cli", *args],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(extra_pythonpath=extra_pythonpath),
        text=True,
        capture_output=True,
        check=False,
    )


def _run_python_api(code: str) -> subprocess.CompletedProcess[str]:
    """Run a public Python API probe in a subprocess."""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
    )


def _json_from_stdout(stdout: str) -> JsonObject:
    """Parse the leading JSON object from CLI stdout."""
    payload = stdout.split("\n\nCommand:", maxsplit=1)[0]
    return cast(JsonObject, json.loads(payload))


def _as_object(value: object, label: str) -> JsonObject:
    """Return *value* as a JSON object with an assertion label."""
    assert isinstance(value, dict), label
    return cast(JsonObject, value)


def _as_list(value: object, label: str) -> list[object]:
    """Return *value* as a JSON list with an assertion label."""
    assert isinstance(value, list), label
    return value


def _as_str_list(value: object, label: str) -> list[str]:
    """Return *value* as a string list with an assertion label."""
    assert isinstance(value, list), label
    assert all(isinstance(item, str) for item in value), label
    return cast(list[str], value)


def _write_cloud_sdk_poison(module_dir: Path) -> Path:
    """Write a ``google.cloud.aiplatform`` module that fails on import."""
    cloud_dir = module_dir / "google" / "cloud"
    cloud_dir.mkdir(parents=True)
    (module_dir / "google" / "__init__.py").write_text("", encoding="utf-8")
    (cloud_dir / "__init__.py").write_text("", encoding="utf-8")
    (cloud_dir / "aiplatform.py").write_text(
        'raise RuntimeError("vertex dry-run imported google.cloud.aiplatform")\n',
        encoding="utf-8",
    )
    return module_dir


def test_managed_training_unit_guard_has_real_surface_companion() -> None:
    """Ensure the legacy managed-training guard is backed by this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_managed_training_jobs.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_managed_training_jobs_real_surface.py" in reason


def test_vertex_training_cli_dry_run_emits_cloud_job_without_sdk_import(
    tmp_path: Path,
) -> None:
    """Exercise Vertex managed training through the public CLI dry-run boundary."""
    poison_dir = _write_cloud_sdk_poison(tmp_path / "poison")

    completed = _run_director_cli(
        "train",
        "submit",
        "--backend",
        "vertex",
        "--dataset-uri",
        "gs://director-data/managed/train.jsonl",
        "--eval-uri",
        "gs://director-data/managed/eval.jsonl",
        "--output-uri",
        "gs://director-artifacts/jobs/job-1",
        "--project",
        "director-project",
        "--region",
        "europe-west4",
        "--image",
        "us-docker.pkg.dev/director/train:2026.07",
        "--machine",
        "n1-standard-8",
        "--gpu",
        "NVIDIA_T4",
        "--gpu-count",
        "1",
        "--boot-disk-gb",
        "120",
        "--timeout-min",
        "7",
        "--service-account",
        "trainer@director-project.iam.gserviceaccount.com",
        "--network",
        "projects/director-project/global/networks/private-training",
        "--dry-run",
        extra_pythonpath=poison_dir,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Command:" not in completed.stdout
    body = _json_from_stdout(completed.stdout)
    assert body["backend"] == "vertex"
    assert body["dry_run"] is True
    assert body["state"] == "dry_run"
    assert str(body["job_id"]).startswith(
        "projects/director-project/locations/europe-west4/customJobs/dry-run-"
    )

    request = _as_object(body["request"], "submission request")
    job_spec = _as_object(request["job_spec"], "vertex job_spec")
    assert job_spec["service_account"] == (
        "trainer@director-project.iam.gserviceaccount.com"
    )
    assert job_spec["network"] == (
        "projects/director-project/global/networks/private-training"
    )
    assert _as_object(job_spec["scheduling"], "scheduling")["timeout"] == "420s"
    labels = _as_object(request["labels"], "labels")
    assert labels["director-ai-caller"] == "product"
    assert labels["director-ai-task"] == "finetune-nli"

    pools = _as_list(job_spec["worker_pool_specs"], "worker pools")
    pool = _as_object(pools[0], "worker pool")
    machine = _as_object(pool["machine_spec"], "machine spec")
    assert machine == {
        "accelerator_count": 1,
        "accelerator_type": "NVIDIA_TESLA_T4",
        "machine_type": "n1-standard-8",
    }
    disk = _as_object(pool["disk_spec"], "disk spec")
    assert disk["boot_disk_size_gb"] == 120
    container = _as_object(pool["container_spec"], "container spec")
    assert container["image_uri"] == "us-docker.pkg.dev/director/train:2026.07"
    assert _as_str_list(container["command"], "container command") == ["python"]
    args = _as_str_list(container["args"], "container args")
    assert args[:3] == ["-m", "director_ai.core.training.vertex_runner", "--train-uri"]
    assert "gs://director-data/managed/eval.jsonl" in args
    assert "yaxili96/FactCG-DeBERTa-v3-Large" in args


def test_portable_training_public_api_redacts_env_and_emits_provenance() -> None:
    """Exercise the provider-neutral portable request through the public API."""
    completed = _run_python_api(
        """
from __future__ import annotations

from director_ai.core.training.jobs import (
    TrainingHardware,
    TrainingJobSpec,
    submission_to_json,
    submit_training_job,
)

spec = TrainingJobSpec(
    display_name="portable-contract",
    dataset_uri="s3://director-data/train.jsonl",
    eval_uri="azure://director-data/eval.jsonl",
    output_uri="file:///mnt/director/job-1",
    container_image_uri="registry.example.com/director-ai/train:2026.07",
    hardware=TrainingHardware(
        machine_type="a10g.xlarge",
        accelerator_type="NVIDIA_A10G",
        accelerator_count=1,
        boot_disk_gb=256,
    ),
    labels={"tenant": "legal"},
    env={"API_TOKEN": "secret-token", "MODE": "contract"},
)
submission = submit_training_job(spec, backend="portable", dry_run=True)
print(submission_to_json(submission))
""",
    )

    assert completed.returncode == 0, completed.stderr
    body = _json_from_stdout(completed.stdout)
    assert body["backend"] == "portable"
    assert body["dry_run"] is True
    assert str(body["job_id"]).startswith("portable-")

    request = _as_object(body["request"], "portable request")
    assert request["schema"] == "director-ai.portable-training-job.v1"
    assert request["display_name"] == "portable-contract"
    assert _as_object(request["inputs"], "inputs") == {
        "dataset_uri": "s3://director-data/train.jsonl",
        "eval_uri": "azure://director-data/eval.jsonl",
    }
    container = _as_object(request["container"], "container")
    assert container["image_uri"] == "registry.example.com/director-ai/train:2026.07"
    assert _as_object(container["env"], "container env") == {
        "API_TOKEN": "<redacted>",
        "MODE": "contract",
    }
    command = _as_str_list(container["command"], "portable command")
    args = _as_str_list(container["args"], "portable args")
    assert command == ["python"]
    assert args[:3] == ["-m", "director_ai.core.training.vertex_runner", "--train-uri"]
    assert "s3://director-data/train.jsonl" in args

    resources = _as_object(request["resources"], "resources")
    assert resources["machine_type"] == "a10g.xlarge"
    assert resources["accelerator_type"] == "NVIDIA_A10G"
    assert resources["boot_disk_gb"] == 256
    provenance = _as_object(request["provenance"], "provenance")
    assert len(str(provenance["dataset_hash"])) == 16
    assert len(str(provenance["config_hash"])) == 16
    model = _as_object(request["model"], "model")
    assert model["alias"] == "factcg-deberta-v3-large"


def test_local_training_cli_dry_run_reports_executable_command(
    tmp_path: Path,
) -> None:
    """Exercise local managed training through the CLI without running training."""
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "artifacts"
    train_path.write_text(
        '{"premise":"A contract is signed.","hypothesis":"A contract exists.","label":1}\n',
        encoding="utf-8",
    )
    eval_path.write_text(
        '{"premise":"A refund was denied.","hypothesis":"A refund happened.","label":0}\n',
        encoding="utf-8",
    )

    completed = _run_director_cli(
        "train",
        "submit",
        "--backend",
        "local",
        "--dataset-uri",
        str(train_path),
        "--eval-uri",
        str(eval_path),
        "--output-uri",
        str(output_dir),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--lr",
        "0.001",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    body = _json_from_stdout(completed.stdout)
    assert body["backend"] == "local"
    assert body["dry_run"] is True
    request = _as_object(body["request"], "local request")
    command = _as_str_list(request["command"], "local command")
    assert command[:4] == [
        "director-ai",
        "finetune",
        str(train_path),
        "--output",
    ]
    assert str(output_dir) in command
    assert str(eval_path) in command
    assert "\n\nCommand: director-ai finetune" in completed.stdout
