# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex container CUDA policy tests

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vertex_benchmark_container_overrides_to_vertex_compatible_cuda() -> None:
    text = (ROOT / "training" / "Dockerfile.benchmarks").read_text()
    torch_lock = (ROOT / "training" / "requirements-cuda121-torch.txt").read_text()

    assert "requirements-cuda121-torch.txt" in text
    assert "torch-2.5.1%2Bcu121" in torch_lock
    assert "download-r2.pytorch.org/whl/cu121" in torch_lock
    assert "DIRECTOR_REQUIRE_CUDA=1" in text


def test_vertex_lite_scorer_container_overrides_to_vertex_compatible_cuda() -> None:
    text = (ROOT / "training" / "Dockerfile.lite_scorer_v2").read_text()
    torch_lock = (ROOT / "training" / "requirements-cuda121-torch.txt").read_text()

    assert "requirements-cuda121-torch.txt" in text
    assert "torch-2.5.1%2Bcu121" in torch_lock
    assert "download-r2.pytorch.org/whl/cu121" in torch_lock
    assert "DIRECTOR_REQUIRE_CUDA=1" in text


def test_vertex_cuda_override_uses_hash_pinned_requirement_files() -> None:
    benchmark_text = (ROOT / "training" / "Dockerfile.benchmarks").read_text()
    lite_text = (ROOT / "training" / "Dockerfile.lite_scorer_v2").read_text()
    torch_lock = (ROOT / "training" / "requirements-cuda121-torch.txt").read_text()
    benchmark_tools_lock = (
        ROOT / "training" / "requirements-benchmark-tools.txt"
    ).read_text()

    assert "requirements-cuda121-torch.txt" in benchmark_text
    assert "requirements-cuda121-torch.txt" in lite_text
    assert "requirements-benchmark-tools.txt" in benchmark_text
    assert "torch-2.5.1%2Bcu121" in torch_lock
    assert (
        "sha256=c8ab8c92eab928a93c483f83ca8c63f13dafc10fc93ad90ed2dcb7c82ea50410"
        in torch_lock
    )
    assert (
        "sha256=222be02548c2e74a21a8fbc8e5b8d2eef9f9faee865d70385d2eb1b9aabcbc76"
        in torch_lock
    )
    assert "--hash=sha256:" in benchmark_tools_lock


def test_vertex_benchmark_entrypoint_fails_fast_without_cuda() -> None:
    text = (ROOT / "benchmarks" / "run_in_container.sh").read_text()

    assert "DIRECTOR_REQUIRE_CUDA" in text
    assert "torch.cuda.is_available()" in text
    assert 'torch.ones(1, device="cuda")' in text


def test_vertex_entrypoint_supports_model_package_campaign() -> None:
    text = (ROOT / "benchmarks" / "run_in_container.sh").read_text()

    assert "DIRECTOR_MODEL_PACKAGE_CAMPAIGN" in text
    assert "benchmarks.model_package_campaign" in text
    assert "DIRECTOR_MODEL_PACKAGE_NO_UPLOAD" in text
    assert "--min-free-gb" in text
    assert "--no-upload" in text


def test_vertex_entrypoint_does_not_upload_package_campaign_twice() -> None:
    text = (ROOT / "benchmarks" / "run_in_container.sh").read_text()

    assert "DIRECTOR_OUTPUT_ALREADY_UPLOADED=1" in text
    assert 'DIRECTOR_OUTPUT_ALREADY_UPLOADED:-0}" != "1"' in text


def test_model_package_campaign_submitter_uses_large_disk_and_provenance() -> None:
    text = (ROOT / "benchmarks" / "run_vertex_model_package_campaign.sh").read_text()

    assert "--dry-run" in text
    assert "--config-out" in text
    assert 'BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-500}"' in text
    assert "BOOT_DISK_SIZE_GB must be at least 500" in text
    assert "MIN_FREE_GB must be at least 25" in text
    assert "DIRECTOR_MODEL_PACKAGE_CAMPAIGN" in text
    assert "DIRECTOR_REQUIRE_CUDA" in text
    assert "DIRECTOR_GIT_COMMIT" in text
    assert "DIRECTOR_GIT_BRANCH" in text
    assert "boot-disk-size" in text


def test_model_package_campaign_submitter_dry_run_writes_vertex_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "custom-job.json"

    completed = subprocess.run(
        [
            "bash",
            "benchmarks/run_vertex_model_package_campaign.sh",
            "--dry-run",
            "--config-out",
            str(config_path),
            "--skip-build",
            "--model-aliases",
            "balanced-default,deberta-small",
            "--stage-ids",
            "aggrefact_anchor_vertex,ragtruth_vertex",
            "--suffix",
            "unit",
        ],
        cwd=ROOT,
        text=True,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    config = json.loads(config_path.read_text(encoding="utf-8"))
    worker = config["workerPoolSpecs"][0]
    env = {item["name"]: item["value"] for item in worker["containerSpec"]["env"]}
    assert worker["diskSpec"]["bootDiskSizeGb"] == 500
    assert env["DIRECTOR_MODEL_PACKAGE_CAMPAIGN"] == "1"
    assert env["DIRECTOR_MODEL_PACKAGE_ALIASES"] == "balanced-default,deberta-small"
    assert env["DIRECTOR_MODEL_PACKAGE_STAGE_IDS"] == (
        "aggrefact_anchor_vertex,ragtruth_vertex"
    )


def test_local_model_package_campaign_runner_is_no_upload_by_default() -> None:
    text = (ROOT / "benchmarks" / "run_model_package_campaign.sh").read_text()

    assert "benchmarks.model_package_campaign" in text
    assert "--no-upload" in text
    assert "--upload-uri" in text
    assert "UPLOAD_URI" in text
    assert "DIRECTOR_GIT_COMMIT" in text
    assert "DIRECTOR_GIT_BRANCH" in text
    assert "--dry-run" in text


def test_local_model_package_campaign_dry_run_prints_command(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            "bash",
            "benchmarks/run_model_package_campaign.sh",
            "--dry-run",
            "--output-root",
            str(tmp_path / "campaign"),
            "--model-aliases",
            "balanced-default",
            "--stage-ids",
            "aggrefact_anchor_vertex",
        ],
        cwd=ROOT,
        text=True,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "benchmarks.model_package_campaign" in completed.stdout
    assert "--no-upload" in completed.stdout
    assert "balanced-default" in completed.stdout


def test_local_model_package_campaign_dry_run_accepts_upload_uri(
    tmp_path: Path,
) -> None:
    upload_root = tmp_path / "uploaded"
    completed = subprocess.run(
        [
            "bash",
            "benchmarks/run_model_package_campaign.sh",
            "--dry-run",
            "--output-root",
            str(tmp_path / "campaign"),
            "--upload-uri",
            f"file://{upload_root}",
            "--prefix",
            "provider/run",
            "--model-aliases",
            "balanced-default",
        ],
        cwd=ROOT,
        text=True,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--upload-uri" in completed.stdout
    assert "--prefix provider/run" in completed.stdout
    assert "--no-upload" not in completed.stdout
