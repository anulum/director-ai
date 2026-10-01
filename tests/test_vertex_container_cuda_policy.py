# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vertex container CUDA policy tests

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def test_vertex_benchmark_container_meets_project_nli_requirements() -> None:
    """Training locks must satisfy the installed project's NLI constraints."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    requirements = project["project"]["optional-dependencies"]["nli"]
    for filename in ("requirements-distil.txt", "requirements-contradiction.txt"):
        locked = {
            match[0]: Version(match[1])
            for match in re.findall(
                r"^([a-zA-Z0-9_-]+)==([^ ;\\]+)",
                (ROOT / "training" / filename).read_text(),
                flags=re.MULTILINE,
            )
        }
        for raw in requirements:
            requirement = Requirement(raw)
            assert locked[requirement.name] in requirement.specifier


def test_vertex_lite_scorer_container_uses_digest_pinned_clean_base() -> None:
    """All managed recipes resolve CUDA through the hashed Python closure."""
    for name in ("benchmarks", "lite_scorer_v2", "distil", "contradiction"):
        text = (ROOT / "training" / f"Dockerfile.{name}").read_text()
        images = re.findall(r"^FROM (\S+)", text, flags=re.MULTILINE)
        assert images
        assert all(re.search(r"@sha256:[a-f0-9]{64}$", image) for image in images)
        assert images[-1].startswith("python:3.12-slim@")
        assert "--force-reinstall" not in text
        assert "requirements-cuda121-torch.txt" not in text
        if name in {"benchmarks", "lite_scorer_v2"}:
            assert "DIRECTOR_REQUIRE_CUDA=1" in text


def test_vertex_containers_install_hash_pinned_requirement_files() -> None:
    """External installs must use hashed locks without bypassing resolution."""
    for name in ("benchmarks", "lite_scorer_v2", "distil", "contradiction"):
        text = (ROOT / "training" / f"Dockerfile.{name}").read_text()
        commands = text.replace("\\\n", " ").splitlines()
        for command in commands:
            if command.startswith("RUN pip install") and "-r " in command:
                assert "--require-hashes" in command
                for path in re.findall(r"-r ([^ ]+)", command):
                    assert "--hash=sha256:" in (ROOT / path).read_text()
        assert "pip check" in text
        if "-e ." in text:
            assert "--no-deps --no-build-isolation -e ." in text
            assert "requirements/docker-build.txt" in text


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
