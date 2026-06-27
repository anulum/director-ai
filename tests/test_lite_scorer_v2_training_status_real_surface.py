# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 training status real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 training-status CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "tools" / "status_lite_scorer_v2_training.py"


def _write_manifest(root: Path) -> Path:
    """Write the minimal run manifest consumed by the production status CLI."""
    manifest = root / "benchmarks" / "lite_scorer_v2_run_manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "\n".join(
            [
                'train_output_dir = "MODELS/lite-scorer-v2/student"',
                'student_artifact = "MODELS/lite-scorer-v2/student/model.safetensors"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_run_dir(
    root: Path,
    name: str,
    *,
    log: str,
    pid: int | None = None,
) -> Path:
    """Write a local training run directory under the repository run root."""
    run_dir = root / ".coordination" / "runs" / "DIRECTOR-AI" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    if pid is not None:
        (run_dir / "pid").write_text(f"{pid}\n", encoding="utf-8")
    (run_dir / "train.log").write_text(log, encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_id": name, "public_score_claim": False}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _write_completed_artifacts(root: Path) -> None:
    """Write the student artefact and training manifest expected after success."""
    output_dir = root / "MODELS" / "lite-scorer-v2" / "student"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model.safetensors").write_bytes(b"student model bytes\n")
    (output_dir / "training_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "public_score_claim": False,
                "seed": 20260518,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _load_status(stdout: str) -> Mapping[str, object]:
    """Decode the JSON object emitted by the production status CLI."""
    payload = json.loads(stdout)
    assert isinstance(payload, dict)
    return cast(Mapping[str, object], payload)


def _artefacts(status: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    """Return typed artefact status mappings from a decoded CLI payload."""
    value = status["artefacts"]
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    assert all(isinstance(item, dict) for item in value.values())
    return cast(Mapping[str, Mapping[str, object]], value)


def test_lite_scorer_v2_training_status_cli_reports_completed_export_ready(
    tmp_path: Path,
) -> None:
    """The production CLI should report completed runs as export-ready."""
    manifest = _write_manifest(tmp_path)
    run_dir = _write_run_dir(
        tmp_path,
        "lite_scorer_v2_train_2026-06-27T140000",
        log="INFO: Final: BA=0.8125\nEXIT 0 2026-06-27T14:00:00+02:00\n",
    )
    _write_completed_artifacts(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(STATUS),
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--run-dir",
            str(run_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    status = _load_status(result.stdout)
    assert status["state"] == "completed"
    assert status["exit_code"] == 0
    assert status["export_ready"] is True
    assert status["public_score_claim"] is False
    artefacts = _artefacts(status)
    assert artefacts["student_artifact"] == {
        "path": "MODELS/lite-scorer-v2/student/model.safetensors",
        "exists": True,
        "size_bytes": len(b"student model bytes\n"),
    }
    assert artefacts["training_run_manifest"] == {
        "path": "MODELS/lite-scorer-v2/student/training_run_manifest.json",
        "exists": True,
        "size_bytes": len(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "public_score_claim": False,
                    "seed": 20260518,
                }
            )
            + "\n"
        ),
    }


def test_lite_scorer_v2_training_status_cli_reports_live_process_as_running(
    tmp_path: Path,
) -> None:
    """The production CLI should identify an alive recorded PID as running."""
    manifest = _write_manifest(tmp_path)
    run_dir = _write_run_dir(
        tmp_path,
        "lite_scorer_v2_train_2026-06-27T141000",
        log="START 2026-06-27T14:10:00+02:00\n",
        pid=os.getpid(),
    )

    result = subprocess.run(
        [
            sys.executable,
            str(STATUS),
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--run-dir",
            str(run_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    status = _load_status(result.stdout)
    assert status["state"] == "running"
    assert status["process_running"] is True
    assert status["export_ready"] is False


def test_lite_scorer_v2_training_status_cli_rejects_missing_manifest(
    tmp_path: Path,
) -> None:
    """The production CLI should fail closed when the manifest is absent."""
    run_dir = _write_run_dir(
        tmp_path,
        "lite_scorer_v2_train_2026-06-27T142000",
        log="START 2026-06-27T14:20:00+02:00\n",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(STATUS),
            str(tmp_path),
            "--run-dir",
            str(run_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "benchmarks/lite_scorer_v2_run_manifest.toml: missing manifest" in (
        result.stderr
    )
