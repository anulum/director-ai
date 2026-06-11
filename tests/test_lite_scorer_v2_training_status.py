# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 training status tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "tools" / "status_lite_scorer_v2_training.py"
SPEC = importlib.util.spec_from_file_location("status_lite_scorer_v2_training", STATUS)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

inspect_lite_scorer_v2_training = MODULE.inspect_lite_scorer_v2_training


def _write_manifest(root: Path) -> Path:
    manifest = root / "benchmarks" / "lite_scorer_v2_run_manifest.toml"
    manifest.parent.mkdir(parents=True)
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


def _write_run(root: Path, name: str, *, pid: int = 101, log: str = "") -> Path:
    run_dir = root / ".coordination" / "runs" / "DIRECTOR-AI" / name
    run_dir.mkdir(parents=True)
    (run_dir / "pid").write_text(f"{pid}\n", encoding="utf-8")
    (run_dir / "train.log").write_text(log, encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_id": name, "public_score_claim": False}) + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_training_status_reports_running_process_without_export_readiness(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    run_dir = _write_run(
        tmp_path,
        "lite_scorer_v2_train_2026-05-18T030000",
        log="START 2026-05-18T03:00:00+02:00\nINFO: Device: cpu\n",
    )

    status = inspect_lite_scorer_v2_training(
        tmp_path,
        manifest=manifest,
        run_dir=run_dir,
        process_running=lambda pid: pid == 101,
    )

    assert status["state"] == "running"
    assert status["process_running"] is True
    assert status["exit_code"] is None
    assert status["export_ready"] is False
    assert status["public_score_claim"] is False
    assert status["artefacts"]["student_artifact"]["exists"] is False


def test_training_status_requires_exit_zero_and_student_artifact_for_export(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    run_dir = _write_run(
        tmp_path,
        "lite_scorer_v2_train_2026-05-18T031000",
        log="INFO: Final: BA=0.5000\nEXIT 0 2026-05-18T03:10:00+02:00\n",
    )
    artifact = tmp_path / "MODELS" / "lite-scorer-v2" / "student" / "model.safetensors"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"real-model-bytes")

    status = inspect_lite_scorer_v2_training(
        tmp_path,
        manifest=manifest,
        run_dir=run_dir,
        process_running=lambda _pid: False,
    )

    assert status["state"] == "completed"
    assert status["exit_code"] == 0
    assert status["export_ready"] is True
    assert status["artefacts"]["student_artifact"] == {
        "path": "MODELS/lite-scorer-v2/student/model.safetensors",
        "exists": True,
        "size_bytes": len(b"real-model-bytes"),
    }


def test_training_status_reports_failed_exit_and_blocks_export(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    run_dir = _write_run(
        tmp_path,
        "lite_scorer_v2_train_2026-05-18T032000",
        log="Traceback omitted\nEXIT 2 2026-05-18T03:20:00+02:00\n",
    )

    status = inspect_lite_scorer_v2_training(
        tmp_path,
        manifest=manifest,
        run_dir=run_dir,
        process_running=lambda _pid: False,
    )

    assert status["state"] == "failed"
    assert status["exit_code"] == 2
    assert status["export_ready"] is False


def test_training_status_reports_stale_when_pid_gone_without_exit(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    run_dir = _write_run(
        tmp_path,
        "lite_scorer_v2_train_2026-05-18T033000",
        log="START 2026-05-18T03:30:00+02:00\n",
    )

    status = inspect_lite_scorer_v2_training(
        tmp_path,
        manifest=manifest,
        run_dir=run_dir,
        process_running=lambda _pid: False,
    )

    assert status["state"] == "stale"
    assert status["exit_code"] is None
    assert status["export_ready"] is False
