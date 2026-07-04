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
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "tools" / "status_lite_scorer_v2_training.py"
SPEC = importlib.util.spec_from_file_location("status_lite_scorer_v2_training", STATUS)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

inspect_lite_scorer_v2_training = MODULE.inspect_lite_scorer_v2_training
resolve_lite_scorer_v2_run_root = MODULE.resolve_lite_scorer_v2_run_root
StatusError = MODULE.StatusError


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


def test_training_status_process_probe_handles_kernel_outcomes(
    monkeypatch: Any,
) -> None:
    """Process probing should handle missing, denied, and visible PIDs."""
    assert MODULE.is_process_running(0) is False

    def missing_process(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(MODULE.os, "kill", missing_process)
    assert MODULE.is_process_running(101) is False

    def denied_process(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(MODULE.os, "kill", denied_process)
    assert MODULE.is_process_running(101) is True

    def visible_process(_pid: int, _signal: int) -> None:
        return None

    monkeypatch.setattr(MODULE.os, "kill", visible_process)
    assert MODULE.is_process_running(101) is True


def test_training_status_run_root_falls_back_to_local_coordination(
    tmp_path: Path,
) -> None:
    """Non-code-repo roots should keep durable state under the provided root."""
    assert resolve_lite_scorer_v2_run_root(tmp_path) == (
        tmp_path.resolve() / ".coordination" / "runs" / "DIRECTOR-AI"
    )


def test_training_status_code_repo_without_monorepo_coordination_falls_back(
    tmp_path: Path,
) -> None:
    """Code-repo-shaped temp roots without monorepo state should stay local."""
    repo = tmp_path / "aaa_God_of_the_Math_Collection" / "03_CODE" / "DIRECTOR-AI"
    repo.mkdir(parents=True)

    assert resolve_lite_scorer_v2_run_root(repo) == (
        repo.resolve() / ".coordination" / "runs" / "DIRECTOR-AI"
    )


def test_training_status_rejects_missing_latest_run(tmp_path: Path) -> None:
    """Latest-run discovery should fail clearly when no run exists."""
    try:
        MODULE._latest_run_dir(tmp_path)
    except StatusError as exc:
        assert "no Lite Scorer v2 training runs found under" in str(exc)
    else:
        raise AssertionError("expected StatusError")


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


def test_training_status_accepts_relative_run_dir(tmp_path: Path) -> None:
    """Relative run-dir arguments should resolve inside the repository root."""
    manifest = _write_manifest(tmp_path)
    run_dir = _write_run(
        tmp_path,
        "lite_scorer_v2_train_2026-07-04T130000",
        log="EXIT 0 2026-07-04T13:00:00+02:00\n",
    )
    relative_run_dir = run_dir.relative_to(tmp_path)

    status = inspect_lite_scorer_v2_training(
        tmp_path,
        manifest=manifest,
        run_dir=relative_run_dir,
        process_running=lambda _pid: False,
    )

    assert status["run_dir"] == relative_run_dir.as_posix()
    assert status["state"] == "completed"
    assert status["export_ready"] is False


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


def test_training_status_reads_missing_pid_as_not_running(tmp_path: Path) -> None:
    """Missing pid files should not make a completed run look live."""
    manifest = _write_manifest(tmp_path)
    run_dir = (
        tmp_path / ".coordination" / "runs" / "DIRECTOR-AI" / "lite_scorer_v2_train"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text(
        "EXIT 0 2026-07-04T13:30:00+02:00\n",
        encoding="utf-8",
    )

    status = inspect_lite_scorer_v2_training(
        tmp_path,
        manifest=manifest,
        run_dir=run_dir,
        process_running=lambda _pid: True,
    )

    assert status["pid"] is None
    assert status["process_running"] is False
    assert status["state"] == "completed"


def test_training_status_rejects_invalid_pid(tmp_path: Path) -> None:
    """Invalid pid files should fail closed instead of being ignored."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "pid").write_text("not-a-pid\n", encoding="utf-8")

    try:
        MODULE._read_pid(run_dir)
    except StatusError as exc:
        assert str(exc) == f"{run_dir / 'pid'}: invalid pid"
    else:
        raise AssertionError("expected StatusError")


def test_training_status_rejects_invalid_manifest_and_fields(tmp_path: Path) -> None:
    """Invalid TOML and unsafe manifest fields should fail closed."""
    missing_manifest = tmp_path / "missing.toml"
    try:
        MODULE._read_manifest(tmp_path, missing_manifest)
    except StatusError as exc:
        assert str(exc) == f"{missing_manifest}: missing manifest"
    else:
        raise AssertionError("expected StatusError")

    manifest = tmp_path / "manifest.toml"
    manifest.write_text("[broken\n", encoding="utf-8")

    try:
        MODULE._read_manifest(tmp_path, manifest)
    except StatusError as exc:
        assert f"{manifest}: invalid TOML:" in str(exc)
    else:
        raise AssertionError("expected StatusError")

    for data, message in [
        ({}, "manifest field student_artifact must be a non-empty string"),
        (
            {"student_artifact": "/tmp/model.safetensors"},
            "manifest field student_artifact must stay inside the repository",
        ),
        (
            {"student_artifact": "../model.safetensors"},
            "manifest field student_artifact must stay inside the repository",
        ),
    ]:
        try:
            MODULE._string_field(data, "student_artifact")
        except StatusError as exc:
            assert str(exc) == message
        else:
            raise AssertionError("expected StatusError")


def test_training_status_exit_marker_handles_absent_and_noisy_logs(
    tmp_path: Path,
) -> None:
    """Exit-marker parsing should tolerate absent logs and non-marker lines."""
    missing_log = tmp_path / "missing.log"
    assert MODULE._exit_marker(missing_log) == (None, None)

    noisy_log = tmp_path / "train.log"
    noisy_log.write_text("START now\nnot an exit marker\n", encoding="utf-8")
    assert MODULE._exit_marker(noisy_log) == (None, None)


def test_training_status_latest_run_uses_monorepo_coordination_for_code_repo(
    tmp_path: Path,
) -> None:
    """Status inspection should read real code-repo runs from monorepo state."""
    monorepo = tmp_path / "aaa_God_of_the_Math_Collection"
    repo = monorepo / "03_CODE" / "DIRECTOR-AI"
    repo.mkdir(parents=True)
    coordination = monorepo / ".coordination"
    coordination.mkdir()
    manifest = _write_manifest(repo)
    run_dir = (
        coordination / "runs" / "DIRECTOR-AI" / "lite_scorer_v2_train_2026-07-04T121500"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "pid").write_text("2026\n", encoding="utf-8")
    (run_dir / "train.log").write_text(
        "EXIT 0 2026-07-04T12:15:00+02:00\n",
        encoding="utf-8",
    )
    artifact = repo / "MODELS" / "lite-scorer-v2" / "student" / "model.safetensors"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"student-model")

    status = inspect_lite_scorer_v2_training(
        repo,
        manifest=manifest,
        process_running=lambda _pid: False,
    )

    assert (
        resolve_lite_scorer_v2_run_root(repo) == coordination / "runs" / "DIRECTOR-AI"
    )
    assert status["run_dir"] == run_dir.as_posix()
    assert status["state"] == "completed"
    assert status["export_ready"] is True


def test_training_status_main_prints_json_on_success(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The CLI entrypoint should emit status as JSON."""

    def fake_inspect(
        root: Path,
        *,
        manifest: Path,
        run_dir: Path | None,
    ) -> dict[str, object]:
        assert root == tmp_path
        assert manifest == tmp_path / "manifest.toml"
        assert run_dir == tmp_path / "run"
        return {"state": "completed", "export_ready": True}

    monkeypatch.setattr(MODULE, "inspect_lite_scorer_v2_training", fake_inspect)
    stdout = StringIO()

    with redirect_stdout(stdout):
        exit_code = MODULE.main(
            [
                str(tmp_path),
                "--manifest",
                str(tmp_path / "manifest.toml"),
                "--run-dir",
                str(tmp_path / "run"),
            ]
        )

    assert exit_code == 0
    assert json.loads(stdout.getvalue()) == {"state": "completed", "export_ready": True}


def test_training_status_main_reports_status_errors(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """The CLI entrypoint should write status failures to stderr."""

    def fake_inspect(
        _root: Path,
        *,
        manifest: Path,
        run_dir: Path | None,
    ) -> dict[str, object]:
        assert manifest == MODULE.MANIFEST
        assert run_dir is None
        raise StatusError("status rejected")

    monkeypatch.setattr(MODULE, "inspect_lite_scorer_v2_training", fake_inspect)
    stderr = StringIO()

    with redirect_stderr(stderr):
        exit_code = MODULE.main([str(tmp_path)])

    assert exit_code == 1
    assert stderr.getvalue() == "status rejected\n"
