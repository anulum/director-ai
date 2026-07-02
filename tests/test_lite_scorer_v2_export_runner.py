# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 export runner tests
"""Unit guard for the Lite Scorer v2 guarded ONNX export runner."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

import tools.run_lite_scorer_v2_export as runner
from tools.run_lite_scorer_v2_export import (
    ExportError,
    find_export_argv,
    main,
    run_lite_scorer_v2_export,
)


def _patch_export_ready_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the training-status dependency to report export readiness."""
    monkeypatch.setattr(
        runner,
        "inspect_lite_scorer_v2_training",
        lambda *args, **kwargs: {
            "state": "completed",
            "export_ready": True,
            "run_dir": ".coordination/runs/DIRECTOR-AI/run",
        },
    )


def _patch_export_plan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact: str = "MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
    command: list[str] | None = None,
) -> None:
    """Patch the run-plan dependency with an export command and artifact path."""
    monkeypatch.setattr(
        runner,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {
                "plan_id": "lite-scorer-v2-run-plan",
                "commands": [
                    {
                        "name": "export_onnx",
                        "argv": command
                        or ["uv", "run", "--frozen", "director-ai", "export"],
                    }
                ],
                "generated_paths": {"onnx_artifact": artifact},
            },
            [],
        ),
    )


def test_find_export_argv_returns_export_onnx_command() -> None:
    """The runner should select the export_onnx command from the run plan."""
    plan = {
        "commands": [
            {"name": "train", "argv": ["train"]},
            {"name": "export_onnx", "argv": ["director-ai", "export"]},
        ]
    }

    assert find_export_argv(plan) == ["director-ai", "export"]


def test_find_export_argv_rejects_missing_command_list() -> None:
    """The runner should reject run plans without a command vector."""
    with pytest.raises(ExportError, match="run plan does not contain a commands list"):
        find_export_argv({})


def test_find_export_argv_rejects_invalid_export_argv() -> None:
    """The runner should reject malformed export command argv values."""
    plan = {"commands": [{"name": "export_onnx", "argv": ["director-ai", ""]}]}

    with pytest.raises(
        ExportError,
        match="run plan export_onnx command must be a non-empty argv list",
    ):
        find_export_argv(plan)


def test_find_export_argv_rejects_missing_export_command() -> None:
    """The runner should require an export_onnx command in the plan."""
    plan = {"commands": [{"name": "train", "argv": ["train"]}]}

    with pytest.raises(
        ExportError,
        match="run plan does not contain an export_onnx command",
    ):
        find_export_argv(plan)


def test_export_runner_refuses_when_training_is_not_export_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "inspect_lite_scorer_v2_training",
        lambda *args, **kwargs: {
            "state": "running",
            "export_ready": False,
            "run_dir": ".coordination/runs/DIRECTOR-AI/run",
        },
    )

    try:
        run_lite_scorer_v2_export(tmp_path)
    except ExportError as exc:
        assert str(exc) == (
            "Lite Scorer v2 training is not export-ready: state=running, "
            "export_ready=False"
        )
    else:
        raise AssertionError("expected ExportError")


def test_export_runner_uses_plan_command_and_verifies_onnx_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should execute the planned command and record the ONNX artefact."""
    calls: list[dict[str, object]] = []
    artifact = tmp_path / "MODELS" / "lite-scorer-v2" / "onnx" / "model_quantized.onnx"

    _patch_export_ready_status(monkeypatch)
    _patch_export_plan(monkeypatch)

    def fake_run(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, "cwd": cwd, "check": check})
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"onnx-bytes")
        return subprocess.CompletedProcess(argv, 0)

    result = run_lite_scorer_v2_export(tmp_path, run_command=fake_run)

    assert calls == [
        {
            "argv": ["uv", "run", "--frozen", "director-ai", "export"],
            "cwd": tmp_path.resolve(),
            "check": True,
        }
    ]
    assert result == {
        "schema_version": "1.0.0",
        "command": ["uv", "run", "--frozen", "director-ai", "export"],
        "onnx_artifact": {
            "path": "MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
            "exists": True,
            "size_bytes": len(b"onnx-bytes"),
        },
        "public_score_claim": False,
        "status": "recorded",
    }


def test_export_runner_rejects_plan_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should surface run-plan validation errors."""
    _patch_export_ready_status(monkeypatch)
    monkeypatch.setattr(
        runner,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: ({}, ["manifest is invalid"]),
    )

    with pytest.raises(ExportError, match="manifest is invalid"):
        run_lite_scorer_v2_export(tmp_path)


def test_export_runner_rejects_missing_generated_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should require generated_paths in the run plan."""
    _patch_export_ready_status(monkeypatch)
    monkeypatch.setattr(
        runner,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {
                "commands": [{"name": "export_onnx", "argv": ["export"]}],
            },
            [],
        ),
    )

    with pytest.raises(
        ExportError,
        match="run plan does not contain generated_paths",
    ):
        run_lite_scorer_v2_export(tmp_path)


def test_export_runner_rejects_missing_generated_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should require generated_paths.onnx_artifact in the plan."""
    _patch_export_ready_status(monkeypatch)
    monkeypatch.setattr(
        runner,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {
                "commands": [{"name": "export_onnx", "argv": ["export"]}],
                "generated_paths": {"onnx_artifact": ""},
            },
            [],
        ),
    )

    with pytest.raises(
        ExportError,
        match="run plan generated_paths.onnx_artifact must be a string",
    ):
        run_lite_scorer_v2_export(tmp_path)


def test_export_runner_rejects_artifact_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should keep the exported ONNX artefact inside the repository."""
    _patch_export_ready_status(monkeypatch)
    _patch_export_plan(monkeypatch, artifact="../model.onnx")

    with pytest.raises(
        ExportError,
        match="run plan ONNX artifact must stay inside the repository",
    ):
        run_lite_scorer_v2_export(tmp_path)


def test_export_runner_reports_command_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should wrap failed export commands as export errors."""
    _patch_export_ready_status(monkeypatch)
    _patch_export_plan(monkeypatch, command=["export"])

    def fail_run(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(7, argv)

    with pytest.raises(
        ExportError,
        match="ONNX export command failed with exit code 7",
    ):
        run_lite_scorer_v2_export(tmp_path, run_command=fail_run)


def test_export_runner_rejects_empty_onnx_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should not record a zero-byte ONNX artefact as completed."""
    artifact = tmp_path / "MODELS" / "lite-scorer-v2" / "onnx" / "model_quantized.onnx"
    _patch_export_ready_status(monkeypatch)
    _patch_export_plan(monkeypatch)

    def write_empty_artifact(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"")
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(
        ExportError,
        match="ONNX export command completed but .* is empty",
    ):
        run_lite_scorer_v2_export(tmp_path, run_command=write_empty_artifact)


def test_export_runner_fails_when_command_does_not_create_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should reject successful commands that leave no ONNX file."""
    _patch_export_ready_status(monkeypatch)
    _patch_export_plan(monkeypatch, artifact="MODELS/missing.onnx", command=["export"])

    try:
        run_lite_scorer_v2_export(
            tmp_path,
            run_command=lambda argv, *, cwd, check: subprocess.CompletedProcess(
                argv,
                0,
            ),
        )
    except ExportError as exc:
        assert str(exc) == (
            "ONNX export command completed but MODELS/missing.onnx is missing"
        )
    else:
        raise AssertionError("expected ExportError")


def test_export_runner_main_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint should print a recorded export receipt."""
    monkeypatch.setattr(
        runner,
        "run_lite_scorer_v2_export",
        lambda root, *, manifest, run_dir: {
            "schema_version": "1.0.0",
            "status": "recorded",
            "command": ["export"],
            "onnx_artifact": {
                "path": "MODELS/model.onnx",
                "exists": True,
                "size_bytes": 4,
            },
            "public_score_claim": False,
        },
    )

    assert main([str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert '"status": "recorded"' in captured.out
    assert captured.err == ""


def test_export_runner_main_reports_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint should return non-zero and print export errors."""

    def fail_export(root: Path, *, manifest: Path, run_dir: Path | None) -> object:
        raise ExportError("export blocked")

    monkeypatch.setattr(runner, "run_lite_scorer_v2_export", fail_export)

    assert main([str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "export blocked\n"


def test_export_runner_load_tool_reports_unloadable_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool loader should fail closed when importlib cannot build a spec."""
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda name, path: None,
    )

    with pytest.raises(RuntimeError, match="cannot load missing"):
        runner._load_tool("missing", "missing.py")
