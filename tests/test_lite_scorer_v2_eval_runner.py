# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evaluation runner tests
"""Unit guard for the Lite Scorer v2 guarded evaluation runner."""

from __future__ import annotations

import importlib.machinery
import importlib.util as importlib_util
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

import tools.run_lite_scorer_v2_eval as runner
from tools.run_lite_scorer_v2_eval import (
    EvalError,
    find_evaluate_argv,
    main,
    run_lite_scorer_v2_eval,
)


def _patch_eval_plan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result_path: str = "benchmarks/results/lite_scorer_v2_eval.json",
    onnx_path: str = "MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
    command: list[str] | None = None,
) -> None:
    """Patch the run-plan dependency with an evaluate command and paths."""
    monkeypatch.setattr(
        runner,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {
                "plan_id": "lite-scorer-v2-run-plan",
                "commands": [
                    {
                        "name": "evaluate",
                        "argv": command
                        or [
                            "uv",
                            "run",
                            "--frozen",
                            "python",
                            "tools/eval_lite_scorer_v2.py",
                        ],
                    }
                ],
                "generated_paths": {
                    "eval_result": result_path,
                    "onnx_artifact": onnx_path,
                },
            },
            [],
        ),
    )


def _write_onnx_artifact(root: Path, *, payload: bytes = b"onnx") -> Path:
    """Write the ONNX artefact expected by the evaluation runner."""
    onnx = root / "MODELS" / "lite-scorer-v2" / "onnx" / "model_quantized.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(payload)
    return onnx


def _write_eval_result(
    path: Path,
    payload: Mapping[str, object] | None = None,
) -> None:
    """Write evaluator JSON with valid defaults and optional overrides."""
    result: dict[str, object] = {
        "heldout_eval_balanced_accuracy": 0.71,
        "heldout_eval_rows": 1000,
        "heldout_eval_threshold": 0.53,
        "latency_p50_ms": 2.4,
        "latency_p95_ms": 4.8,
        "latency_sample_count": 100,
    }
    if payload is not None:
        result.update(payload)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(result) + "\n", encoding="utf-8")


def test_load_tool_rejects_missing_import_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dependency loader should reject tools without an import spec."""
    monkeypatch.setattr(
        importlib_util,
        "spec_from_file_location",
        lambda name, path: None,
    )

    with pytest.raises(RuntimeError, match="cannot load missing_tool from"):
        runner._load_tool("missing_tool", "missing_tool.py")


def test_load_tool_rejects_import_spec_without_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dependency loader should reject tool specs without loaders."""
    monkeypatch.setattr(
        importlib_util,
        "spec_from_file_location",
        lambda name, path: importlib.machinery.ModuleSpec(name, loader=None),
    )

    with pytest.raises(RuntimeError, match="cannot load missing_tool from"):
        runner._load_tool("missing_tool", "missing_tool.py")


def test_find_evaluate_argv_returns_planned_evaluate_command() -> None:
    """The runner should select the evaluate command from the run plan."""
    plan = {
        "commands": [
            {"name": "export_onnx", "argv": ["export"]},
            {"name": "evaluate", "argv": ["python", "tools/eval_lite_scorer_v2.py"]},
        ]
    }

    assert find_evaluate_argv(plan) == ["python", "tools/eval_lite_scorer_v2.py"]


def test_find_evaluate_argv_rejects_missing_command_list() -> None:
    """The runner should reject run plans without a command vector."""
    with pytest.raises(EvalError, match="run plan does not contain a commands list"):
        find_evaluate_argv({})


def test_find_evaluate_argv_rejects_invalid_evaluate_argv() -> None:
    """The runner should reject malformed evaluate command argv values."""
    plan = {"commands": [{"name": "evaluate", "argv": ["python", ""]}]}

    with pytest.raises(
        EvalError,
        match="run plan evaluate command must be a non-empty argv list",
    ):
        find_evaluate_argv(plan)


def test_find_evaluate_argv_rejects_missing_evaluate_command() -> None:
    """The runner should require an evaluate command in the run plan."""
    plan = {"commands": [{"name": "export_onnx", "argv": ["export"]}]}

    with pytest.raises(
        EvalError,
        match="run plan does not contain an evaluate command",
    ):
        find_evaluate_argv(plan)


def test_eval_runner_refuses_missing_onnx_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should fail before evaluation when the ONNX artefact is absent."""
    _patch_eval_plan(monkeypatch)

    with pytest.raises(
        EvalError,
        match=(
            "Lite Scorer v2 ONNX artifact is missing: "
            "MODELS/lite-scorer-v2/onnx/model_quantized.onnx"
        ),
    ):
        run_lite_scorer_v2_eval(tmp_path)


def test_eval_runner_rejects_empty_onnx_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should not evaluate a zero-byte ONNX artefact."""
    _write_onnx_artifact(tmp_path, payload=b"")
    _patch_eval_plan(monkeypatch)

    with pytest.raises(
        EvalError,
        match="Lite Scorer v2 ONNX artifact is empty: .*model_quantized.onnx",
    ):
        run_lite_scorer_v2_eval(tmp_path)


def test_eval_runner_uses_plan_command_and_validates_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should execute the planned command and record result status."""
    calls: list[dict[str, object]] = []
    _write_onnx_artifact(tmp_path)
    result_path = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _patch_eval_plan(monkeypatch)

    def fake_run(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, "cwd": cwd, "check": check})
        _write_eval_result(result_path)
        return subprocess.CompletedProcess(argv, 0)

    result = run_lite_scorer_v2_eval(tmp_path, run_command=fake_run)

    assert calls == [
        {
            "argv": [
                "uv",
                "run",
                "--frozen",
                "python",
                "tools/eval_lite_scorer_v2.py",
            ],
            "cwd": tmp_path.resolve(),
            "check": True,
        }
    ]
    assert result == {
        "schema_version": "1.0.0",
        "status": "recorded",
        "command": [
            "uv",
            "run",
            "--frozen",
            "python",
            "tools/eval_lite_scorer_v2.py",
        ],
        "eval_result": {
            "path": "benchmarks/results/lite_scorer_v2_eval.json",
            "exists": True,
            "size_bytes": result_path.stat().st_size,
        },
        "public_score_claim": False,
    }


def test_eval_runner_rejects_plan_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should surface run-plan validation errors."""
    monkeypatch.setattr(
        runner,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: ({}, ["manifest is invalid"]),
    )

    with pytest.raises(EvalError, match="manifest is invalid"):
        run_lite_scorer_v2_eval(tmp_path)


def test_eval_runner_rejects_missing_generated_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should require generated_paths in the run plan."""
    _write_onnx_artifact(tmp_path)
    monkeypatch.setattr(
        runner,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {"commands": [{"name": "evaluate", "argv": ["evaluate"]}]},
            [],
        ),
    )

    with pytest.raises(
        EvalError,
        match="run plan does not contain generated_paths",
    ):
        run_lite_scorer_v2_eval(tmp_path)


def test_eval_runner_rejects_missing_generated_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should require generated_paths.eval_result in the plan."""
    _write_onnx_artifact(tmp_path)
    _patch_eval_plan(monkeypatch, result_path="")

    with pytest.raises(
        EvalError,
        match="run plan generated_paths.eval_result must be a string",
    ):
        run_lite_scorer_v2_eval(tmp_path)


def test_eval_runner_rejects_generated_path_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should keep generated paths inside the repository."""
    _write_onnx_artifact(tmp_path)
    _patch_eval_plan(monkeypatch, result_path="../lite_scorer_v2_eval.json")

    with pytest.raises(
        EvalError,
        match="run plan generated_paths.eval_result must stay inside the repository",
    ):
        run_lite_scorer_v2_eval(tmp_path)


def test_eval_runner_reports_command_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should wrap failed evaluation commands as eval errors."""
    _write_onnx_artifact(tmp_path)
    _patch_eval_plan(monkeypatch, command=["evaluate"])

    def fail_run(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(7, argv)

    with pytest.raises(
        EvalError,
        match="Lite Scorer v2 evaluation failed with exit code 7",
    ):
        run_lite_scorer_v2_eval(tmp_path, run_command=fail_run)


def test_eval_runner_fails_when_command_does_not_create_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should reject successful commands that leave no result file."""
    _write_onnx_artifact(tmp_path)
    _patch_eval_plan(monkeypatch, command=["evaluate"])

    with pytest.raises(
        EvalError,
        match=(
            "evaluation command completed but "
            "benchmarks/results/lite_scorer_v2_eval.json is missing"
        ),
    ):
        run_lite_scorer_v2_eval(
            tmp_path,
            run_command=lambda argv, *, cwd, check: subprocess.CompletedProcess(
                argv,
                0,
            ),
        )


def test_eval_runner_rejects_invalid_json_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should reject evaluator output that is not valid JSON."""
    _write_onnx_artifact(tmp_path)
    result_path = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _patch_eval_plan(monkeypatch, command=["evaluate"])

    def write_invalid_json(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        result_path.parent.mkdir(parents=True)
        result_path.write_text("{not-json}\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(EvalError, match="invalid JSON"):
        run_lite_scorer_v2_eval(tmp_path, run_command=write_invalid_json)


def test_eval_runner_rejects_non_object_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should require a JSON object result payload."""
    _write_onnx_artifact(tmp_path)
    result_path = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _patch_eval_plan(monkeypatch, command=["evaluate"])

    def write_list_result(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        result_path.parent.mkdir(parents=True)
        result_path.write_text("[]\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(EvalError, match="result must be a JSON object"):
        run_lite_scorer_v2_eval(tmp_path, run_command=write_list_result)


def test_eval_runner_rejects_result_missing_required_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner should report all missing required evaluator fields."""
    _write_onnx_artifact(tmp_path)
    result_path = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _patch_eval_plan(monkeypatch, command=["evaluate"])

    def write_incomplete_result(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        payload = {"heldout_eval_rows": 1000}
        result_path.parent.mkdir(parents=True)
        result_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(
        EvalError,
        match=(
            "benchmarks/results/lite_scorer_v2_eval.json: missing required fields: "
            "heldout_eval_balanced_accuracy, heldout_eval_threshold, "
            "latency_p50_ms, latency_p95_ms, latency_sample_count"
        ),
    ):
        run_lite_scorer_v2_eval(tmp_path, run_command=write_incomplete_result)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"heldout_eval_balanced_accuracy": "0.71"},
            "heldout_eval_balanced_accuracy must be numeric",
        ),
        ({"heldout_eval_rows": 0}, "heldout_eval_rows must be positive"),
        ({"latency_sample_count": 0}, "latency_sample_count must be positive"),
        (
            {"heldout_eval_threshold": 1.1},
            "heldout_eval_threshold must be between 0 and 1",
        ),
        ({"latency_p50_ms": -0.1}, "latency_p50_ms must be non-negative"),
        (
            {"latency_p50_ms": 5.0, "latency_p95_ms": 4.0},
            "latency_p95_ms must be greater than or equal to latency_p50_ms",
        ),
    ],
)
def test_eval_runner_rejects_invalid_numeric_result_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: Mapping[str, object],
    message: str,
) -> None:
    """The runner should fail closed on invalid evaluator metric values."""
    _write_onnx_artifact(tmp_path)
    result_path = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _patch_eval_plan(monkeypatch, command=["evaluate"])

    def write_result(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        _write_eval_result(result_path, payload)
        return subprocess.CompletedProcess(argv, 0)

    with pytest.raises(EvalError, match=message):
        run_lite_scorer_v2_eval(tmp_path, run_command=write_result)


def test_eval_runner_main_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint should print a recorded evaluation receipt."""
    monkeypatch.setattr(
        runner,
        "run_lite_scorer_v2_eval",
        lambda root, *, manifest: {
            "schema_version": "1.0.0",
            "status": "recorded",
            "command": ["evaluate"],
            "eval_result": {
                "path": "benchmarks/results/lite_scorer_v2_eval.json",
                "exists": True,
                "size_bytes": 12,
            },
            "public_score_claim": False,
        },
    )

    assert main([str(tmp_path)]) == 0

    captured = capsys.readouterr()
    assert '"status": "recorded"' in captured.out
    assert captured.err == ""


def test_eval_runner_main_reports_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI entrypoint should print validation errors to stderr."""
    monkeypatch.setattr(
        runner,
        "run_lite_scorer_v2_eval",
        lambda root, *, manifest: (_ for _ in ()).throw(EvalError("eval failed")),
    )

    assert main([str(tmp_path)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "eval failed\n"
