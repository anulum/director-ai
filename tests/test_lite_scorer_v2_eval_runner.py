# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evaluation runner tests

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_lite_scorer_v2_eval.py"
SPEC = importlib.util.spec_from_file_location("run_lite_scorer_v2_eval", RUNNER)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

EvalError = MODULE.EvalError
find_evaluate_argv = MODULE.find_evaluate_argv
run_lite_scorer_v2_eval = MODULE.run_lite_scorer_v2_eval


def _patch_plan(monkeypatch: Any, *, result_path: str, onnx_path: str) -> None:
    monkeypatch.setattr(
        MODULE,
        "build_lite_scorer_v2_run_plan",
        lambda root, manifest: (
            {
                "plan_id": "lite-scorer-v2-run-plan",
                "commands": [
                    {
                        "name": "evaluate",
                        "argv": [
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
                    "heldout_eval_dataset": "benchmarks/heldout/lite_scorer_v2.jsonl",
                    "onnx_artifact": onnx_path,
                },
            },
            [],
        ),
    )


def _write_eval_result(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "heldout_eval_balanced_accuracy": 0.71,
                "heldout_eval_rows": 1000,
                "heldout_eval_threshold": 0.53,
                "latency_p50_ms": 2.4,
                "latency_p95_ms": 4.8,
                "latency_sample_count": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_find_evaluate_argv_returns_planned_evaluate_command() -> None:
    plan = {
        "commands": [
            {"name": "export_onnx", "argv": ["export"]},
            {"name": "evaluate", "argv": ["python", "tools/eval_lite_scorer_v2.py"]},
        ]
    }

    assert find_evaluate_argv(plan) == ["python", "tools/eval_lite_scorer_v2.py"]


def test_eval_runner_refuses_missing_onnx_artifact(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _patch_plan(
        monkeypatch,
        result_path="benchmarks/results/lite_scorer_v2_eval.json",
        onnx_path="MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
    )

    try:
        run_lite_scorer_v2_eval(tmp_path)
    except EvalError as exc:
        assert str(exc) == (
            "Lite Scorer v2 ONNX artifact is missing: "
            "MODELS/lite-scorer-v2/onnx/model_quantized.onnx"
        )
    else:
        raise AssertionError("expected EvalError")


def test_eval_runner_uses_plan_command_and_validates_result(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []
    onnx = tmp_path / "MODELS" / "lite-scorer-v2" / "onnx" / "model_quantized.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"onnx")
    result_path = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _patch_plan(
        monkeypatch,
        result_path="benchmarks/results/lite_scorer_v2_eval.json",
        onnx_path="MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
    )

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


def test_eval_runner_rejects_result_missing_required_fields(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    onnx = tmp_path / "MODELS" / "lite-scorer-v2" / "onnx" / "model_quantized.onnx"
    onnx.parent.mkdir(parents=True)
    onnx.write_bytes(b"onnx")
    result_path = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    _patch_plan(
        monkeypatch,
        result_path="benchmarks/results/lite_scorer_v2_eval.json",
        onnx_path="MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
    )

    def fake_run(
        argv: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        result_path.parent.mkdir(parents=True)
        result_path.write_text('{"heldout_eval_rows": 1000}\n', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    try:
        run_lite_scorer_v2_eval(tmp_path, run_command=fake_run)
    except EvalError as exc:
        assert str(exc) == (
            "benchmarks/results/lite_scorer_v2_eval.json: missing required fields: "
            "heldout_eval_balanced_accuracy, heldout_eval_threshold, "
            "latency_p50_ms, latency_p95_ms, latency_sample_count"
        )
    else:
        raise AssertionError("expected EvalError")
