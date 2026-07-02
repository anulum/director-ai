#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 guarded evaluation runner
"""Run the guarded Lite Scorer v2 evaluation command from the run plan."""

from __future__ import annotations

import argparse
import importlib.util
import json

# The runner executes only repo-validated argv without shell expansion.
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
REQUIRED_RESULT_FIELDS = {
    "heldout_eval_balanced_accuracy",
    "heldout_eval_rows",
    "heldout_eval_threshold",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_sample_count",
}


def _load_tool(name: str, filename: str) -> Any:
    path = TOOLS_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PLANNER = _load_tool("plan_lite_scorer_v2_run", "plan_lite_scorer_v2_run.py")

MANIFEST = _PLANNER.MANIFEST
build_lite_scorer_v2_run_plan = _PLANNER.build_lite_scorer_v2_run_plan


class EvalError(RuntimeError):
    """Raised when Lite Scorer v2 evaluation is unsafe or incomplete."""


def find_evaluate_argv(plan: dict[str, Any]) -> list[str]:
    """Return the evaluate command argv from a Lite Scorer v2 run plan.

    Parameters
    ----------
    plan:
        Run-plan mapping emitted by ``tools/plan_lite_scorer_v2_run.py``.

    Returns
    -------
    list[str]
        Command vector for the ``evaluate`` step.

    Raises
    ------
    EvalError
        If the plan does not contain a valid evaluate command.
    """
    commands = plan.get("commands")
    if not isinstance(commands, list):
        raise EvalError("run plan does not contain a commands list")
    for command in commands:
        if not isinstance(command, dict) or command.get("name") != "evaluate":
            continue
        argv = command.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise EvalError("run plan evaluate command must be a non-empty argv list")
        return list(argv)
    raise EvalError("run plan does not contain an evaluate command")


def _generated_path(plan: dict[str, Any], field: str) -> str:
    generated_paths = plan.get("generated_paths")
    if not isinstance(generated_paths, dict):
        raise EvalError("run plan does not contain generated_paths")
    value = generated_paths.get(field)
    if not isinstance(value, str) or not value:
        raise EvalError(f"run plan generated_paths.{field} must be a string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise EvalError(
            f"run plan generated_paths.{field} must stay inside the repository"
        )
    return value


def _artifact_status(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    exists = path.is_file()
    return {
        "path": relative_path,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def _read_eval_result(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.is_file():
        raise EvalError(f"evaluation command completed but {relative_path} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalError(f"{relative_path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvalError(f"{relative_path}: result must be a JSON object")
    missing = sorted(REQUIRED_RESULT_FIELDS.difference(payload))
    if missing:
        raise EvalError(
            f"{relative_path}: missing required fields: {', '.join(missing)}"
        )
    return payload


def _assert_onnx_artifact_ready(root: Path, relative_path: str) -> None:
    path = root / relative_path
    if not path.is_file():
        raise EvalError(f"Lite Scorer v2 ONNX artifact is missing: {relative_path}")
    if path.stat().st_size <= 0:
        raise EvalError(f"Lite Scorer v2 ONNX artifact is empty: {relative_path}")


def _validate_numeric_result(relative_path: str, payload: dict[str, Any]) -> None:
    for field in REQUIRED_RESULT_FIELDS:
        value = payload[field]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise EvalError(f"{relative_path}: {field} must be numeric")
    if int(payload["heldout_eval_rows"]) < 1:
        raise EvalError(f"{relative_path}: heldout_eval_rows must be positive")
    if int(payload["latency_sample_count"]) < 1:
        raise EvalError(f"{relative_path}: latency_sample_count must be positive")
    for field in ("heldout_eval_balanced_accuracy", "heldout_eval_threshold"):
        value = float(payload[field])
        if value < 0.0 or value > 1.0:
            raise EvalError(f"{relative_path}: {field} must be between 0 and 1")
    for field in ("latency_p50_ms", "latency_p95_ms"):
        if float(payload[field]) < 0.0:
            raise EvalError(f"{relative_path}: {field} must be non-negative")
    if float(payload["latency_p95_ms"]) < float(payload["latency_p50_ms"]):
        raise EvalError(
            f"{relative_path}: latency_p95_ms must be greater than or equal to "
            "latency_p50_ms"
        )


def run_lite_scorer_v2_eval(
    root: Path,
    *,
    manifest: Path = MANIFEST,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    """Run the guarded evaluation command and return a recorded receipt.

    Parameters
    ----------
    root:
        Repository root containing the Lite Scorer v2 run manifest and ONNX
        artefact.
    manifest:
        Run manifest path, either absolute or relative to ``root``.
    run_command:
        Command runner compatible with ``subprocess.run``. Tests may provide a
        protocol-preserving command runner for the external evaluation process.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable evaluation receipt with artefact status and no public
        score claim.

    Raises
    ------
    EvalError
        If the run plan is invalid, the ONNX artefact is missing or empty, the
        evaluation command fails, or the evaluator JSON is incomplete.
    """
    root = root.resolve()
    plan, errors = build_lite_scorer_v2_run_plan(root, manifest)
    if errors:
        raise EvalError("; ".join(errors))
    command = find_evaluate_argv(plan)
    onnx_artifact = _generated_path(plan, "onnx_artifact")
    eval_result = _generated_path(plan, "eval_result")

    _assert_onnx_artifact_ready(root, onnx_artifact)

    try:
        run_command(command, cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        raise EvalError(
            f"Lite Scorer v2 evaluation failed with exit code {exc.returncode}"
        ) from exc

    payload = _read_eval_result(root, eval_result)
    _validate_numeric_result(eval_result, payload)
    return {
        "schema_version": "1.0.0",
        "status": "recorded",
        "command": command,
        "eval_result": _artifact_status(root, eval_result),
        "public_score_claim": False,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Repository root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="Lite Scorer v2 run manifest path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line guarded evaluation runner."""
    args = _build_parser().parse_args(argv)
    try:
        result = run_lite_scorer_v2_eval(args.root, manifest=args.manifest)
    except EvalError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
