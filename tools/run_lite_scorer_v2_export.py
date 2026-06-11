#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 guarded ONNX export runner

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent


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
_STATUS = _load_tool(
    "status_lite_scorer_v2_training",
    "status_lite_scorer_v2_training.py",
)

MANIFEST = _PLANNER.MANIFEST
build_lite_scorer_v2_run_plan = _PLANNER.build_lite_scorer_v2_run_plan
inspect_lite_scorer_v2_training = _STATUS.inspect_lite_scorer_v2_training


class ExportError(RuntimeError):
    """Raised when Lite Scorer v2 ONNX export is not safe or complete."""


def find_export_argv(plan: dict[str, Any]) -> list[str]:
    commands = plan.get("commands")
    if not isinstance(commands, list):
        raise ExportError("run plan does not contain a commands list")
    for command in commands:
        if not isinstance(command, dict) or command.get("name") != "export_onnx":
            continue
        argv = command.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise ExportError(
                "run plan export_onnx command must be a non-empty argv list"
            )
        return list(argv)
    raise ExportError("run plan does not contain an export_onnx command")


def _onnx_artifact_path(plan: dict[str, Any]) -> str:
    generated_paths = plan.get("generated_paths")
    if not isinstance(generated_paths, dict):
        raise ExportError("run plan does not contain generated_paths")
    artifact = generated_paths.get("onnx_artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ExportError("run plan generated_paths.onnx_artifact must be a string")
    path = Path(artifact)
    if path.is_absolute() or ".." in path.parts:
        raise ExportError("run plan ONNX artifact must stay inside the repository")
    return artifact


def _artifact_status(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    exists = path.is_file()
    return {
        "path": relative_path,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def _assert_export_ready(status: dict[str, Any]) -> None:
    if status.get("export_ready") is True:
        return
    raise ExportError(
        "Lite Scorer v2 training is not export-ready: "
        f"state={status.get('state')}, export_ready={status.get('export_ready')}"
    )


def run_lite_scorer_v2_export(
    root: Path,
    *,
    manifest: Path = MANIFEST,
    run_dir: Path | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> dict[str, Any]:
    root = root.resolve()
    status = inspect_lite_scorer_v2_training(
        root,
        manifest=manifest,
        run_dir=run_dir,
    )
    _assert_export_ready(status)
    plan, errors = build_lite_scorer_v2_run_plan(root, manifest)
    if errors:
        raise ExportError("; ".join(errors))
    command = find_export_argv(plan)
    artifact = _onnx_artifact_path(plan)

    try:
        run_command(command, cwd=root, check=True)
    except subprocess.CalledProcessError as exc:
        raise ExportError(
            f"ONNX export command failed with exit code {exc.returncode}"
        ) from exc

    artifact_status = _artifact_status(root, artifact)
    if not artifact_status["exists"]:
        raise ExportError(f"ONNX export command completed but {artifact} is missing")

    return {
        "schema_version": "1.0.0",
        "status": "recorded",
        "command": command,
        "onnx_artifact": artifact_status,
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
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Specific training run directory for readiness verification",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = run_lite_scorer_v2_export(
            args.root,
            manifest=args.manifest,
            run_dir=args.run_dir,
        )
    except ExportError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
