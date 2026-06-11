#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 training status verifier

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUN_ROOT = Path(".coordination/runs/DIRECTOR-AI")
RUN_PREFIX = "lite_scorer_v2_train"
MANIFEST = Path("benchmarks/lite_scorer_v2_run_manifest.toml")
EXIT_PATTERN = re.compile(r"^EXIT (?P<code>\d+) (?P<timestamp>\S+)$")


class StatusError(RuntimeError):
    """Raised when training status cannot be inspected safely."""


def is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _latest_run_dir(root: Path) -> Path:
    run_root = root / RUN_ROOT
    candidates = sorted(run_root.glob(f"{RUN_PREFIX}_*")) if run_root.exists() else []
    if not candidates:
        raise StatusError(f"no Lite Scorer v2 training runs found under {run_root}")
    return candidates[-1]


def _read_pid(run_dir: Path) -> int | None:
    pid_file = run_dir / "pid"
    if not pid_file.is_file():
        return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise StatusError(f"{pid_file}: invalid pid") from exc


def _read_manifest(root: Path, manifest: Path) -> dict[str, Any]:
    manifest_path = manifest if manifest.is_absolute() else root / manifest
    try:
        return tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StatusError(f"{manifest_path}: missing manifest") from exc
    except tomllib.TOMLDecodeError as exc:
        raise StatusError(f"{manifest_path}: invalid TOML: {exc}") from exc


def _string_field(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise StatusError(f"manifest field {field} must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise StatusError(f"manifest field {field} must stay inside the repository")
    return value


def _exit_marker(log_path: Path) -> tuple[int | None, str | None]:
    if not log_path.is_file():
        return None, None
    exit_code: int | None = None
    exit_timestamp: str | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EXIT_PATTERN.match(line.strip())
        if match is None:
            continue
        exit_code = int(match.group("code"))
        exit_timestamp = match.group("timestamp")
    return exit_code, exit_timestamp


def _artefact_status(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    exists = path.is_file()
    return {
        "path": relative_path,
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def _state(
    *,
    running: bool,
    exit_code: int | None,
) -> str:
    if running:
        return "running"
    if exit_code is None:
        return "stale"
    if exit_code == 0:
        return "completed"
    return "failed"


def inspect_lite_scorer_v2_training(
    root: Path,
    *,
    manifest: Path = MANIFEST,
    run_dir: Path | None = None,
    process_running: Callable[[int], bool] = is_process_running,
) -> dict[str, Any]:
    root = root.resolve()
    inspected_run_dir = run_dir if run_dir is not None else _latest_run_dir(root)
    if not inspected_run_dir.is_absolute():
        inspected_run_dir = root / inspected_run_dir
    data = _read_manifest(root, manifest)
    student_artifact = _string_field(data, "student_artifact")
    train_output_dir = _string_field(data, "train_output_dir")

    pid = _read_pid(inspected_run_dir)
    running = process_running(pid) if pid is not None else False
    exit_code, exit_timestamp = _exit_marker(inspected_run_dir / "train.log")
    student_status = _artefact_status(root, student_artifact)
    manifest_status = _artefact_status(
        root,
        f"{train_output_dir}/training_run_manifest.json",
    )
    state = _state(running=running, exit_code=exit_code)
    export_ready = state == "completed" and student_status["exists"]

    return {
        "schema_version": "1.0.0",
        "run_dir": _relative(root, inspected_run_dir),
        "pid": pid,
        "process_running": running,
        "state": state,
        "exit_code": exit_code,
        "exit_timestamp": exit_timestamp,
        "export_ready": export_ready,
        "public_score_claim": False,
        "artefacts": {
            "training_run_manifest": manifest_status,
            "student_artifact": student_status,
        },
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
        help="Specific run directory; defaults to latest Lite Scorer v2 training run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        status = inspect_lite_scorer_v2_training(
            args.root,
            manifest=args.manifest,
            run_dir=args.run_dir,
        )
    except StatusError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
