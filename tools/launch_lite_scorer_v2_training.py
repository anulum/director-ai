#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 durable training launcher

"""Launch Lite Scorer v2 training as a durable, auditable background run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import signal
import subprocess  # nosec B404
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

PLANNER = Path(__file__).resolve().parent / "plan_lite_scorer_v2_run.py"
SPEC = importlib.util.spec_from_file_location("plan_lite_scorer_v2_run", PLANNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load Lite Scorer v2 planner from {PLANNER}")
_PLANNER_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = _PLANNER_MODULE
SPEC.loader.exec_module(_PLANNER_MODULE)
MANIFEST = _PLANNER_MODULE.MANIFEST
build_lite_scorer_v2_run_plan = _PLANNER_MODULE.build_lite_scorer_v2_run_plan

RUN_ROOT = Path(".coordination/runs/DIRECTOR-AI")
RUN_PREFIX = "lite_scorer_v2_train"
LAUNCH_WRAPPER = """set -o pipefail
echo "START $(date -Is)"
echo "PWD $PWD"
free -h || true
env PYTHONUNBUFFERED=1 "$@"
rc=$?
echo "EXIT $rc $(date -Is)"
free -h || true
exit "$rc"
"""


class LaunchError(RuntimeError):
    """Raised when the durable training launcher cannot start safely."""


def find_train_argv(plan: dict[str, Any]) -> list[str]:
    """Return the training argv from a Lite Scorer v2 run plan."""
    commands = plan.get("commands")
    if not isinstance(commands, list):
        raise LaunchError("run plan does not contain a commands list")
    for command in commands:
        if not isinstance(command, dict) or command.get("name") != "train":
            continue
        argv = command.get("argv")
        if not isinstance(argv, list) or not all(
            isinstance(value, str) and value for value in argv
        ):
            raise LaunchError("run plan train command must be a non-empty argv list")
        return list(argv)
    raise LaunchError("run plan does not contain a train command")


def is_process_running(pid: int) -> bool:
    """Return whether ``pid`` appears to identify a live process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_lite_scorer_v2_run_root(root: Path) -> Path:
    """Return the durable directory used for Lite Scorer v2 training runs."""
    resolved_root = root.resolve()
    if resolved_root.name == "DIRECTOR-AI" and resolved_root.parent.name == "03_CODE":
        coordination_root = resolved_root.parent.parent / ".coordination"
        if coordination_root.is_dir():
            return coordination_root / "runs" / "DIRECTOR-AI"
    return resolved_root / RUN_ROOT


def _active_training_runs(
    root: Path,
    process_running: Callable[[int], bool],
) -> list[tuple[Path, int]]:
    """Return active Lite Scorer v2 training run directories and PIDs."""
    run_root = resolve_lite_scorer_v2_run_root(root)
    if not run_root.exists():
        return []
    active: list[tuple[Path, int]] = []
    for run_dir in sorted(run_root.glob(f"{RUN_PREFIX}_*")):
        pid_file = run_dir / "pid"
        if not pid_file.is_file():
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            continue
        if process_running(pid):
            active.append((run_dir, pid))
    return active


def _timestamp() -> str:
    """Return the launch timestamp used in durable run directory names."""
    return datetime.now().strftime("%Y-%m-%dT%H%M%S")


def _normalise_manifest(root: Path, manifest: Path) -> Path:
    """Resolve ``manifest`` relative to ``root`` when it is not absolute."""
    return manifest if manifest.is_absolute() else root / manifest


def _write_launch_files(
    run_dir: Path,
    *,
    command: list[str],
    pid: int,
    session_id: int | None,
    plan: dict[str, Any],
) -> None:
    """Write launch metadata files inside ``run_dir``."""
    (run_dir / "pid").write_text(f"{pid}\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    metadata = {
        "command": command,
        "pid": pid,
        "plan_id": plan.get("plan_id"),
        "public_score_claim": False,
        "run_id": run_dir.name,
        "session_id": session_id,
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def launch_lite_scorer_v2_training(
    root: Path,
    *,
    manifest: Path = MANIFEST,
    timestamp: str | None = None,
    popen: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    process_running: Callable[[int], bool] = is_process_running,
) -> dict[str, Any]:
    """Launch Lite Scorer v2 training and return durable run metadata."""
    root = root.resolve()
    active = _active_training_runs(root, process_running)
    if active:
        run_dir, pid = active[0]
        raise LaunchError(
            "active Lite Scorer v2 training run already exists: "
            f"{run_dir.as_posix()} (pid {pid})"
        )

    manifest_path = _normalise_manifest(root, manifest)
    plan, errors = build_lite_scorer_v2_run_plan(root, manifest_path)
    if errors:
        raise LaunchError("; ".join(errors))
    command = find_train_argv(plan)

    stamp = timestamp or _timestamp()
    run_dir = resolve_lite_scorer_v2_run_root(root) / f"{RUN_PREFIX}_{stamp}"
    if run_dir.exists():
        raise LaunchError(f"run directory already exists: {run_dir.as_posix()}")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train.log"

    with log_path.open("ab") as log_file:
        process = popen(
            ["bash", "-c", LAUNCH_WRAPPER, "lite-scorer-v2-train", *command],
            cwd=root,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        session_id = os.getsid(process.pid)
    except ProcessLookupError:
        session_id = None
    _write_launch_files(
        run_dir,
        command=command,
        pid=process.pid,
        session_id=session_id,
        plan=plan,
    )
    return {
        "run_dir": run_dir.as_posix(),
        "pid": process.pid,
        "session_id": session_id,
        "log": log_path.as_posix(),
    }


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the durable launcher."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Repository root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="Lite Scorer v2 run manifest path",
    )
    parser.add_argument(
        "--timestamp",
        help="Run timestamp override for deterministic tests or handovers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the durable launcher command-line interface."""
    args = _build_parser().parse_args(argv)
    try:
        result = launch_lite_scorer_v2_training(
            args.root,
            manifest=args.manifest,
            timestamp=args.timestamp,
        )
    except LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    raise SystemExit(main())
