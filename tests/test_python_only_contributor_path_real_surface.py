# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Python-only contributor real-surface tests
"""Real Makefile/CLI coverage for the Python-only contributor path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the checked-out repository root."""
    return Path(__file__).resolve().parents[1]


def _tool_env(repo_root: Path) -> dict[str, str]:
    """Return an environment that imports the repository source tree."""
    env = os.environ.copy()
    entries = [str(repo_root / "src"), str(repo_root)]
    existing = env.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def test_runner_cli_prints_python_only_plan() -> None:
    """The public runner CLI should emit a Python-only gate plan."""
    repo_root = _repo_root()

    result = subprocess.run(
        [sys.executable, "tools/python_only_check.py", "--no-tests", "--print-plan"],
        cwd=repo_root,
        env=_tool_env(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads(result.stdout)
    assert [gate["name"] for gate in plan] == ["preflight-fast"]
    assert plan[0]["command"][:2] == [sys.executable, "tools/preflight.py"]


def test_make_target_forwards_python_and_runner_args() -> None:
    """The Make target should forward interpreter and runner arguments."""
    repo_root = _repo_root()

    result = subprocess.run(
        [
            "make",
            "--dry-run",
            "--silent",
            "python-only-check",
            f"PYTHON={sys.executable}",
            "PYTHON_ONLY_CHECK_ARGS=--print-plan",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == (
        f"{sys.executable} tools/python_only_check.py --print-plan"
    )


def test_make_target_executes_runner_plan_with_overrides() -> None:
    """The real Make target should execute the selected Python-only plan."""
    repo_root = _repo_root()

    result = subprocess.run(
        [
            "make",
            "--silent",
            "python-only-check",
            f"PYTHON={sys.executable}",
            "PYTHON_ONLY_CHECK_ARGS=--no-tests --print-plan",
        ],
        cwd=repo_root,
        env=_tool_env(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    plan = json.loads(result.stdout)
    assert [gate["name"] for gate in plan] == ["preflight-fast"]
    assert plan[0]["command"][:2] == [sys.executable, "tools/preflight.py"]
