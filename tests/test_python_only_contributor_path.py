# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Unit guard for the Python-only contributor path metadata and runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

import pytest

from tools import python_only_check

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "requirements/python_only_contributor_path.toml"
RUNNER = ROOT / "tools/python_only_check.py"


class PythonOnlyContributorSpec(TypedDict):
    """Typed view of the Python-only contributor path specification."""

    status: str
    make_target: str
    runner: str
    python_variable: str
    argument_variable: str
    blocked_toolchains: list[str]


def _read(path: str) -> str:
    """Read a repository-relative text file."""
    return (ROOT / path).read_text(encoding="utf-8")


def _load_spec() -> PythonOnlyContributorSpec:
    """Load the contributor-path TOML specification."""
    return cast(
        "PythonOnlyContributorSpec",
        tomllib.loads(SPEC_PATH.read_text(encoding="utf-8")),
    )


def test_spec_points_at_runner_and_make_target() -> None:
    """The spec should identify the active Make target and runner."""
    data = _load_spec()

    assert data["status"] == "active"
    assert data["make_target"] == "python-only-check"
    assert data["runner"] == "tools/python_only_check.py"
    assert data["python_variable"] == "PYTHON"
    assert data["argument_variable"] == "PYTHON_ONLY_CHECK_ARGS"
    assert RUNNER.is_file()


def test_makefile_exposes_python_only_check() -> None:
    """The Make target should invoke the Python-only runner via override hooks."""
    makefile = _read("Makefile")

    assert "python-only-check:" in makefile
    assert "$(PYTHON) tools/python_only_check.py $(PYTHON_ONLY_CHECK_ARGS)" in makefile


def test_runner_plan_avoids_optional_toolchains() -> None:
    """The printed gate plan should avoid optional runtime toolchains."""
    data = _load_spec()
    blocked = set(data["blocked_toolchains"])
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--print-plan"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)

    assert plan
    for gate in plan:
        command = gate["command"]
        assert Path(command[0]).name not in blocked
        joined = " ".join(command)
        assert all(f" {tool} " not in f" {joined} " for tool in blocked)


def test_runner_main_prints_custom_test_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public runner should print custom pytest paths without executing."""
    assert (
        python_only_check.main(
            ["--print-plan", "tests/test_python_only_contributor_path.py"]
        )
        == 0
    )

    plan = json.loads(capsys.readouterr().out)
    assert [gate["name"] for gate in plan] == [
        "preflight-fast",
        "pytest-python-smoke",
    ]
    assert plan[1]["command"][-2:] == [
        "tests/test_python_only_contributor_path.py",
        "-q",
    ]


def test_runner_main_executes_gates_with_python_only_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public runner should execute gates with the Python-only marker."""
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        calls.append((tuple(command), env.get("DIRECTOR_AI_PYTHON_ONLY")))
        return subprocess.CompletedProcess(args=list(command), returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert python_only_check.main(["--no-tests"]) == 0
    assert calls == [
        (
            (sys.executable, "tools/preflight.py", "--no-tests"),
            "1",
        )
    ]


def test_runner_main_returns_failing_gate_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public runner should stop on the first failing gate."""

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=list(command), returncode=17)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert python_only_check.main(["--no-tests"]) == 17
    assert "preflight-fast failed with exit code 17" in capsys.readouterr().out


def test_runner_rejects_blocked_toolchain_gate() -> None:
    """The unit guard should still cover blocked-toolchain rejection."""
    gate = python_only_check.Gate("bad", ("cargo", "test"))

    try:
        python_only_check._reject_blocked_toolchains([gate])
    except ValueError as exc:
        assert "blocked optional toolchain" in str(exc)
    else:
        raise AssertionError("blocked toolchain gate was accepted")


def test_runner_accepts_empty_gate_list() -> None:
    """The blocked-toolchain guard should accept an empty plan."""
    python_only_check._reject_blocked_toolchains(())


def test_contributing_docs_describe_python_only_path() -> None:
    """Contributor docs should describe the Python-only path and boundaries."""
    docs = _read("CONTRIBUTING.md")

    assert "make python-only-check" in docs
    assert "Rust, Go, Julia, Lean, or WASM" in docs
    assert "PYTHON_ONLY_CHECK_ARGS" in docs
