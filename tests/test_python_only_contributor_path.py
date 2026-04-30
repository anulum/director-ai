# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Python-only contributor path tests

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "requirements/python_only_contributor_path.toml"
RUNNER = ROOT / "tools/python_only_check.py"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _load_spec() -> dict[str, object]:
    return tomllib.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("python_only_check", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_spec_points_at_runner_and_make_target() -> None:
    data = _load_spec()

    assert data["status"] == "active"
    assert data["make_target"] == "python-only-check"
    assert data["runner"] == "tools/python_only_check.py"
    assert RUNNER.is_file()


def test_makefile_exposes_python_only_check() -> None:
    makefile = _read("Makefile")

    assert "python-only-check:" in makefile
    assert "python tools/python_only_check.py" in makefile


def test_runner_plan_avoids_optional_toolchains() -> None:
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


def test_runner_rejects_blocked_toolchain_gate() -> None:
    runner = _load_runner()
    gate = runner.Gate("bad", ("cargo", "test"))

    try:
        runner._reject_blocked_toolchains([gate])
    except ValueError as exc:
        assert "blocked optional toolchain" in str(exc)
    else:
        raise AssertionError("blocked toolchain gate was accepted")


def test_contributing_docs_describe_python_only_path() -> None:
    docs = _read("CONTRIBUTING.md")

    assert "make python-only-check" in docs
    assert "Rust, Go, Julia, Lean, or WASM" in docs
