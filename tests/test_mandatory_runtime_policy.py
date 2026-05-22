# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Repository policy tests for mandatory runtime capabilities."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _python_files(*roots: str) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        files.extend((ROOT / root).rglob("*.py"))
    return sorted(files)


def test_no_contextlib_suppression_remains() -> None:
    """No production or test code may hide exceptions with contextlib suppression."""
    forbidden = "suppress" + "("
    offenders = [
        str(path.relative_to(ROOT))
        for path in _python_files("src", "benchmarks", "tools", "tests")
        if forbidden in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_rust_kernel_is_mandatory_dependency() -> None:
    """The Rust kernel is part of the supported runtime, not an optional extra."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]

    assert any(dep.startswith("backfire-kernel") for dep in dependencies)
    assert extras["rust"] == []


def test_rust_availability_flags_do_not_disable_accelerators() -> None:
    """Backfire import guards must not silently mark Rust acceleration unavailable."""
    pattern = re.compile(r"_RUST_[A-Z0-9_]+\\s*=\\s*False")
    offenders = [
        f"{path.relative_to(ROOT)}:{match.group(0)}"
        for path in _python_files("src/director_ai")
        for match in pattern.finditer(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
