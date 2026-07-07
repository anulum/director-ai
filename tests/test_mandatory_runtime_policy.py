# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Repository policy tests for the ADR-0001 hybrid accelerator contract.

The Rust kernel is an opt-in ``rust`` extra: a base install runs a pure-Python
floor, and the compiled accelerator ships only when the caller asks for
``director-ai[rust]``. Bit-exactness of any pure-Python fallback is proven per
kernel by the parity tests (e.g. ``test_accelerator_fallback_parity.py``) and the
base install is exercised end-to-end by the ``floor`` CI job -- a repository grep
cannot verify numerical equivalence, so this file asserts only the packaging
policy. See ``docs/adr/0001-rust-accelerator-hybrid-fallback.md``.
"""

from __future__ import annotations

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


def test_rust_kernel_is_optional_rust_extra() -> None:
    """The Rust kernel is an opt-in ``rust`` extra, not a base dependency (ADR-0001).

    A base ``pip install director-ai`` must resolve to the pure-Python floor; the
    compiled accelerator ships only when the caller asks for ``director-ai[rust]``.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    extras = pyproject["project"]["optional-dependencies"]

    assert not any(dep.startswith("backfire-kernel") for dep in dependencies)
    assert any(req.startswith("backfire-kernel") for req in extras["rust"])
