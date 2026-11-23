# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — tests that only the Core wheel ships the support contract
"""Refuse a paid wheel that ships the contract, and a Core wheel that lost it (C-1)."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILES = (
    "director_ai/support_contract.py",
    "director_ai/support_contract_catalog.py",
)


def _load(name: str) -> ModuleType:
    """Import a wheel-check script by path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAID = _load("check_paid_wheel")
FREE = _load("check_free_wheel")


def _wheel(directory: Path, name: str, files: tuple[str, ...]) -> Path:
    """Write a wheel archive holding the named empty files."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with zipfile.ZipFile(path, "w") as archive:
        for file in files:
            archive.writestr(file, "")
    return path


def test_a_paid_wheel_may_not_ship_the_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Name the contract files a Pro wheel ships, alongside its other faults."""
    _wheel(tmp_path, "director_ai_pro-0-py3-none-any.whl", CONTRACT_FILES)
    assert PAID.main(str(tmp_path), "pro") == 1
    assert (
        f"ships the Core support contract: {list(CONTRACT_FILES)}"
        in capsys.readouterr().out
    )


def test_a_paid_wheel_without_the_contract_passes_that_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Raise no contract fault for a paid wheel that does not ship it."""
    _wheel(tmp_path, "director_ai_pro-0-py3-none-any.whl", ("director_ai/server.py",))
    PAID.main(str(tmp_path), "pro")
    assert "support contract" not in capsys.readouterr().out


def test_the_core_wheel_must_keep_the_contract(tmp_path: Path) -> None:
    """List the contract among the lost free essentials of a Core wheel."""
    path = _wheel(
        tmp_path, "director_ai-0-py3-none-any.whl", ("director_ai/__init__.py",)
    )
    failures = FREE._check_contents(str(path))
    lost = next(
        failure for failure in failures if failure.startswith("lost free essentials")
    )
    for file in CONTRACT_FILES:
        assert file in lost
