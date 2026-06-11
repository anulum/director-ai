# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — optional extra lock policy tests

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements" / "uv_extra_lock_policy.toml"
DOC_PATH = ROOT / "requirements" / "OPTIONAL_EXTRA_LOCKS.md"
PYPROJECT_PATH = ROOT / "pyproject.toml"
LOCK_PATH = ROOT / "uv.lock"

EXPECTED_EXTRAS = {"enterprise", "nli", "onnx", "physical", "server", "ui", "vector"}


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _dependency_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[\] ]", requirement, maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _has_upper_bound(requirement: str) -> bool:
    return "==" in requirement or "<" in requirement


def test_policy_covers_target_optional_extras() -> None:
    policy = _load_toml(POLICY_PATH)

    assert set(policy["policy"]["checked_extras"]) == EXPECTED_EXTRAS
    assert {extra["name"] for extra in policy["extras"]} == EXPECTED_EXTRAS


def test_optional_extra_roots_are_capped_and_locked() -> None:
    pyproject = _load_toml(PYPROJECT_PATH)
    lock = _load_toml(LOCK_PATH)
    policy = _load_toml(POLICY_PATH)

    optional = pyproject["project"]["optional-dependencies"]
    locked_names = {package["name"] for package in lock["package"]}

    for extra in policy["extras"]:
        declared = {_dependency_name(item): item for item in optional[extra["name"]]}

        for package_name in extra["packages"]:
            assert package_name in declared, f"{extra['name']} missing {package_name}"
            assert _has_upper_bound(declared[package_name]), declared[package_name]
            assert package_name in locked_names, f"uv.lock missing {package_name}"


def test_lock_policy_is_documented() -> None:
    text = DOC_PATH.read_text()

    assert "uv.lock" in text
    assert "uv lock" in text
    assert "uv sync --locked" in text
