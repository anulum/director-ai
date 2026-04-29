# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — heavy dependency supply-chain tests

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements" / "heavy_optional_dependency_policy.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
LOCK_PATH = ROOT / "uv.lock"
DOC_PATH = ROOT / "docs-site" / "deployment" / "supply-chain.md"


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _dependency_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~;\[\] ]", requirement, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def test_heavy_packages_are_declared_and_locked() -> None:
    policy = _load_toml(POLICY_PATH)
    pyproject = _load_toml(PYPROJECT_PATH)
    lock = _load_toml(LOCK_PATH)

    extras = pyproject["project"]["optional-dependencies"]
    locked_names = {package["name"] for package in lock["package"]}

    for package in policy["packages"]:
        assert package["extra"] in extras
        assert package["name"] in _dependency_names(extras[package["extra"]])
        assert package["name"] in locked_names


def test_export_only_packages_are_hash_pinned() -> None:
    policy = _load_toml(POLICY_PATH)

    for package in policy["export_only"]:
        text = (ROOT / package["requirements"]).read_text()
        assert f"{package['name']}==" in text
        assert "--hash=sha256:" in text


def test_supply_chain_doc_covers_controls_and_packages() -> None:
    policy = _load_toml(POLICY_PATH)
    text = DOC_PATH.read_text()

    for control in policy["policy"]["required_controls"]:
        assert control in text

    for package in policy["packages"]:
        assert package["name"] in text
        assert package["risk"] in text

    for package in policy["export_only"]:
        assert package["name"] in text
