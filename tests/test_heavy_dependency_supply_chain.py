# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — heavy dependency supply-chain tests

"""Check maintained optional-dependency declarations, lock entries and controls."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements" / "heavy_optional_dependency_policy.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
LOCK_PATH = ROOT / "uv.lock"
DOC_PATH = ROOT / "docs-site" / "deployment" / "supply-chain.md"


def _load_toml(path: Path) -> dict[str, object]:
    """Read a maintained TOML policy, lock or package metadata file.

    Parameters
    ----------
    path : Path
        Maintained file to read.

    Returns
    -------
    dict[str, object]
        String-keyed mapping preserving all decoded TOML values.

    Raises
    ------
    OSError
        If the file cannot be read.
    tomllib.TOMLDecodeError
        If its content is invalid TOML.
    ValueError
        If its decoded root is not a string-keyed mapping.
    """
    return object_mapping(tomllib.loads(path.read_text()))


def _dependency_names(requirements: object) -> set[str]:
    """Normalize names from decoded optional-dependency strings.

    Parameters
    ----------
    requirements : object
        Decoded optional-dependency list.

    Returns
    -------
    set[str]
        Lowercase names normalized to hyphen separators.

    Raises
    ------
    AssertionError
        If requirements are not a list of strings.
    """
    assert isinstance(requirements, list)
    names: set[str] = set()
    for requirement in requirements:
        assert isinstance(requirement, str)
        name = re.split(r"[<>=!~;\[\] ]", requirement, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def test_heavy_packages_are_declared_and_locked() -> None:
    """Match heavy-runtime policy packages to declared extras and the lock."""
    policy = _load_toml(POLICY_PATH)
    pyproject = _load_toml(PYPROJECT_PATH)
    lock = _load_toml(LOCK_PATH)

    extras = object_mapping(
        object_mapping(pyproject["project"])["optional-dependencies"]
    )
    locked_packages = lock["package"]
    assert isinstance(locked_packages, list)
    locked_names = {object_mapping(package)["name"] for package in locked_packages}

    packages = policy["packages"]
    assert isinstance(packages, list)
    for record in packages:
        package = object_mapping(record)
        extra = package["extra"]
        assert isinstance(extra, str)
        assert extra in extras
        assert package["name"] in _dependency_names(extras[extra])
        assert package["name"] in locked_names


def test_export_only_packages_are_hash_pinned() -> None:
    """Check that export-only requirements declare package and hash pins."""
    policy = _load_toml(POLICY_PATH)

    export_only = policy["export_only"]
    assert isinstance(export_only, list)
    for record in export_only:
        package = object_mapping(record)
        requirements = package["requirements"]
        assert isinstance(requirements, str)
        text = (ROOT / requirements).read_text()
        assert f"{package['name']}==" in text
        assert "--hash=sha256:" in text


def test_supply_chain_doc_covers_controls_and_packages() -> None:
    """Require the guide to cover policy controls, packages and risks."""
    policy = _load_toml(POLICY_PATH)
    text = DOC_PATH.read_text()

    controls = object_mapping(policy["policy"])["required_controls"]
    assert isinstance(controls, list)
    for control in controls:
        assert isinstance(control, str)
        assert control in text

    packages = policy["packages"]
    assert isinstance(packages, list)
    for record in packages:
        package = object_mapping(record)
        name = package["name"]
        assert isinstance(name, str)
        assert name in text
        risk = package["risk"]
        assert isinstance(risk, str)
        assert risk in text

    export_only = policy["export_only"]
    assert isinstance(export_only, list)
    for record in export_only:
        package = object_mapping(record)
        name = package["name"]
        assert isinstance(name, str)
        assert name in text
