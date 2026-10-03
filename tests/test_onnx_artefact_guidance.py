# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ONNX artefact guidance tests

"""Check ONNX dependency, hash-pin and documentation manifests without inference."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements" / "onnx_wheel_targets.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
DOC_PATH = ROOT / "docs-site" / "deployment" / "onnx-artefacts.md"


def _load_toml(path: Path) -> dict[str, object]:
    """Read a maintained TOML file without erasing decoded value types.

    Parameters
    ----------
    path : Path
        Policy or package metadata file to read.

    Returns
    -------
    dict[str, object]
        String-keyed mapping retaining the original decoded TOML values.

    Raises
    ------
    OSError
        If the file cannot be read.
    tomllib.TOMLDecodeError
        If the file contains invalid TOML.
    ValueError
        If its decoded root is not a string-keyed mapping.
    """
    return object_mapping(tomllib.loads(path.read_text()))


def _dependency_names(requirements: object) -> set[str]:
    """Normalize package names from declared dependency strings.

    Parameters
    ----------
    requirements : object
        Decoded optional-dependency list from package metadata.

    Returns
    -------
    set[str]
        Lowercase package names with underscores replaced by hyphens.

    Raises
    ------
    AssertionError
        If the decoded value is not a list of dependency strings.
    """
    assert isinstance(requirements, list)
    names: set[str] = set()
    for requirement in requirements:
        assert isinstance(requirement, str)
        name = re.split(r"[<>=!~;\[\] ]", requirement, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def test_onnx_targets_map_to_project_extras() -> None:
    """Check that each declared ONNX target uses a declared package extra."""
    policy = _load_toml(POLICY_PATH)
    pyproject = _load_toml(PYPROJECT_PATH)
    extras = object_mapping(
        object_mapping(pyproject["project"])["optional-dependencies"]
    )

    targets = policy["targets"]
    assert isinstance(targets, list)
    for record in targets:
        target = object_mapping(record)
        extra = target["extra"]
        assert isinstance(extra, str)
        assert extra in extras
        assert target["package"] in _dependency_names(extras[extra])


def test_export_wheel_file_is_pinned() -> None:
    """Require the export requirement file to retain versions and hashes."""
    policy = _load_toml(POLICY_PATH)
    controls = object_mapping(policy["policy"])
    export_requirements = controls["export_requirements"]
    assert isinstance(export_requirements, str)
    requirements_path = ROOT / export_requirements
    text = requirements_path.read_text()

    assert "onnx==" in text
    assert "onnxruntime==" in text
    assert "onnxscript==" in text
    assert "--hash=sha256:" in text


def test_onnx_artefact_docs_cover_targets_and_files() -> None:
    """Match documented ONNX providers and artefacts to the target manifest."""
    policy = _load_toml(POLICY_PATH)
    text = DOC_PATH.read_text()

    controls = object_mapping(policy["policy"])
    model_dir = controls["quickstart_model_dir"]
    assert isinstance(model_dir, str)
    assert model_dir in text
    assert "director-ai export --format onnx" in text

    required_files = controls["required_files"]
    assert isinstance(required_files, list)
    for required_file in required_files:
        assert isinstance(required_file, str)
        assert required_file in text

    targets = policy["targets"]
    assert isinstance(targets, list)
    for record in targets:
        target = object_mapping(record)
        target_id = target["id"]
        assert isinstance(target_id, str)
        assert target_id in text
        provider = target["provider"]
        assert isinstance(provider, str)
        assert provider in text
