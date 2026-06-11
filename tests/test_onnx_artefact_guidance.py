# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ONNX artefact guidance tests

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements" / "onnx_wheel_targets.toml"
PYPROJECT_PATH = ROOT / "pyproject.toml"
DOC_PATH = ROOT / "docs-site" / "deployment" / "onnx-artefacts.md"


def _load_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def _dependency_names(requirements: list[str]) -> set[str]:
    names = set()
    for requirement in requirements:
        name = re.split(r"[<>=!~;\[\] ]", requirement, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def test_onnx_targets_map_to_project_extras() -> None:
    policy = _load_toml(POLICY_PATH)
    pyproject = _load_toml(PYPROJECT_PATH)
    extras = pyproject["project"]["optional-dependencies"]

    for target in policy["targets"]:
        assert target["extra"] in extras
        assert target["package"] in _dependency_names(extras[target["extra"]])


def test_export_wheel_file_is_pinned() -> None:
    policy = _load_toml(POLICY_PATH)
    requirements_path = ROOT / policy["policy"]["export_requirements"]
    text = requirements_path.read_text()

    assert "onnx==" in text
    assert "onnxruntime==" in text
    assert "onnxscript==" in text
    assert "--hash=sha256:" in text


def test_onnx_artefact_docs_cover_targets_and_files() -> None:
    policy = _load_toml(POLICY_PATH)
    text = DOC_PATH.read_text()

    assert policy["policy"]["quickstart_model_dir"] in text
    assert "director-ai export --format onnx" in text

    for required_file in policy["policy"]["required_files"]:
        assert required_file in text

    for target in policy["targets"]:
        assert target["id"] in text
        assert target["provider"] in text
