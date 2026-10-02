# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — executable repository contract

from __future__ import annotations

import tomllib
from pathlib import Path

from tools.audit_ci_workflow_modularity import read_workflow
from tools.ci_workflow_inventory import (
    object_mapping,
    string_tuple,
    workflow_path_for_job,
)

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI_WORKFLOW = workflow_path_for_job("test-extras", ROOT)


def _extras() -> dict[str, list[str]]:
    """Read validated real optional dependencies from the package contract."""
    data = object_mapping(tomllib.loads(PYPROJECT.read_text(encoding="utf-8")))
    extras = object_mapping(object_mapping(data["project"])["optional-dependencies"])
    return {
        name: list(string_tuple(requirements)) for name, requirements in extras.items()
    }


def _ci_extras_matrix() -> set[str]:
    """Read the real executable extras matrix through its declared owner."""
    workflow = read_workflow(CI_WORKFLOW)
    job = object_mapping(object_mapping(workflow["jobs"])["test-extras"])
    matrix = object_mapping(object_mapping(job["strategy"])["matrix"])
    return set(string_tuple(matrix["extras"]))


def test_optional_dependencies_expose_formal_solver_extra() -> None:
    extras = _extras()

    assert "formal" in extras
    assert any(req.startswith("z3-solver") for req in extras["formal"])


def test_ci_extras_matrix_covers_heavy_optional_science_lanes() -> None:
    matrix = _ci_extras_matrix()

    required = {
        "dev,server",
        "dev,grpc",
        "dev,faiss",
        "dev,rust",
        "dev,formal",
        "dev,finetune,train,managed-training",
    }
    assert required <= matrix
