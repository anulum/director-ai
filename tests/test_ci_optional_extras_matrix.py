# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI -- CI optional extras contract

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _extras() -> dict[str, list[str]]:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]


def _ci_extras_matrix() -> set[str]:
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    return set(workflow["jobs"]["test-extras"]["strategy"]["matrix"]["extras"])


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
