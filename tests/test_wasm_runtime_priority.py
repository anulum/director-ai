# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WASM runtime priority tests

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "requirements" / "wasm_release_plan.toml"
DOC_PATH = ROOT / "docs-site" / "deployment" / "wasm-runtime.md"
MAKEFILE_PATH = ROOT / "Makefile"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "wheels.yml"


def _load_plan() -> dict:
    return tomllib.loads(PLAN_PATH.read_text())


def test_wasm_plan_points_at_existing_runtime_surface() -> None:
    plan = _load_plan()
    policy = plan["policy"]

    assert (ROOT / policy["crate"]).is_dir()
    assert (ROOT / policy["doc"]).is_file()
    assert (ROOT / policy["workflow"]).is_file()
    assert (ROOT / policy["package_validator"]).is_file()
    assert "coherence score per token" in policy["scope"]


def test_wasm_make_targets_and_workflow_are_wired() -> None:
    plan = _load_plan()
    make_text = MAKEFILE_PATH.read_text()
    workflow_text = WORKFLOW_PATH.read_text()

    for target in plan["policy"]["make_targets"]:
        assert f"{target}:" in make_text

    assert "build-wasm:" in workflow_text
    assert "wasm-pack build --target web --release" in workflow_text
    assert "wasm-edge-runtime" in workflow_text


def test_wasm_deployment_doc_covers_release_plan() -> None:
    plan = _load_plan()
    doc = DOC_PATH.read_text()

    assert "make test-wasm" in doc
    assert "make wasm-build" in doc
    assert "tools/check_wasm_release_package.py" in doc
    assert plan["policy"]["package_name"] in doc

    for phase in plan["phases"]:
        assert phase["id"] in doc
        assert phase["status"] in doc

    for target in plan["targets"]:
        assert target["id"] in doc
        assert target["target"] in doc
