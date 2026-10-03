# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WASM runtime priority tests

"""Check maintained WASM plan, documentation and static build wiring."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tools.ci_workflow_inventory import object_mapping

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "requirements" / "wasm_release_plan.toml"
DOC_PATH = ROOT / "docs-site" / "deployment" / "wasm-runtime.md"
MAKEFILE_PATH = ROOT / "Makefile"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "wheels.yml"


def _load_plan() -> dict[str, object]:
    """Read the maintained WASM release plan as a string-keyed mapping.

    Returns
    -------
    dict[str, object]
        Original TOML values, including release phases and target records.

    Raises
    ------
    OSError
        If the maintained release plan cannot be read.
    tomllib.TOMLDecodeError
        If the plan contains invalid TOML.
    ValueError
        If the decoded root is not a string-keyed mapping.
    """
    return object_mapping(tomllib.loads(PLAN_PATH.read_text()))


def test_wasm_plan_points_at_existing_runtime_surface() -> None:
    """Check that release-plan paths resolve to maintained runtime assets."""
    plan = _load_plan()
    policy = object_mapping(plan["policy"])

    crate = policy["crate"]
    assert isinstance(crate, str)
    assert (ROOT / crate).is_dir()
    doc = policy["doc"]
    assert isinstance(doc, str)
    assert (ROOT / doc).is_file()
    workflow = policy["workflow"]
    assert isinstance(workflow, str)
    assert (ROOT / workflow).is_file()
    package_validator = policy["package_validator"]
    assert isinstance(package_validator, str)
    assert (ROOT / package_validator).is_file()
    scope = policy["scope"]
    assert isinstance(scope, str)
    assert "coherence score per token" in scope


def test_wasm_make_targets_and_workflow_are_wired() -> None:
    """Match declared Make targets and build commands to maintained CI files."""
    plan = _load_plan()
    make_text = MAKEFILE_PATH.read_text()
    workflow_text = WORKFLOW_PATH.read_text()

    policy = object_mapping(plan["policy"])
    make_targets = policy["make_targets"]
    assert isinstance(make_targets, list)
    for target in make_targets:
        assert isinstance(target, str)
        assert f"{target}:" in make_text

    assert "build-wasm:" in workflow_text
    assert "wasm-pack build --target web --release" in workflow_text
    assert "wasm-edge-runtime" in workflow_text


def test_wasm_deployment_doc_covers_release_plan() -> None:
    """Require the deployment guide to cover declared phases and targets."""
    plan = _load_plan()
    doc = DOC_PATH.read_text()

    assert "make test-wasm" in doc
    assert "make wasm-build" in doc
    assert "tools/check_wasm_release_package.py" in doc
    policy = object_mapping(plan["policy"])
    package_name = policy["package_name"]
    assert isinstance(package_name, str)
    assert package_name in doc

    phases = plan["phases"]
    assert isinstance(phases, list)
    for record in phases:
        phase = object_mapping(record)
        phase_id = phase["id"]
        assert isinstance(phase_id, str)
        assert phase_id in doc
        phase_status = phase["status"]
        assert isinstance(phase_status, str)
        assert phase_status in doc

    targets = plan["targets"]
    assert isinstance(targets, list)
    for record in targets:
        target = object_mapping(record)
        target_id = target["id"]
        assert isinstance(target_id, str)
        assert target_id in doc
        runtime_target = target["target"]
        assert isinstance(runtime_target, str)
        assert runtime_target in doc
