# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Rust kernel extraction plan tests

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "requirements/rust_kernel_extraction_plan.toml"


def _load_toml(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_extraction_plan_matches_workspace_members() -> None:
    plan = _load_toml(PLAN_PATH)
    cargo = _read("backfire-kernel/Cargo.toml")

    crates = plan["crates"]
    assert isinstance(crates, list)
    assert len(crates) == 7

    for crate in crates:
        assert isinstance(crate, dict)
        path = crate["path"]
        name = crate["name"]
        assert isinstance(path, str)
        assert isinstance(name, str)
        assert (ROOT / path).is_dir(), path
        assert f'"crates/{name}"' in cargo


def test_python_extra_and_wheel_workflow_are_bound_to_plan() -> None:
    plan = _load_toml(PLAN_PATH)
    pyproject = _read("pyproject.toml")
    workflow = _read(".github/workflows/wheels.yml")
    versioning = plan["versioning"]

    assert plan["python_extra"] == "rust"
    assert plan["python_module"] == "backfire_kernel"
    assert isinstance(versioning, dict)
    supported_range = versioning["python_supported_range"]
    assert f'"backfire-kernel{supported_range}"' in pyproject
    assert "rust = []" in pyproject
    assert "backfire-kernel/crates/backfire-ffi" in workflow
    assert "PyO3/maturin-action" in workflow
    assert "backfire-kernel/crates/backfire-wasm" in workflow
    assert "pypa/gh-action-pypi-publish" in workflow


def test_docs_cover_each_phase_and_crate() -> None:
    plan = _load_toml(PLAN_PATH)
    doc = _read("docs-site/deployment/rust-kernel-extraction.md")
    runtime_doc = _read("docs-site/guide/runtime-extraction-evaluation.md")
    nav = _read("mkdocs.yml")

    for crate in plan["crates"]:
        assert isinstance(crate, dict)
        assert str(crate["name"]) in doc

    for phase in plan["phases"]:
        assert isinstance(phase, dict)
        assert str(phase["id"]) in doc

    assert "deployment/rust-kernel-extraction.md" in nav
    assert "rust-kernel-extraction.md" in runtime_doc


def test_proof_manifest_maps_existing_theorems_to_runtime_surfaces() -> None:
    plan = _load_toml(PLAN_PATH)
    manifest_path = ROOT / str(plan["proof_manifest"])
    manifest = _load_toml(manifest_path)

    assert (ROOT / str(manifest["model"])).is_file()
    assert (ROOT / str(manifest["properties"])).is_file()
    assert (ROOT / str(manifest["monotonicity"])).is_file()
    assert (ROOT / str(manifest["python_surface"])).is_file()
    assert (ROOT / str(manifest["rust_surface"])).is_dir()

    theorems = manifest["theorems"]
    assert isinstance(theorems, list)
    assert len(theorems) == 7
    for theorem in theorems:
        assert isinstance(theorem, dict)
        # Each theorem is declared in the .lean file the manifest points to.
        source = _read(str(theorem["file"]))
        assert f"theorem {theorem['name']}" in source
        assert str(theorem["contract"])
