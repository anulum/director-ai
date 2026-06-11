# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Backfire kernel release contract tests

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "requirements/backfire_kernel_release.toml"


def _toml(path: str | Path) -> dict[str, object]:
    source = path if isinstance(path, Path) else ROOT / path
    return tomllib.loads(source.read_text(encoding="utf-8"))


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_release_contract_versions_are_aligned() -> None:
    contract = _toml(CONTRACT)
    sources = contract["version_sources"]
    assert isinstance(sources, dict)

    workspace = _toml(str(sources["workspace_manifest"]))
    wheel = _toml(str(sources["python_wheel_manifest"]))
    pyproject = _read(str(sources["python_extra_manifest"]))

    version = contract["current_version"]
    supported_range = contract["python_supported_range"]

    assert workspace["workspace"]["package"]["version"] == version
    assert wheel["project"]["version"] == version
    if "[tool.uv.sources]" in pyproject:
        assert (
            'backfire-kernel = { path = "backfire-kernel/crates/backfire-ffi" }'
            in pyproject
        )
    else:
        assert f"backfire-kernel{supported_range}" in pyproject


def test_all_crates_use_workspace_version_and_licence() -> None:
    plan = _toml("requirements/rust_kernel_extraction_plan.toml")
    for crate in plan["crates"]:
        assert isinstance(crate, dict)
        manifest = _toml(f"{crate['path']}/Cargo.toml")
        package = manifest["package"]
        assert package["version"]["workspace"] is True
        assert package["license"]["workspace"] is True


def test_release_notes_and_ci_entrypoint_are_bound() -> None:
    contract = _toml(CONTRACT)
    version = str(contract["current_version"])
    notes_path = str(contract["release_notes"])
    ci_path = str(contract["crate_ci"])

    notes = _read(notes_path)
    ci = _read(ci_path)

    assert f"## {version}" in notes
    assert "cargo fmt --all -- --check" in ci
    assert "cargo check -p backfire-ffi" in ci
    assert "cargo test --workspace" in ci
    assert "maturin build --release -m crates/backfire-ffi/Cargo.toml" in ci


def test_python_wheel_contract_exports_version_and_core_symbols() -> None:
    contract = _toml(CONTRACT)
    ffi_source = _read("backfire-kernel/crates/backfire-ffi/src/lib.rs")

    assert 'm.add("__version__", env!("CARGO_PKG_VERSION"))?' in ffi_source
    for symbol in contract["wheel_symbols"]:
        assert isinstance(symbol, dict)
        name = str(symbol["name"])
        if name == "__version__":
            continue
        assert name in ffi_source


def test_extraction_plan_points_at_release_contract() -> None:
    plan = _toml("requirements/rust_kernel_extraction_plan.toml")
    contract = _toml(CONTRACT)

    assert plan["status"] == "executed"
    assert plan["release_contract"] == "requirements/backfire_kernel_release.toml"
    assert plan["release_notes"] == contract["release_notes"]
    assert plan["crate_ci"] == contract["crate_ci"]
