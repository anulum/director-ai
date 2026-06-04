# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WASM release package tests

from __future__ import annotations

import json

from tools.check_wasm_release_package import (
    REQUIRED_FILES,
    main,
    validate_wasm_release_package,
)


def _write_package(package_dir) -> None:
    package_dir.mkdir(parents=True)
    metadata = {
        "name": "backfire-wasm",
        "type": "module",
        "description": "WebAssembly bindings",
        "version": "0.1.1",
        "license": "AGPL-3.0-or-later",
        "repository": {
            "type": "git",
            "url": "https://github.com/anulum/director-ai",
        },
        "files": [
            "backfire_wasm_bg.wasm",
            "backfire_wasm.js",
            "backfire_wasm.d.ts",
        ],
        "main": "backfire_wasm.js",
        "types": "backfire_wasm.d.ts",
    }
    (package_dir / "package.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name in REQUIRED_FILES:
        if name == "package.json":
            continue
        (package_dir / name).write_bytes(f"{name}\n".encode())


def test_wasm_release_package_accepts_complete_package(tmp_path) -> None:
    package_dir = tmp_path / "pkg"
    _write_package(package_dir)

    report = validate_wasm_release_package(package_dir)

    assert report.ready is True
    assert report.package_name == "backfire-wasm"
    assert report.package_type == "module"
    assert report.licence == "AGPL-3.0-or-later"
    assert report.repository == "https://github.com/anulum/director-ai"
    assert len(report.files) == len(REQUIRED_FILES)
    assert all(len(file.sha256) == 64 for file in report.files)
    assert report.blockers == ()


def test_wasm_release_package_blocks_missing_generated_files(tmp_path) -> None:
    package_dir = tmp_path / "pkg"
    _write_package(package_dir)
    (package_dir / "backfire_wasm_bg.wasm").unlink()

    report = validate_wasm_release_package(package_dir)

    assert report.ready is False
    assert any(
        blocker["code"] == "required_file_missing" for blocker in report.blockers
    )


def test_wasm_release_package_blocks_metadata_drift(tmp_path) -> None:
    package_dir = tmp_path / "pkg"
    _write_package(package_dir)
    metadata = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
    metadata["repository"] = {"url": "https://example.invalid/repo"}
    metadata["files"] = ["backfire_wasm.js"]
    (package_dir / "package.json").write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_wasm_release_package(package_dir)
    codes = {blocker["code"] for blocker in report.blockers}

    assert report.ready is False
    assert "package_repository_missing" in codes
    assert "package_file_not_declared" in codes


def test_wasm_release_package_cli_writes_json(tmp_path) -> None:
    package_dir = tmp_path / "pkg"
    _write_package(package_dir)
    output = tmp_path / "report.json"

    exit_code = main(["--package-dir", str(package_dir), "--json", str(output)])

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["schema_version"] == "director-ai.wasm-release-package.v1"
    assert payload["ready"] is True
