#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Validate the generated backfire-wasm release package."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "director-ai.wasm-release-package.v1"
DEFAULT_PACKAGE_DIR = Path("backfire-kernel/crates/backfire-wasm/pkg")
REQUIRED_FILES = (
    "backfire_wasm_bg.wasm",
    "backfire_wasm.js",
    "backfire_wasm.d.ts",
    "backfire_wasm_bg.wasm.d.ts",
    "package.json",
    "README.md",
    "LICENSE",
)


@dataclass(frozen=True)
class WasmPackageFile:
    """One generated WASM package file with release digest metadata."""

    path: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise file metadata."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class WasmReleasePackageReport:
    """Release-package validation report for backfire-wasm."""

    schema_version: str
    ready: bool
    package_dir: str
    package_name: str
    package_version: str
    package_type: str
    licence: str
    repository: str
    files: tuple[WasmPackageFile, ...]
    blockers: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the validation report."""

        return {
            "schema_version": self.schema_version,
            "ready": self.ready,
            "package_dir": self.package_dir,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "package_type": self.package_type,
            "licence": self.licence,
            "repository": self.repository,
            "files": [file.to_dict() for file in self.files],
            "blockers": [dict(blocker) for blocker in self.blockers],
        }

    def to_markdown(self) -> str:
        """Return a compact operator-readable report."""

        rows = [
            "| File | Size bytes | sha256 |",
            "|---|---:|---|",
        ]
        rows.extend(
            f"| `{file.path}` | {file.size_bytes} | `{file.sha256}` |"
            for file in self.files
        )
        blockers = [
            f"- {blocker['code']} — {blocker['message']}" for blocker in self.blockers
        ]
        if not blockers:
            blockers = ["- none"]
        return "\n".join(
            [
                "# WASM Release Package",
                "",
                f"ready: {str(self.ready).lower()}",
                f"package: {self.package_name} {self.package_version}",
                f"licence: {self.licence}",
                f"repository: {self.repository}",
                "",
                *rows,
                "",
                "## Blockers",
                "",
                *blockers,
                "",
            ]
        )


def validate_wasm_release_package(
    package_dir: str | Path,
) -> WasmReleasePackageReport:
    """Validate generated WASM package metadata and release digests."""

    package_path = Path(package_dir).resolve()
    package_json_path = package_path / "package.json"
    blockers: list[dict[str, str]] = []
    files: list[WasmPackageFile] = []
    if not package_path.is_dir():
        blockers.append(
            _blocker("package_dir_missing", "WASM package directory is missing")
        )
        return _report(package_path, {}, files, blockers)
    metadata: dict[str, Any] = {}
    if not package_json_path.is_file():
        blockers.append(_blocker("package_json_missing", "package.json is missing"))
    else:
        try:
            metadata = json.loads(package_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            blockers.append(
                _blocker(
                    "package_json_invalid",
                    f"package.json is not valid JSON: {exc.msg}",
                )
            )
    for name in REQUIRED_FILES:
        path = package_path / name
        if not path.is_file():
            blockers.append(_blocker("required_file_missing", f"{name} is missing"))
            continue
        files.append(
            WasmPackageFile(
                path=name,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    _validate_metadata(metadata, blockers)
    return _report(package_path, metadata, files, blockers)


def _report(
    package_path: Path,
    metadata: dict[str, Any],
    files: list[WasmPackageFile],
    blockers: list[dict[str, str]],
) -> WasmReleasePackageReport:
    repository = metadata.get("repository", {})
    repository_url = (
        repository.get("url", "") if isinstance(repository, dict) else str(repository)
    )
    return WasmReleasePackageReport(
        schema_version=SCHEMA_VERSION,
        ready=not blockers,
        package_dir=package_path.as_posix(),
        package_name=str(metadata.get("name", "")),
        package_version=str(metadata.get("version", "")),
        package_type=str(metadata.get("type", "")),
        licence=str(metadata.get("license", "")),
        repository=repository_url,
        files=tuple(sorted(files, key=lambda item: item.path)),
        blockers=tuple(blockers),
    )


def _validate_metadata(
    metadata: dict[str, Any],
    blockers: list[dict[str, str]],
) -> None:
    expected = {
        "name": "backfire-wasm",
        "type": "module",
        "license": "AGPL-3.0-or-later",
        "main": "backfire_wasm.js",
        "types": "backfire_wasm.d.ts",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            blockers.append(
                _blocker(
                    "package_metadata_mismatch",
                    f"package.json {key!r} must be {value!r}",
                )
            )
    version = str(metadata.get("version", ""))
    if not version.strip():
        blockers.append(
            _blocker("package_version_missing", "package.json version is missing")
        )
    repository = metadata.get("repository", {})
    repository_url = (
        repository.get("url", "") if isinstance(repository, dict) else str(repository)
    )
    if "github.com/anulum/director-ai" not in repository_url:
        blockers.append(
            _blocker(
                "package_repository_missing",
                "package.json repository must point at anulum/director-ai",
            )
        )
    declared_files = metadata.get("files", [])
    if not isinstance(declared_files, list):
        blockers.append(_blocker("package_files_invalid", "package files is not a list"))
        return
    for name in ("backfire_wasm_bg.wasm", "backfire_wasm.js", "backfire_wasm.d.ts"):
        if name not in declared_files:
            blockers.append(
                _blocker(
                    "package_file_not_declared",
                    f"{name} is not declared in package.json files",
                )
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blocker(code: str, message: str) -> dict[str, str]:
    return {
        "code": code,
        "severity": "error",
        "message": message,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the WASM package validator from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON report")
    args = parser.parse_args(argv)

    report = validate_wasm_release_package(args.package_dir)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(report.to_markdown())
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
