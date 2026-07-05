#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - cross-language contract validator

"""Validate Director-AI cross-language contract manifests."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

DEFAULT_MANIFEST = Path("requirements/cross_language_contracts.toml")
EXPECTED_MANIFEST_ID = "cross-language-contracts"
EXPECTED_STATUS = "active"


@dataclass(frozen=True, slots=True)
class BoundaryReport:
    """Validation report for one language or schema boundary.

    Parameters
    ----------
    id:
        Stable boundary identifier from the TOML manifest.
    language:
        Language family inferred from the boundary identifier.
    schema:
        Repository-relative schema contract path.
    implementation:
        Repository-relative implementation path for the boundary.
    generated:
        Optional repository-relative generated artefact path.
    tests:
        Repository-relative test paths that gate the boundary.
    """

    id: str
    language: str
    schema: str
    implementation: str
    generated: str | None
    tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateReport:
    """Validation report for one executable contract gate.

    Parameters
    ----------
    id:
        Stable gate identifier from the TOML manifest.
    command:
        Command operators run to execute the contract gate.
    working_directory:
        Optional repository-relative working directory for the command.
    """

    id: str
    command: str
    working_directory: str | None


@dataclass(frozen=True, slots=True)
class Summary:
    """Count and language summary for a cross-language contract manifest.

    Parameters
    ----------
    boundaries:
        Number of declared contract boundaries.
    gates:
        Number of executable gate declarations.
    required_languages:
        Sorted language families covered by the declared boundaries.
    """

    boundaries: int
    gates: int
    required_languages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Complete cross-language contract validation result.

    Parameters
    ----------
    ok:
        ``True`` when the manifest and every referenced path is valid.
    manifest:
        Repository-relative manifest path.
    summary:
        Boundary, gate, and language coverage summary.
    boundaries:
        Boundary-level reports in manifest order.
    gates:
        Gate-level reports in manifest order.
    errors:
        Fail-closed validation diagnostics.
    """

    ok: bool
    manifest: str
    summary: Summary
    boundaries: tuple[BoundaryReport, ...]
    gates: tuple[GateReport, ...]
    errors: tuple[str, ...]

    def to_json_payload(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible validation payload."""
        return cast(dict[str, object], asdict(self))


def _manifest_label(manifest: Path) -> str:
    return manifest.as_posix()


def _normalise_manifest(root: Path, manifest: Path) -> tuple[Path, Path]:
    if manifest.is_absolute():
        return manifest, manifest
    return root / manifest, manifest


def _load_manifest(path: Path, label: str) -> tuple[dict[str, object], list[str]]:
    if not path.exists():
        return {}, [f"{label}: missing manifest"]
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{label}: invalid TOML: {exc}"]
    return cast(dict[str, object], payload), []


def _string_field(
    table: Mapping[str, object],
    field: str,
    *,
    label: str,
) -> tuple[str | None, list[str]]:
    value = table.get(field)
    if isinstance(value, str) and value.strip():
        return value, []
    return None, [f"{label}: {field} must be a non-empty string"]


def _optional_string_field(
    table: Mapping[str, object],
    field: str,
    *,
    label: str,
) -> tuple[str | None, list[str]]:
    value = table.get(field, "")
    if value == "":
        return None, []
    if isinstance(value, str) and value.strip():
        return value, []
    return None, [f"{label}: {field} must be a string when provided"]


def _string_list_field(
    table: Mapping[str, object],
    field: str,
    *,
    label: str,
) -> tuple[tuple[str, ...], list[str]]:
    value = table.get(field)
    if not isinstance(value, list) or not value:
        return (), [f"{label}: {field} must be a non-empty list"]
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            return (), [f"{label}: {field}[{index}] must be a non-empty string"]
        items.append(item)
    return tuple(items), []


def _table_list_field(
    table: Mapping[str, object],
    field: str,
    *,
    label: str,
) -> tuple[tuple[Mapping[str, object], ...], list[str]]:
    value = table.get(field)
    if not isinstance(value, list) or not value:
        return (), [f"{label}: {field} must be a non-empty array of tables"]
    tables: list[Mapping[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            return (), [f"{label}: {field}[{index}] must be a TOML table"]
        tables.append(cast(Mapping[str, object], item))
    return tuple(tables), []


def _language_from_boundary(boundary_id: str) -> str:
    prefix = boundary_id.split("-", 1)[0].strip()
    return prefix or "unknown"


def _path_exists(root: Path, relative_path: str) -> bool:
    return (root / relative_path).exists()


def _validate_path(
    *,
    root: Path,
    label: str,
    boundary_id: str,
    field: str,
    relative_path: str | None,
    required: bool,
) -> list[str]:
    if relative_path is None:
        if required:
            return [
                f"{label}: boundary {boundary_id} {field} path must be a non-empty string"
            ]
        return []
    if _path_exists(root, relative_path):
        return []
    return [f"{label}: boundary {boundary_id} {field} path missing: {relative_path}"]


def _validate_boundary(
    *,
    root: Path,
    label: str,
    table: Mapping[str, object],
) -> tuple[BoundaryReport | None, list[str]]:
    boundary_id, errors = _string_field(table, "id", label=label)
    schema, schema_errors = _string_field(table, "schema", label=label)
    implementation, implementation_errors = _string_field(
        table,
        "implementation",
        label=label,
    )
    generated, generated_errors = _optional_string_field(
        table,
        "generated",
        label=label,
    )
    tests, tests_errors = _string_list_field(table, "tests", label=label)
    errors.extend(schema_errors)
    errors.extend(implementation_errors)
    errors.extend(generated_errors)
    errors.extend(tests_errors)

    if boundary_id is None or schema is None or implementation is None or not tests:
        return None, errors

    errors.extend(
        _validate_path(
            root=root,
            label=label,
            boundary_id=boundary_id,
            field="schema",
            relative_path=schema,
            required=True,
        )
    )
    errors.extend(
        _validate_path(
            root=root,
            label=label,
            boundary_id=boundary_id,
            field="implementation",
            relative_path=implementation,
            required=True,
        )
    )
    errors.extend(
        _validate_path(
            root=root,
            label=label,
            boundary_id=boundary_id,
            field="generated",
            relative_path=generated,
            required=False,
        )
    )
    for test_path in tests:
        errors.extend(
            _validate_path(
                root=root,
                label=label,
                boundary_id=boundary_id,
                field="test",
                relative_path=test_path,
                required=True,
            )
        )

    return (
        BoundaryReport(
            id=boundary_id,
            language=_language_from_boundary(boundary_id),
            schema=schema,
            implementation=implementation,
            generated=generated,
            tests=tests,
        ),
        errors,
    )


def _validate_gate(
    *,
    root: Path,
    label: str,
    table: Mapping[str, object],
) -> tuple[GateReport | None, list[str]]:
    gate_id, errors = _string_field(table, "id", label=label)
    command, command_errors = _string_field(table, "command", label=label)
    working_directory, working_directory_errors = _optional_string_field(
        table,
        "working_directory",
        label=label,
    )
    errors.extend(command_errors)
    errors.extend(working_directory_errors)
    if gate_id is None or command is None:
        return None, errors
    if working_directory is not None and not (root / working_directory).is_dir():
        errors.append(
            f"{label}: gate {gate_id} working_directory missing: {working_directory}"
        )
    return (
        GateReport(
            id=gate_id,
            command=command,
            working_directory=working_directory,
        ),
        errors,
    )


def _duplicates(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def validate_cross_language_contracts(
    root: Path,
    manifest: Path = DEFAULT_MANIFEST,
) -> ValidationReport:
    """Validate cross-language manifest structure and repository paths.

    Parameters
    ----------
    root:
        Repository root used to resolve manifest references.
    manifest:
        Manifest path, absolute or repository-relative.

    Returns
    -------
    ValidationReport
        Deterministic report with boundary summaries and fail-closed errors.
    """
    root = root.resolve()
    manifest_path, manifest_relative = _normalise_manifest(root, manifest)
    label = _manifest_label(manifest_relative)
    payload, errors = _load_manifest(manifest_path, label)

    boundary_reports: list[BoundaryReport] = []
    gate_reports: list[GateReport] = []
    if payload:
        manifest_id = payload.get("id")
        if manifest_id != EXPECTED_MANIFEST_ID:
            errors.append(
                f"{label}: id must be {EXPECTED_MANIFEST_ID!r}, got {manifest_id!r}"
            )
        status = payload.get("status")
        if status != EXPECTED_STATUS:
            errors.append(f"{label}: status must be {EXPECTED_STATUS!r}")
        roadmap_item = payload.get("roadmap_item")
        if not isinstance(roadmap_item, str) or not roadmap_item.strip():
            errors.append(f"{label}: roadmap_item must be a non-empty string")

        boundaries, boundary_errors = _table_list_field(
            payload,
            "boundaries",
            label=label,
        )
        gates, gate_errors = _table_list_field(payload, "gates", label=label)
        errors.extend(boundary_errors)
        errors.extend(gate_errors)

        for boundary in boundaries:
            boundary_report, boundary_report_errors = _validate_boundary(
                root=root,
                label=label,
                table=boundary,
            )
            errors.extend(boundary_report_errors)
            if boundary_report is not None:
                boundary_reports.append(boundary_report)

        for gate in gates:
            gate_report, gate_report_errors = _validate_gate(
                root=root,
                label=label,
                table=gate,
            )
            errors.extend(gate_report_errors)
            if gate_report is not None:
                gate_reports.append(gate_report)

    boundary_ids = tuple(report.id for report in boundary_reports)
    for duplicate in _duplicates(boundary_ids):
        errors.append(f"{label}: duplicate boundary id {duplicate}")
    gate_ids = tuple(report.id for report in gate_reports)
    for duplicate in _duplicates(gate_ids):
        errors.append(f"{label}: duplicate gate id {duplicate}")

    summary = Summary(
        boundaries=len(boundary_reports),
        gates=len(gate_reports),
        required_languages=tuple(
            sorted({report.language for report in boundary_reports})
        ),
    )
    return ValidationReport(
        ok=not errors,
        manifest=label,
        summary=summary,
        boundaries=tuple(boundary_reports),
        gates=tuple(gate_reports),
        errors=tuple(errors),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate Director-AI cross-language contract manifests.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve manifest references.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest path, absolute or relative to --root.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the cross-language contract manifest validator."""
    args = _build_parser().parse_args(argv)
    report = validate_cross_language_contracts(
        root=cast(Path, args.root),
        manifest=cast(Path, args.manifest),
    )
    if cast(bool, args.json):
        print(
            json.dumps(
                report.to_json_payload(),
                indent=2,
                sort_keys=True,
            )
        )
    elif report.ok:
        print(
            "cross_language_contracts_ok: "
            f"{report.summary.boundaries} boundaries, {report.summary.gates} gates"
        )
    else:
        for error in report.errors:
            print(error, file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
