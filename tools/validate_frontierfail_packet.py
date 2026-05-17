#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - FrontierFail seed packet validator

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

PACKET = Path("benchmarks/frontierfail_seed_packet.toml")
DEFAULT_CASES = Path("benchmarks/frontierfail_cases.jsonl")
SOURCE_TYPES = {"synthetic_regression", "sanitized_production", "public_incident"}
DECISIONS = {"halt", "warn", "allow"}
REQUIRED_ROW_FIELDS = {
    "id",
    "source_type",
    "category",
    "domain",
    "prompt",
    "source",
    "bad_response",
    "expected_failure",
    "expected_decision",
    "evidence_ref",
    "redaction",
    "benchmark_eligible",
}


def _load_packet(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"{PACKET}: missing FrontierFail seed packet"]
    try:
        packet = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{PACKET}: invalid TOML: {exc}"]
    if not isinstance(packet, dict):
        return {}, [f"{PACKET}: packet must be a TOML table"]
    return packet, []


def _load_cases(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{DEFAULT_CASES}: missing FrontierFail cases file"]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(f"{DEFAULT_CASES}:{line_number}: blank lines are not allowed")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{DEFAULT_CASES}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{DEFAULT_CASES}:{line_number}: row must be a JSON object")
            continue
        rows.append(row)
    return rows, errors


def _validate_packet(packet: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "packet_id",
        "cases",
        "public_benchmark_eligible",
        "claim_boundary",
        "minimum_cases",
        "required_categories",
    }
    missing = sorted(required - set(packet))
    if missing:
        return [f"{PACKET}: missing required fields {', '.join(missing)}"]

    if packet["public_benchmark_eligible"] is True:
        errors.append(f"{PACKET}: seed packet must not set public_benchmark_eligible=true")
    if packet["cases"] != DEFAULT_CASES.as_posix():
        errors.append(f"{PACKET}: cases must be {DEFAULT_CASES.as_posix()}")
    if not isinstance(packet["claim_boundary"], str) or "not" not in packet["claim_boundary"].lower():
        errors.append(f"{PACKET}: claim_boundary must explicitly state what the packet is not")
    if not isinstance(packet["minimum_cases"], int) or packet["minimum_cases"] < 1:
        errors.append(f"{PACKET}: minimum_cases must be a positive integer")
    categories = packet["required_categories"]
    if not isinstance(categories, list) or not categories:
        errors.append(f"{PACKET}: required_categories must be a non-empty list")
    elif not all(isinstance(category, str) and category for category in categories):
        errors.append(f"{PACKET}: required_categories must contain non-empty strings")
    return errors


def _non_empty_string(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    return isinstance(value, str) and bool(value.strip())


def _validate_row(row: dict[str, Any], line_number: int) -> list[str]:
    prefix = f"{DEFAULT_CASES}:{line_number}"
    errors: list[str] = []
    missing = sorted(REQUIRED_ROW_FIELDS - set(row))
    if missing:
        return [f"{prefix}: missing required fields {', '.join(missing)}"]

    for field in (
        "id",
        "source_type",
        "category",
        "domain",
        "prompt",
        "source",
        "bad_response",
        "expected_failure",
        "expected_decision",
        "evidence_ref",
        "redaction",
    ):
        if not _non_empty_string(row, field):
            errors.append(f"{prefix}: {field} must be a non-empty string")

    source_type = row.get("source_type")
    if source_type not in SOURCE_TYPES:
        errors.append(f"{prefix}: unsupported source_type {source_type!r}")
    if row.get("expected_decision") not in DECISIONS:
        errors.append(f"{prefix}: unsupported expected_decision {row.get('expected_decision')!r}")
    if row.get("benchmark_eligible") is not False and source_type == "synthetic_regression":
        errors.append(f"{prefix}: synthetic_regression rows must not be benchmark_eligible")
    if source_type in {"sanitized_production", "public_incident"} and row.get("benchmark_eligible") is True:
        evidence_ref = row.get("evidence_ref")
        if not isinstance(evidence_ref, str) or not evidence_ref.startswith(("https://", "report:")):
            errors.append(f"{prefix}: benchmark-eligible sourced rows require public or report evidence")
    if row.get("prompt") == row.get("bad_response"):
        errors.append(f"{prefix}: bad_response must not duplicate prompt")
    if row.get("source") == row.get("bad_response"):
        errors.append(f"{prefix}: bad_response must differ from source")
    return errors


def validate_frontierfail_packet(root: Path) -> list[str]:
    root = root.resolve()
    packet, errors = _load_packet(root / PACKET)
    if errors:
        return errors

    errors.extend(_validate_packet(packet))
    rows, row_errors = _load_cases(root / DEFAULT_CASES)
    errors.extend(row_errors)
    if row_errors:
        return errors

    minimum_cases = packet.get("minimum_cases")
    if isinstance(minimum_cases, int) and len(rows) < minimum_cases:
        errors.append(f"{PACKET}: expected at least {minimum_cases} cases, found {len(rows)}")

    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        row_id = row.get("id")
        if isinstance(row_id, str):
            if row_id in seen_ids:
                errors.append(f"{DEFAULT_CASES}:{line_number}: duplicate id {row_id}")
            seen_ids.add(row_id)
        errors.extend(_validate_row(row, line_number))

    category_counts: Counter[str] = Counter()
    for row in rows:
        category = row.get("category")
        if isinstance(category, str):
            category_counts[category] += 1
    categories = packet.get("required_categories")
    if isinstance(categories, list):
        for category in categories:
            if isinstance(category, str) and category_counts.get(category, 0) == 0:
                errors.append(f"{PACKET}: category {category} has 0 cases")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing benchmarks/frontierfail_seed_packet.toml",
    )
    args = parser.parse_args(argv)

    errors = validate_frontierfail_packet(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("frontierfail_packet_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
