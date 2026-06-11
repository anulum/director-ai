#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - FrontierFail seed packet validator

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

PACKET = Path("benchmarks/frontierfail_seed_packet.toml")
DEFAULT_CASES = Path("benchmarks/frontierfail_cases.jsonl")
SOURCE_TYPES = {"synthetic_regression", "sanitized_production", "public_incident"}
DECISIONS = {"halt", "warn", "allow"}
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
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
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
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
        "minimum_public_incident_cases",
        "minimum_public_incident_categories",
        "minimum_public_incident_domains",
        "minimum_public_incident_publishers",
        "minimum_public_incident_evidence_refs",
        "required_categories",
    }
    missing = sorted(required - set(packet))
    if missing:
        return [f"{PACKET}: missing required fields {', '.join(missing)}"]

    if packet["public_benchmark_eligible"] is True:
        errors.append(
            f"{PACKET}: seed packet must not set public_benchmark_eligible=true"
        )
    if packet["cases"] != DEFAULT_CASES.as_posix():
        errors.append(f"{PACKET}: cases must be {DEFAULT_CASES.as_posix()}")
    if (
        not isinstance(packet["claim_boundary"], str)
        or "not" not in packet["claim_boundary"].lower()
    ):
        errors.append(
            f"{PACKET}: claim_boundary must explicitly state what the packet is not"
        )
    if not isinstance(packet["minimum_cases"], int) or packet["minimum_cases"] < 1:
        errors.append(f"{PACKET}: minimum_cases must be a positive integer")
    if (
        not isinstance(packet["minimum_public_incident_cases"], int)
        or packet["minimum_public_incident_cases"] < 0
    ):
        errors.append(
            f"{PACKET}: minimum_public_incident_cases must be a non-negative integer"
        )
    if (
        not isinstance(packet["minimum_public_incident_categories"], int)
        or packet["minimum_public_incident_categories"] < 0
    ):
        errors.append(
            f"{PACKET}: minimum_public_incident_categories must be a non-negative integer"
        )
    if (
        not isinstance(packet["minimum_public_incident_domains"], int)
        or packet["minimum_public_incident_domains"] < 0
    ):
        errors.append(
            f"{PACKET}: minimum_public_incident_domains must be a non-negative integer"
        )
    if (
        not isinstance(packet["minimum_public_incident_publishers"], int)
        or packet["minimum_public_incident_publishers"] < 0
    ):
        errors.append(
            f"{PACKET}: minimum_public_incident_publishers must be a non-negative integer"
        )
    if (
        not isinstance(packet["minimum_public_incident_evidence_refs"], int)
        or packet["minimum_public_incident_evidence_refs"] < 0
    ):
        errors.append(
            f"{PACKET}: minimum_public_incident_evidence_refs must be a non-negative integer"
        )
    categories = packet["required_categories"]
    if not isinstance(categories, list) or not categories:
        errors.append(f"{PACKET}: required_categories must be a non-empty list")
    elif not all(isinstance(category, str) and category for category in categories):
        errors.append(f"{PACKET}: required_categories must contain non-empty strings")
    return errors


def _non_empty_string(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    return isinstance(value, str) and bool(value.strip())


def _is_iso_calendar_date(value: str) -> bool:
    if not ISO_DATE_RE.fullmatch(value):
        return False
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return parsed <= date.today()


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
        errors.append(
            f"{prefix}: unsupported expected_decision {row.get('expected_decision')!r}"
        )
    if (
        row.get("benchmark_eligible") is not False
        and source_type == "synthetic_regression"
    ):
        errors.append(
            f"{prefix}: synthetic_regression rows must not be benchmark_eligible"
        )
    if (
        source_type in {"sanitized_production", "public_incident"}
        and row.get("benchmark_eligible") is True
    ):
        evidence_ref = row.get("evidence_ref")
        if not isinstance(evidence_ref, str) or not evidence_ref.startswith(
            ("https://", "report:")
        ):
            errors.append(
                f"{prefix}: benchmark-eligible sourced rows require public or report evidence"
            )
        if source_type == "public_incident":
            for field in ("evidence_publisher", "evidence_title"):
                if not _non_empty_string(row, field):
                    errors.append(
                        f"{prefix}: public_incident benchmark-eligible rows require {field}"
                    )
            accessed_date = row.get("evidence_accessed_date")
            if not isinstance(accessed_date, str) or not _is_iso_calendar_date(
                accessed_date
            ):
                errors.append(
                    f"{prefix}: public_incident benchmark-eligible rows require evidence_accessed_date"
                )
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
        errors.append(
            f"{PACKET}: expected at least {minimum_cases} cases, found {len(rows)}"
        )
    minimum_public = packet.get("minimum_public_incident_cases")
    if isinstance(minimum_public, int):
        public_rows = [
            row
            for row in rows
            if row.get("source_type") == "public_incident"
            and row.get("benchmark_eligible") is True
        ]
        public_count = len(public_rows)
        if public_count < minimum_public:
            errors.append(
                f"{PACKET}: expected at least {minimum_public} "
                f"public_incident benchmark-eligible cases, found {public_count}"
            )
        minimum_public_categories = packet.get("minimum_public_incident_categories")
        if isinstance(minimum_public_categories, int):
            public_categories = {
                category
                for row in public_rows
                if isinstance(category := row.get("category"), str) and category
            }
            public_category_count = len(public_categories)
            if public_category_count < minimum_public_categories:
                errors.append(
                    f"{PACKET}: expected at least {minimum_public_categories} "
                    "public_incident benchmark-eligible categories, "
                    f"found {public_category_count}"
                )
        minimum_public_domains = packet.get("minimum_public_incident_domains")
        if isinstance(minimum_public_domains, int):
            public_domains = {
                domain
                for row in public_rows
                if isinstance(domain := row.get("domain"), str) and domain
            }
            public_domain_count = len(public_domains)
            if public_domain_count < minimum_public_domains:
                errors.append(
                    f"{PACKET}: expected at least {minimum_public_domains} "
                    "public_incident benchmark-eligible domains, "
                    f"found {public_domain_count}"
                )
        minimum_public_publishers = packet.get("minimum_public_incident_publishers")
        if isinstance(minimum_public_publishers, int):
            public_publishers = {
                publisher.strip()
                for row in public_rows
                if isinstance(publisher := row.get("evidence_publisher"), str)
                and publisher.strip()
            }
            public_publisher_count = len(public_publishers)
            if public_publisher_count < minimum_public_publishers:
                errors.append(
                    f"{PACKET}: expected at least {minimum_public_publishers} "
                    "public_incident publishers, "
                    f"found {public_publisher_count}"
                )
        minimum_public_evidence_refs = packet.get(
            "minimum_public_incident_evidence_refs"
        )
        if isinstance(minimum_public_evidence_refs, int):
            public_evidence_refs = {
                evidence_ref.strip()
                for row in public_rows
                if isinstance(evidence_ref := row.get("evidence_ref"), str)
                and evidence_ref.strip()
            }
            public_evidence_ref_count = len(public_evidence_refs)
            if public_evidence_ref_count < minimum_public_evidence_refs:
                errors.append(
                    f"{PACKET}: expected at least {minimum_public_evidence_refs} "
                    "public_incident evidence refs, "
                    f"found {public_evidence_ref_count}"
                )

    seen_ids: set[str] = set()
    public_evidence_refs: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        row_id = row.get("id")
        if isinstance(row_id, str):
            if row_id in seen_ids:
                errors.append(f"{DEFAULT_CASES}:{line_number}: duplicate id {row_id}")
            seen_ids.add(row_id)
        evidence_ref = row.get("evidence_ref")
        if (
            row.get("source_type") == "public_incident"
            and row.get("benchmark_eligible") is True
            and isinstance(evidence_ref, str)
        ):
            if evidence_ref in public_evidence_refs:
                errors.append(
                    f"{DEFAULT_CASES}:{line_number}: duplicate public_incident evidence_ref {evidence_ref}"
                )
            public_evidence_refs.add(evidence_ref)
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
