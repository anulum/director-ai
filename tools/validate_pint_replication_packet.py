#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - PINT replication packet validator

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

PACKET = Path("benchmarks/pint_replication_packet.toml")
DEFAULT_CASES = Path("benchmarks/pint_seed_cases.jsonl")
DETECTOR_CONTRACTS = {"text_to_boolean"}
SOURCE_TYPES = {"synthetic_seed", "official_pint_export"}
LANGUAGE_TAG_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$")
MALICIOUS_CATEGORIES = {
    "direct_injection",
    "indirect_injection",
    "jailbreak",
    "encoding_trick",
}
BENIGN_CATEGORIES = {"benign_hard_negative"}
REQUIRED_PACKET_FIELDS = {
    "schema_version",
    "packet_id",
    "upstream_repository",
    "upstream_blog",
    "seed_cases",
    "public_score_claim",
    "claim_boundary",
    "detector_contract",
    "minimum_seed_cases",
    "minimum_languages",
    "minimum_malicious_languages",
    "minimum_cases_per_required_category",
    "required_categories",
}
REQUIRED_ROW_FIELDS = {
    "id",
    "category",
    "language",
    "input",
    "expected_injection",
    "source_type",
    "benchmark_eligible",
    "notes",
}


def _load_packet(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"{PACKET}: missing PINT replication packet"]
    try:
        packet = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{PACKET}: invalid TOML: {exc}"]
    if not isinstance(packet, dict):
        return {}, [f"{PACKET}: packet must be a TOML table"]
    return packet, []


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not path.exists():
        return [], [f"{DEFAULT_CASES}: missing PINT seed cases"]
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
    missing = sorted(REQUIRED_PACKET_FIELDS - set(packet))
    if missing:
        return [f"{PACKET}: missing required fields {', '.join(missing)}"]

    errors: list[str] = []
    if packet["public_score_claim"] is True:
        errors.append(
            f"{PACKET}: seed replication packet must not set public_score_claim=true"
        )
    if packet["seed_cases"] != DEFAULT_CASES.as_posix():
        errors.append(f"{PACKET}: seed_cases must be {DEFAULT_CASES.as_posix()}")
    if packet["detector_contract"] not in DETECTOR_CONTRACTS:
        errors.append(
            f"{PACKET}: unsupported detector_contract {packet['detector_contract']!r}"
        )
    if (
        not isinstance(packet["minimum_seed_cases"], int)
        or packet["minimum_seed_cases"] < 1
    ):
        errors.append(f"{PACKET}: minimum_seed_cases must be a positive integer")
    if (
        not isinstance(packet["minimum_languages"], int)
        or packet["minimum_languages"] < 1
    ):
        errors.append(f"{PACKET}: minimum_languages must be a positive integer")
    if (
        not isinstance(packet["minimum_malicious_languages"], int)
        or packet["minimum_malicious_languages"] < 1
    ):
        errors.append(
            f"{PACKET}: minimum_malicious_languages must be a positive integer"
        )
    if (
        not isinstance(packet["minimum_cases_per_required_category"], int)
        or packet["minimum_cases_per_required_category"] < 1
    ):
        errors.append(
            f"{PACKET}: minimum_cases_per_required_category must be a positive integer"
        )

    for field in ("upstream_repository", "upstream_blog"):
        value = packet[field]
        if not isinstance(value, str) or not value.startswith("https://"):
            errors.append(f"{PACKET}: {field} must be an HTTPS URL")

    boundary = packet["claim_boundary"]
    if (
        not isinstance(boundary, str)
        or "not" not in boundary.lower()
        or "official" not in boundary.lower()
    ):
        errors.append(
            f"{PACKET}: claim_boundary must state this is not an official score"
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


def _validate_row(row: dict[str, Any], line_number: int) -> list[str]:
    prefix = f"{DEFAULT_CASES}:{line_number}"
    missing = sorted(REQUIRED_ROW_FIELDS - set(row))
    if missing:
        return [f"{prefix}: missing required fields {', '.join(missing)}"]

    errors: list[str] = []
    for field in ("id", "category", "language", "input", "source_type", "notes"):
        if not _non_empty_string(row, field):
            errors.append(f"{prefix}: {field} must be a non-empty string")
    language = row.get("language")
    if isinstance(language, str) and not LANGUAGE_TAG_RE.fullmatch(language):
        errors.append(f"{prefix}: language must be a canonical lowercase BCP-47 tag")
    if row.get("source_type") not in SOURCE_TYPES:
        errors.append(f"{prefix}: unsupported source_type {row.get('source_type')!r}")
    if not isinstance(row.get("expected_injection"), bool):
        errors.append(f"{prefix}: expected_injection must be boolean")
    elif (
        row.get("category") in MALICIOUS_CATEGORIES
        and row.get("expected_injection") is not True
    ):
        errors.append(
            f"{prefix}: category {row['category']} must be labelled expected_injection=true"
        )
    elif (
        row.get("category") in BENIGN_CATEGORIES
        and row.get("expected_injection") is not False
    ):
        errors.append(
            f"{prefix}: category {row['category']} must be labelled expected_injection=false"
        )
    if not isinstance(row.get("benchmark_eligible"), bool):
        errors.append(f"{prefix}: benchmark_eligible must be boolean")
    if (
        row.get("source_type") == "synthetic_seed"
        and row.get("benchmark_eligible") is not False
    ):
        errors.append(f"{prefix}: synthetic_seed rows must not be benchmark_eligible")
    if (
        row.get("source_type") == "official_pint_export"
        and row.get("benchmark_eligible") is True
    ):
        errors.append(
            f"{prefix}: official exports require separate private validation evidence"
        )
    value = row.get("input")
    if isinstance(value, str) and len(value.strip()) < 20:
        errors.append(f"{prefix}: input is too short for injection evaluation")
    return errors


def validate_pint_replication_packet(root: Path) -> list[str]:
    root = root.resolve()
    packet, errors = _load_packet(root / PACKET)
    if errors:
        return errors
    errors.extend(_validate_packet(packet))

    rows, row_errors = _load_rows(root / DEFAULT_CASES)
    errors.extend(row_errors)
    if row_errors:
        return errors

    minimum = packet.get("minimum_seed_cases")
    if isinstance(minimum, int) and len(rows) < minimum:
        errors.append(
            f"{PACKET}: expected at least {minimum} seed cases, found {len(rows)}"
        )

    seen_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    malicious_language_counts: Counter[str] = Counter()
    label_counts: Counter[bool] = Counter()
    for line_number, row in enumerate(rows, 1):
        row_id = row.get("id")
        if isinstance(row_id, str):
            if row_id in seen_ids:
                errors.append(f"{DEFAULT_CASES}:{line_number}: duplicate id {row_id}")
            seen_ids.add(row_id)
        category = row.get("category")
        if isinstance(category, str):
            category_counts[category] += 1
        language = row.get("language")
        if isinstance(language, str):
            language_counts[language] += 1
            if row.get("expected_injection") is True:
                malicious_language_counts[language] += 1
        expected = row.get("expected_injection")
        if isinstance(expected, bool):
            label_counts[expected] += 1
        errors.extend(_validate_row(row, line_number))

    categories = packet.get("required_categories")
    minimum_per_category = packet.get("minimum_cases_per_required_category")
    if isinstance(categories, list):
        for category in categories:
            if not isinstance(category, str):
                continue
            count = category_counts.get(category, 0)
            if count == 0:
                errors.append(f"{PACKET}: category {category} has 0 seed cases")
            elif isinstance(minimum_per_category, int) and count < minimum_per_category:
                errors.append(
                    f"{PACKET}: category {category} has {count} seed cases, "
                    f"expected at least {minimum_per_category}"
                )
    for expected in (True, False):
        if label_counts[expected] == 0:
            errors.append(f"{PACKET}: missing expected_injection={expected} coverage")
    minimum_languages = packet.get("minimum_languages")
    if isinstance(minimum_languages, int) and len(language_counts) < minimum_languages:
        errors.append(
            f"{PACKET}: expected at least {minimum_languages} languages, "
            f"found {len(language_counts)}"
        )
    minimum_malicious_languages = packet.get("minimum_malicious_languages")
    if (
        isinstance(minimum_malicious_languages, int)
        and len(malicious_language_counts) < minimum_malicious_languages
    ):
        errors.append(
            f"{PACKET}: expected at least {minimum_malicious_languages} "
            f"malicious languages, found {len(malicious_language_counts)}"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing benchmarks/pint_replication_packet.toml",
    )
    args = parser.parse_args(argv)

    errors = validate_pint_replication_packet(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("pint_replication_packet_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
