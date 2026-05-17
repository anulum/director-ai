#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - multilingual corpus validator

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

CORPUS = Path("benchmarks/multilingual_corpus.jsonl")
EXPECTED_LANGUAGES = ("en", "de", "fr", "es", "it", "pl", "cs", "nl")
EXPECTED_ROWS_PER_LANGUAGE = 25
EXPECTED_TOTAL_ROWS = len(EXPECTED_LANGUAGES) * EXPECTED_ROWS_PER_LANGUAGE
EXPECTED_CATEGORIES = {
    "factual_consistency",
    "numeric_consistency",
    "policy_compliance",
    "temporal_freshness",
    "retrieval_grounding",
}
LABELS = {"supported", "contradicted"}
DECISIONS = {"allow", "halt"}
REQUIRED_FIELDS = {
    "id",
    "language",
    "language_name",
    "category",
    "prompt",
    "source",
    "response",
    "label",
    "expected_decision",
    "risk_tags",
}


def _load_rows(corpus_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    if not corpus_path.exists():
        return [], [f"{CORPUS}: missing multilingual corpus"]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(corpus_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(f"{CORPUS}:{line_number}: blank lines are not allowed")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{CORPUS}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"{CORPUS}:{line_number}: row must be a JSON object")
            continue
        rows.append(row)
    return rows, errors


def _non_empty_string(row: dict[str, Any], field: str) -> bool:
    value = row.get(field)
    return isinstance(value, str) and bool(value.strip())


def _validate_row(row: dict[str, Any], line_number: int) -> list[str]:
    prefix = f"{CORPUS}:{line_number}"
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        return [f"{prefix}: missing required fields {', '.join(missing)}"]

    for field in (
        "id",
        "language",
        "language_name",
        "category",
        "prompt",
        "source",
        "response",
        "label",
        "expected_decision",
    ):
        if not _non_empty_string(row, field):
            errors.append(f"{prefix}: {field} must be a non-empty string")

    language = row.get("language")
    if language not in EXPECTED_LANGUAGES:
        errors.append(f"{prefix}: unsupported language {language!r}")

    row_id = row.get("id")
    if isinstance(row_id, str) and isinstance(language, str) and not row_id.startswith(f"{language}-"):
        errors.append(f"{prefix}: id must start with language prefix {language}-")

    category = row.get("category")
    if category not in EXPECTED_CATEGORIES:
        errors.append(f"{prefix}: unsupported category {category!r}")

    label = row.get("label")
    if label not in LABELS:
        errors.append(f"{prefix}: unsupported label {label!r}")

    decision = row.get("expected_decision")
    if decision not in DECISIONS:
        errors.append(f"{prefix}: unsupported expected_decision {decision!r}")
    if label == "supported" and decision != "allow":
        errors.append(f"{prefix}: supported rows must use expected_decision=allow")
    if label == "contradicted" and decision != "halt":
        errors.append(f"{prefix}: contradicted rows must use expected_decision=halt")

    risk_tags = row.get("risk_tags")
    if not isinstance(risk_tags, list) or not risk_tags:
        errors.append(f"{prefix}: risk_tags must be a non-empty list")
    elif not all(isinstance(tag, str) and tag.strip() for tag in risk_tags):
        errors.append(f"{prefix}: risk_tags must contain non-empty strings")

    for text_field in ("prompt", "source", "response"):
        value = row.get(text_field)
        if isinstance(value, str) and len(value.strip()) < 12:
            errors.append(f"{prefix}: {text_field} is too short for benchmark use")

    return errors


def _count_values(rows: Iterable[dict[str, Any]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(field)
        if isinstance(value, str):
            counts[value] += 1
    return counts


def validate_multilingual_corpus(root: Path) -> list[str]:
    rows, errors = _load_rows(root.resolve() / CORPUS)
    if errors:
        return errors

    if len(rows) != EXPECTED_TOTAL_ROWS:
        errors.append(f"{CORPUS}: expected {EXPECTED_TOTAL_ROWS} rows, found {len(rows)}")

    seen_ids: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        row_id = row.get("id")
        if isinstance(row_id, str):
            if row_id in seen_ids:
                errors.append(f"{CORPUS}:{line_number}: duplicate id {row_id}")
            seen_ids.add(row_id)
        errors.extend(_validate_row(row, line_number))

    language_counts = _count_values(rows, "language")
    for language in EXPECTED_LANGUAGES:
        count = language_counts.get(language, 0)
        if count != EXPECTED_ROWS_PER_LANGUAGE:
            errors.append(
                f"{CORPUS}: language {language} has {count} rows, expected {EXPECTED_ROWS_PER_LANGUAGE}"
            )

    category_counts = _count_values(rows, "category")
    for category in sorted(EXPECTED_CATEGORIES):
        if category_counts.get(category, 0) == 0:
            errors.append(f"{CORPUS}: missing category coverage for {category}")

    label_counts = _count_values(rows, "label")
    for label in sorted(LABELS):
        if label_counts.get(label, 0) == 0:
            errors.append(f"{CORPUS}: missing label coverage for {label}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing benchmarks/multilingual_corpus.jsonl",
    )
    args = parser.parse_args(argv)

    errors = validate_multilingual_corpus(args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("multilingual_corpus_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
