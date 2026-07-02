# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - multilingual corpus validation tests
"""Unit guard for multilingual benchmark corpus validation rules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_multilingual_corpus import main, validate_multilingual_corpus

ROOT = Path(__file__).resolve().parents[1]
JsonRow = dict[str, object]


def _corpus_row(
    *,
    language: str,
    index: int,
    label: str = "supported",
    expected_decision: str = "allow",
    category: str = "factual_consistency",
) -> JsonRow:
    """Build a validator-shaped multilingual corpus row."""
    return {
        "id": f"{language}-{index + 1:03d}",
        "language": language,
        "language_name": language,
        "category": category,
        "prompt": "What is the refund window?",
        "source": "Refunds are available within 30 days.",
        "response": "Refunds are available within 30 days.",
        "label": label,
        "expected_decision": expected_decision,
        "risk_tags": ["refund_policy"],
    }


def _valid_rows() -> list[JsonRow]:
    """Build a complete multilingual corpus fixture."""
    categories = (
        "factual_consistency",
        "numeric_consistency",
        "policy_compliance",
        "temporal_freshness",
        "retrieval_grounding",
    )
    rows: list[JsonRow] = []
    for language in ("en", "de", "fr", "es", "it", "pl", "cs", "nl"):
        for index in range(25):
            rows.append(
                _corpus_row(
                    language=language,
                    index=index,
                    label="contradicted" if index == 0 else "supported",
                    expected_decision="halt" if index == 0 else "allow",
                    category=categories[index % len(categories)],
                )
            )
    return rows


def _write_jsonl(path: Path, rows: list[JsonRow]) -> None:
    """Write benchmark rows to a JSONL fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_multilingual_corpus_has_required_language_and_case_coverage() -> None:
    """The checked-in corpus should satisfy all language and case gates."""
    assert validate_multilingual_corpus(ROOT) == []


def test_multilingual_corpus_rejects_missing_language_coverage(tmp_path: Path) -> None:
    """The validator should reject fixtures without required language coverage."""
    rows: list[JsonRow] = [
        {
            "id": "en-001",
            "language": "en",
            "language_name": "English",
            "category": "factual_consistency",
            "prompt": "What is the refund window?",
            "source": "Refunds are available within 30 days.",
            "response": "Refunds are available within 30 days.",
            "label": "supported",
            "expected_decision": "allow",
            "risk_tags": ["refund_policy"],
        }
    ]
    _write_jsonl(tmp_path / "benchmarks" / "multilingual_corpus.jsonl", rows)

    errors = validate_multilingual_corpus(tmp_path)

    assert "benchmarks/multilingual_corpus.jsonl: expected 200 rows, found 1" in errors
    assert (
        "benchmarks/multilingual_corpus.jsonl: language en has 1 rows, expected 25"
        in errors
    )


def test_multilingual_corpus_rejects_invalid_decision_for_label(tmp_path: Path) -> None:
    """The validator should reject label and decision contradictions."""
    rows = _valid_rows()
    rows[0]["expected_decision"] = "allow"
    _write_jsonl(tmp_path / "benchmarks" / "multilingual_corpus.jsonl", rows)

    errors = validate_multilingual_corpus(tmp_path)

    assert (
        "benchmarks/multilingual_corpus.jsonl:1: contradicted rows must use expected_decision=halt"
        in errors
    )


def test_multilingual_corpus_rejects_missing_or_malformed_file(
    tmp_path: Path,
) -> None:
    """The validator should reject missing files and malformed JSONL rows."""
    assert validate_multilingual_corpus(tmp_path) == [
        "benchmarks/multilingual_corpus.jsonl: missing multilingual corpus"
    ]

    corpus = tmp_path / "benchmarks" / "multilingual_corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("\n{\n[]\n", encoding="utf-8")

    errors = validate_multilingual_corpus(tmp_path)

    assert (
        "benchmarks/multilingual_corpus.jsonl:1: blank lines are not allowed" in errors
    )
    assert "benchmarks/multilingual_corpus.jsonl:2: invalid JSON:" in errors[1]
    assert "benchmarks/multilingual_corpus.jsonl:3: row must be a JSON object" in errors


def test_multilingual_corpus_rejects_missing_required_fields(
    tmp_path: Path,
) -> None:
    """The validator should report missing fields before row-level checks."""
    _write_jsonl(tmp_path / "benchmarks" / "multilingual_corpus.jsonl", [{}])

    errors = validate_multilingual_corpus(tmp_path)

    assert (
        "benchmarks/multilingual_corpus.jsonl:1: missing required fields "
        "category, expected_decision, id, label, language, language_name, prompt, "
        "response, risk_tags, source"
    ) in errors


def test_multilingual_corpus_rejects_row_shape_errors(tmp_path: Path) -> None:
    """The validator should report invalid row values and duplicate IDs."""
    rows = _valid_rows()
    rows[1]["id"] = rows[0]["id"]
    rows[2].update(
        {
            "id": "bad-prefix",
            "language": "en",
            "language_name": "",
            "category": "unknown",
            "prompt": "short",
            "source": "short",
            "response": "short",
            "label": "supported",
            "expected_decision": "halt",
            "risk_tags": [],
        }
    )
    rows[3].update(
        {
            "id": 123,
            "language": 456,
            "label": "unclear",
            "expected_decision": "maybe",
            "risk_tags": [" "],
        }
    )
    _write_jsonl(tmp_path / "benchmarks" / "multilingual_corpus.jsonl", rows)

    errors = validate_multilingual_corpus(tmp_path)

    assert "benchmarks/multilingual_corpus.jsonl:2: duplicate id en-001" in errors
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: language_name must be a non-empty string"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: id must start with language prefix en-"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: unsupported category 'unknown'"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: supported rows must use expected_decision=allow"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: risk_tags must be a non-empty list"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: prompt is too short for benchmark use"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: source is too short for benchmark use"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:3: response is too short for benchmark use"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:4: id must be a non-empty string"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:4: language must be a non-empty string"
        in errors
    )
    assert "benchmarks/multilingual_corpus.jsonl:4: unsupported language 456" in errors
    assert (
        "benchmarks/multilingual_corpus.jsonl:4: unsupported label 'unclear'" in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:4: unsupported expected_decision 'maybe'"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl:4: risk_tags must contain non-empty strings"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl: language en has 24 rows, expected 25"
        in errors
    )


def test_multilingual_corpus_rejects_missing_category_and_label_coverage(
    tmp_path: Path,
) -> None:
    """The validator should require category and label diversity."""
    rows = [_corpus_row(language="en", index=index) for index in range(25)]
    _write_jsonl(tmp_path / "benchmarks" / "multilingual_corpus.jsonl", rows)

    errors = validate_multilingual_corpus(tmp_path)

    assert (
        "benchmarks/multilingual_corpus.jsonl: missing category coverage for numeric_consistency"
        in errors
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl: missing label coverage for contradicted"
        in errors
    )


def test_multilingual_corpus_main_reports_success_and_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The script entrypoint should print operator-readable results."""
    assert main([str(ROOT)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "multilingual_corpus_ok\n"
    assert captured.err == ""

    assert main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == "benchmarks/multilingual_corpus.jsonl: missing multilingual corpus\n"
    )
