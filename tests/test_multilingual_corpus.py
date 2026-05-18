# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - multilingual corpus validation tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_multilingual_corpus.py"
SPEC = importlib.util.spec_from_file_location("validate_multilingual_corpus", VALIDATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_multilingual_corpus = MODULE.validate_multilingual_corpus


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_multilingual_corpus_has_required_language_and_case_coverage() -> None:
    assert validate_multilingual_corpus(ROOT) == []


def test_multilingual_corpus_rejects_missing_language_coverage(tmp_path: Path) -> None:
    rows = [
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
    rows: list[dict[str, object]] = []
    for language in ("en", "de", "fr", "es", "it", "pl", "cs", "nl"):
        for index in range(25):
            label = "contradicted" if language == "en" and index == 0 else "supported"
            rows.append(
                {
                    "id": f"{language}-{index + 1:03d}",
                    "language": language,
                    "language_name": language,
                    "category": "factual_consistency",
                    "prompt": "What is the refund window?",
                    "source": "Refunds are available within 30 days.",
                    "response": "Refunds are available within 30 days.",
                    "label": label,
                    "expected_decision": "allow",
                    "risk_tags": ["refund_policy"],
                }
            )
    _write_jsonl(tmp_path / "benchmarks" / "multilingual_corpus.jsonl", rows)

    errors = validate_multilingual_corpus(tmp_path)

    assert (
        "benchmarks/multilingual_corpus.jsonl:1: contradicted rows must use expected_decision=halt"
        in errors
    )
