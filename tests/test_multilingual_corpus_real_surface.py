# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - multilingual corpus real-surface tests
"""Real subprocess coverage for the multilingual corpus validator CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_multilingual_corpus.py"
JsonRow = dict[str, object]


def _run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the production multilingual corpus validator CLI for ``root``."""
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )


def _write_jsonl(path: Path, rows: list[JsonRow]) -> None:
    """Write ``rows`` to a JSONL corpus fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _corpus_row(
    *,
    language: str,
    index: int,
    label: str = "supported",
    expected_decision: str = "allow",
) -> JsonRow:
    """Build a validator-shaped multilingual corpus row."""
    return {
        "id": f"{language}-{index + 1:03d}",
        "language": language,
        "language_name": language,
        "category": "factual_consistency",
        "prompt": "What is the refund window?",
        "source": "Refunds are available within 30 days.",
        "response": "Refunds are available within 30 days.",
        "label": label,
        "expected_decision": expected_decision,
        "risk_tags": ["refund_policy"],
    }


def test_multilingual_corpus_unit_guard_has_real_cli_companion() -> None:
    """The unit guard should be reclassified only with a real CLI companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_multilingual_corpus.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_multilingual_corpus_real_surface.py" in category


def test_multilingual_corpus_cli_accepts_checked_in_fixture() -> None:
    """The production CLI should validate the checked-in multilingual corpus."""
    result = _run_validator(ROOT)

    assert result.returncode == 0
    assert result.stdout == "multilingual_corpus_ok\n"
    assert result.stderr == ""


def test_multilingual_corpus_cli_rejects_missing_language_coverage(
    tmp_path: Path,
) -> None:
    """The production CLI should reject corpora missing required languages."""
    _write_jsonl(
        tmp_path / "benchmarks" / "multilingual_corpus.jsonl",
        [_corpus_row(language="en", index=0)],
    )

    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "benchmarks/multilingual_corpus.jsonl: expected 200 rows, found 1" in (
        result.stderr
    )
    assert (
        "benchmarks/multilingual_corpus.jsonl: language de has 0 rows, expected 25"
        in result.stderr
    )


def test_multilingual_corpus_cli_rejects_invalid_label_decision(
    tmp_path: Path,
) -> None:
    """The production CLI should reject label and decision contradictions."""
    rows: list[JsonRow] = []
    for language in ("en", "de", "fr", "es", "it", "pl", "cs", "nl"):
        for index in range(25):
            if language == "en" and index == 0:
                rows.append(
                    _corpus_row(
                        language=language,
                        index=index,
                        label="contradicted",
                        expected_decision="allow",
                    )
                )
            else:
                rows.append(_corpus_row(language=language, index=index))
    _write_jsonl(tmp_path / "benchmarks" / "multilingual_corpus.jsonl", rows)

    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "benchmarks/multilingual_corpus.jsonl:1: contradicted rows must use "
        "expected_decision=halt"
    ) in result.stderr
