# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Contract tests for ``benchmarks/scores/`` AggreFact dumps + ``SCHEMA.md``.

The score files are named ``factcg-<training-corpus>.json``, which invites
reading the suffix as the *evaluation* dataset — a real reviewer misread
``factcg-fever.json`` as FEVER results when it is the FEVER-trained model scored
on AggreFact. These tests pin the actual invariant (every dump is keyed by
exactly the 11 AggreFact subsets with ``[label, score]`` pairs) and keep
``SCHEMA.md`` honest against the data so the naming can never silently drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_SCORES_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "scores"
_SCHEMA_DOC = _SCORES_DIR / "SCHEMA.md"

AGGREFACT_SUBSETS = frozenset(
    {
        "AggreFact-CNN",
        "AggreFact-XSum",
        "TofuEval-MediaS",
        "TofuEval-MeetB",
        "Wice",
        "Reveal",
        "ClaimVerify",
        "FactCheck-GPT",
        "ExpertQA",
        "Lfqa",
        "RAGTruth",
    }
)


def _score_files() -> list[Path]:
    return sorted(_SCORES_DIR.glob("*.json"))


def test_scores_directory_is_populated() -> None:
    files = _score_files()
    assert files, "no benchmarks/scores/*.json dumps found"
    # the file the reviewer misread must be present and therefore covered below
    assert (_SCORES_DIR / "factcg-fever.json") in files


@pytest.mark.parametrize("path", _score_files(), ids=lambda p: p.name)
def test_score_dump_has_canonical_aggrefact_schema(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path.name} is not a JSON object"
    assert set(data) == set(AGGREFACT_SUBSETS), (
        f"{path.name} keys {sorted(data)} != the 11 AggreFact subsets"
    )
    for subset, rows in data.items():
        assert isinstance(rows, list) and rows, f"{path.name}:{subset} empty/not a list"
        for i, pair in enumerate(rows):
            assert isinstance(pair, list) and len(pair) == 2, (
                f"{path.name}:{subset}[{i}] is not a [label, score] pair"
            )
            label, score = pair
            assert label in (0, 1), (
                f"{path.name}:{subset}[{i}] label {label!r} not in {{0, 1}}"
            )
            assert isinstance(score, (int, float)) and not isinstance(score, bool), (
                f"{path.name}:{subset}[{i}] score {score!r} is not numeric"
            )
            assert 0.0 <= score <= 1.0, (
                f"{path.name}:{subset}[{i}] score {score!r} not in [0, 1]"
            )


def test_fever_dump_is_aggrefact_not_fever() -> None:
    """The exact reviewer misread: factcg-fever.json is AggreFact, carries no FEVER key."""
    data = json.loads((_SCORES_DIR / "factcg-fever.json").read_text(encoding="utf-8"))
    assert set(data) == set(AGGREFACT_SUBSETS)
    assert not any("fever" in key.lower() for key in data), "unexpected FEVER-named key"


def test_schema_doc_documents_every_subset_and_the_naming_trap() -> None:
    assert _SCHEMA_DOC.is_file(), "benchmarks/scores/SCHEMA.md is missing"
    text = _SCHEMA_DOC.read_text(encoding="utf-8")
    for subset in AGGREFACT_SUBSETS:
        assert subset in text, f"SCHEMA.md does not document subset {subset}"
    # the naming-convention warning must be present so the doc prevents the misread
    assert "factcg-fever.json" in text
    assert "training corpus" in text.lower()
