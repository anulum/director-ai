# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SimpleQA benchmark harness tests
"""Offline tests for the SimpleQA factual-grounding harness.

Exercises the local JSONL/CSV loaders and ``max_samples`` truncation, the
balanced positive/mismatched-negative pair construction (including the
distinct-answer fallback and empty-field skipping), the ``_first_distinct_answer``
circular scan, and one full dependency-free ``run_simpleqa`` pass that drives the
real scorer loop end-to-end (guarding against the API-drift class that left the
sibling FreshQA harness calling methods that do not exist). No network or NLI
model is touched.
"""

from __future__ import annotations

import csv
import json

import pytest

from benchmarks.e2e_eval import E2EMetrics
from benchmarks.simpleqa_eval import (
    _first_distinct_answer,
    _load_simpleqa,
    build_grounding_pairs,
    run_simpleqa,
)

_RECORDS = [
    {
        "problem": "Who received the IEEE Frank Rosenblatt Award in 2010?",
        "answer": "Michio Sugeno",
    },
    {
        "problem": "In what year was the first Nobel Prize in Physics awarded?",
        "answer": "1901",
    },
    {"problem": "What is the capital of Australia?", "answer": "Canberra"},
]


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return str(path)


def _write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["problem", "answer"])
        writer.writeheader()
        writer.writerows(records)
    return str(path)


class TestLoader:
    def test_loads_jsonl(self, tmp_path):
        src = _write_jsonl(tmp_path / "qa.jsonl", _RECORDS)
        records = _load_simpleqa(source=src)
        assert len(records) == 3
        assert records[0] == _RECORDS[0]

    def test_loads_csv(self, tmp_path):
        src = _write_csv(tmp_path / "qa.csv", _RECORDS)
        records = _load_simpleqa(source=src)
        assert [r["answer"] for r in records] == ["Michio Sugeno", "1901", "Canberra"]

    def test_jsonl_skips_blank_lines(self, tmp_path):
        path = tmp_path / "qa.jsonl"
        path.write_text(
            json.dumps(_RECORDS[0]) + "\n\n" + json.dumps(_RECORDS[1]) + "\n",
            encoding="utf-8",
        )
        assert len(_load_simpleqa(source=str(path))) == 2

    def test_max_samples_truncates(self, tmp_path):
        src = _write_jsonl(tmp_path / "qa.jsonl", _RECORDS)
        assert len(_load_simpleqa(2, source=src)) == 2

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _load_simpleqa(source=str(tmp_path / "absent.jsonl"))


class TestFirstDistinctAnswer:
    def test_returns_first_distinct(self):
        recs = [{"answer": "a"}, {"answer": "a"}, {"answer": "b"}]
        assert _first_distinct_answer(recs, start=1, n=3, gold="a") == "b"

    def test_circular_wrap(self):
        recs = [{"answer": "x"}, {"answer": "y"}, {"answer": "z"}]
        # start past the end wraps around to find a distinct answer
        assert _first_distinct_answer(recs, start=3, n=3, gold="z") == "x"

    def test_all_identical_returns_none(self):
        recs = [{"answer": "same"}] * 3
        assert _first_distinct_answer(recs, start=1, n=3, gold="same") is None

    def test_skips_empty_answers(self):
        recs = [{"answer": "g"}, {"answer": ""}, {"answer": "real"}]
        assert _first_distinct_answer(recs, start=1, n=3, gold="g") == "real"


class TestBuildPairs:
    def test_balanced_positive_and_negative(self):
        pairs = build_grounding_pairs(_RECORDS)
        assert len(pairs) == 6  # 3 records × (1 positive + 1 negative)
        positives = [p for p in pairs if not p[3]]
        negatives = [p for p in pairs if p[3]]
        assert len(positives) == len(negatives) == 3

    def test_positive_response_is_gold(self):
        for _question, gold, response, is_h in build_grounding_pairs(_RECORDS):
            if not is_h:
                assert response == gold

    def test_negative_response_differs_from_gold(self):
        for _question, gold, response, is_h in build_grounding_pairs(_RECORDS):
            if is_h:
                assert response != gold
                assert response  # a real, non-empty SimpleQA answer

    def test_too_few_records_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            build_grounding_pairs(_RECORDS[:1])

    def test_identical_answers_yield_positives_only(self):
        recs = [
            {"problem": "q1", "answer": "same"},
            {"problem": "q2", "answer": "same"},
        ]
        pairs = build_grounding_pairs(recs)
        # No distinct mismatched answer exists → no degenerate negatives.
        assert all(not is_h for *_rest, is_h in pairs)
        assert len(pairs) == 2

    def test_skips_records_with_empty_fields(self):
        recs = [
            {"problem": "", "answer": "orphan"},
            {"problem": "q2", "answer": "a2"},
            {"problem": "q3", "answer": "a3"},
        ]
        pairs = build_grounding_pairs(recs)
        # The empty-question record contributes nothing.
        assert all(question for question, *_rest in pairs)


class TestRunSimpleQA:
    def test_end_to_end_dependency_free(self, tmp_path):
        src = _write_jsonl(tmp_path / "qa.jsonl", _RECORDS)
        metrics = run_simpleqa(source=src, use_nli=False)
        assert isinstance(metrics, E2EMetrics)
        assert metrics.total == 6
        assert all(s.task == "simpleqa" for s in metrics.samples)
        # context carries the grounded gold answer, not the response.
        assert all(s.context for s in metrics.samples)
        assert all(s.latency_ms >= 0.0 for s in metrics.samples)

    def test_to_dict_exposes_guardrail_metrics(self, tmp_path):
        src = _write_jsonl(tmp_path / "qa.jsonl", _RECORDS)
        report = run_simpleqa(source=src, use_nli=False).to_dict()
        for key in ("catch_rate", "false_positive_rate", "precision", "f1", "total"):
            assert key in report
        assert report["total"] == 6

    def test_max_samples_limits_pairs(self, tmp_path):
        src = _write_jsonl(tmp_path / "qa.jsonl", _RECORDS)
        metrics = run_simpleqa(2, source=src, use_nli=False)
        # 2 records → 2 positives + 2 negatives.
        assert metrics.total == 4
