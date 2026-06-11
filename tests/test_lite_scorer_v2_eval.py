# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evaluation runner tests

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "tools" / "eval_lite_scorer_v2.py"

SPEC = importlib.util.spec_from_file_location("eval_lite_scorer_v2", EVALUATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

evaluate_lite_scorer_v2 = MODULE.evaluate_lite_scorer_v2
load_lite_scorer_v2_dataset = MODULE.load_lite_scorer_v2_dataset


class _StaticScorer:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self._index = 0

    def score(self, premise: str, hypothesis: str) -> float:
        assert premise
        assert hypothesis
        score = self._scores[self._index % len(self._scores)]
        self._index += 1
        return score


class _StepClock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        value = self._now
        self._now += 0.002
        return value


def _write_dataset(path: Path) -> None:
    rows = [
        {"premise": "A", "hypothesis": "A", "label": True},
        {"premise": "B", "hypothesis": "not B", "label": False},
        {"premise": "C", "hypothesis": "C", "label": True},
        {"premise": "D", "hypothesis": "not D", "label": False},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_lite_scorer_v2_eval_computes_balanced_accuracy_and_latency(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "heldout.jsonl"
    output = tmp_path / "result.json"
    _write_dataset(dataset)

    result = evaluate_lite_scorer_v2(
        dataset=dataset,
        scorer=_StaticScorer([0.91, 0.12, 0.73, 0.24]),
        threshold=None,
        latency_sample_count=8,
        clock=_StepClock(),
        output=output,
    )

    assert result.rows == 4
    assert result.threshold == 0.5
    assert result.balanced_accuracy == 1.0
    assert result.latency_sample_count == 8
    assert result.latency_p50_ms == 2.0
    assert result.latency_p95_ms == 2.0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["heldout_eval_rows"] == 4
    assert payload["heldout_eval_balanced_accuracy"] == 1.0
    assert payload["latency_p95_ms"] == 2.0


def test_lite_scorer_v2_eval_selects_best_threshold(tmp_path: Path) -> None:
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    result = evaluate_lite_scorer_v2(
        dataset=dataset,
        scorer=_StaticScorer([0.62, 0.58, 0.61, 0.2]),
        threshold=None,
        latency_sample_count=4,
        clock=_StepClock(),
    )

    assert result.threshold == 0.595
    assert result.balanced_accuracy == 1.0


def test_lite_scorer_v2_eval_rejects_single_class_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "heldout.jsonl"
    rows = [
        {"premise": "A", "hypothesis": "A", "label": True},
        {"premise": "B", "hypothesis": "B", "label": True},
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    errors = load_lite_scorer_v2_dataset(dataset)[1]

    assert errors == [f"{dataset}: dataset must contain supported and unsupported rows"]


def test_lite_scorer_v2_eval_rejects_invalid_scores(tmp_path: Path) -> None:
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    try:
        evaluate_lite_scorer_v2(
            dataset=dataset,
            scorer=_StaticScorer([1.2, 0.1, 0.8, 0.3]),
            threshold=0.5,
            latency_sample_count=4,
            clock=_StepClock(),
        )
    except ValueError as exc:
        assert str(exc) == "scorer returned score outside [0, 1]"
    else:
        raise AssertionError("invalid score was accepted")
