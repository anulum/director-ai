# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evaluation tests
"""Unit guard for the Lite Scorer v2 held-out evaluator."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tools import eval_lite_scorer_v2 as evaluator
from tools.eval_lite_scorer_v2 import (
    EvalRow,
    LiteScorerV2EvalResult,
    evaluate_lite_scorer_v2,
    load_lite_scorer_v2_dataset,
)


class _StaticScorer:
    """Deterministic scorer that returns configured scores in order."""

    def __init__(self, scores: list[float]) -> None:
        """Create a scorer with a cyclic score stream."""
        self._scores = scores
        self._index = 0

    def score(self, premise: str, hypothesis: str) -> float:
        """Return the next configured score for non-empty text pairs."""
        assert premise
        assert hypothesis
        score = self._scores[self._index % len(self._scores)]
        self._index += 1
        return score


class _StepClock:
    """Clock that advances by a fixed number of seconds on every call."""

    def __init__(self, *, step_seconds: float = 0.002) -> None:
        """Create a monotonic clock with ``step_seconds`` increments."""
        self._now = 0.0
        self._step_seconds = step_seconds

    def __call__(self) -> float:
        """Return the current timestamp and advance for the next call."""
        value = self._now
        self._now += self._step_seconds
        return value


class _FrozenClock:
    """Clock that never advances."""

    def __call__(self) -> float:
        """Return the same timestamp for every observation."""
        return 1.0


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows to ``path``."""
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_dataset(path: Path) -> None:
    """Write a balanced held-out dataset fixture."""
    _write_jsonl(
        path,
        [
            {"premise": "A", "hypothesis": "A", "label": True},
            {"premise": "B", "hypothesis": "not B", "label": False},
            {"premise": "C", "hypothesis": "C", "label": True},
            {"premise": "D", "hypothesis": "not D", "label": False},
        ],
    )


def _evaluate(
    dataset: Path,
    scores: list[float],
    *,
    threshold: float | None = 0.5,
    latency_sample_count: int = 4,
    clock: Callable[[], float] | None = None,
    output: Path | None = None,
) -> LiteScorerV2EvalResult:
    """Evaluate ``dataset`` with a deterministic scorer."""
    return evaluate_lite_scorer_v2(
        dataset=dataset,
        scorer=_StaticScorer(scores),
        threshold=threshold,
        latency_sample_count=latency_sample_count,
        clock=clock or _StepClock(),
        output=output,
    )


def test_lite_scorer_v2_dataset_accepts_supported_label_variants(
    tmp_path: Path,
) -> None:
    """The dataset loader should accept all supported label encodings."""
    dataset = tmp_path / "heldout.jsonl"
    _write_jsonl(
        dataset,
        [
            {"premise": "p1", "hypothesis": "h1", "label": True},
            {"premise": "p2", "hypothesis": "h2", "label": 1},
            {"premise": "p3", "hypothesis": "h3", "label": "supported"},
            {"premise": "p4", "hypothesis": "h4", "label": "entailment"},
            {"premise": "p5", "hypothesis": "h5", "label": "true"},
            {"premise": "p6", "hypothesis": "h6", "label": "1"},
            {"premise": "n1", "hypothesis": "nh1", "label": False},
            {"premise": "n2", "hypothesis": "nh2", "label": 0},
            {"premise": "n3", "hypothesis": "nh3", "label": "unsupported"},
            {"premise": "n4", "hypothesis": "nh4", "label": "contradiction"},
            {"premise": "n5", "hypothesis": "nh5", "label": "false"},
            {"premise": "n6", "hypothesis": "nh6", "label": "0"},
        ],
    )

    rows, errors = load_lite_scorer_v2_dataset(dataset)

    assert errors == []
    assert [row.label for row in rows] == [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_lite_scorer_v2_dataset_reports_malformed_rows(tmp_path: Path) -> None:
    """Malformed JSONL rows should produce row-specific diagnostics."""
    missing = tmp_path / "missing.jsonl"
    assert load_lite_scorer_v2_dataset(missing) == (
        [],
        [f"{missing}: dataset does not exist"],
    )

    dataset = tmp_path / "heldout.jsonl"
    dataset.write_text(
        "\n".join(
            [
                "",
                "{",
                "[]",
                json.dumps(
                    {
                        "premise": " ",
                        "hypothesis": "",
                        "label": "unknown",
                    }
                ),
                json.dumps(
                    {
                        "premise": "numeric label premise",
                        "hypothesis": "numeric label hypothesis",
                        "label": 0.5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows, errors = load_lite_scorer_v2_dataset(dataset)

    assert rows == []
    assert f"{dataset}:1: blank lines are not allowed" in errors
    assert any(f"{dataset}:2: invalid JSON:" in error for error in errors)
    assert f"{dataset}:3: row must be a JSON object" in errors
    assert f"{dataset}:4: premise must be a non-empty string" in errors
    assert f"{dataset}:4: hypothesis must be a non-empty string" in errors
    assert (f"{dataset}:4: label must be boolean or supported/unsupported") in errors
    assert (f"{dataset}:5: label must be boolean or supported/unsupported") in errors


def test_lite_scorer_v2_dataset_rejects_empty_and_single_class_data(
    tmp_path: Path,
) -> None:
    """Datasets must be non-empty and include both supported classes."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert load_lite_scorer_v2_dataset(empty) == (
        [],
        [f"{empty}: dataset must contain at least one row"],
    )

    single_class = tmp_path / "single_class.jsonl"
    _write_jsonl(
        single_class,
        [
            {"premise": "A", "hypothesis": "A", "label": True},
            {"premise": "B", "hypothesis": "B", "label": True},
        ],
    )

    assert load_lite_scorer_v2_dataset(single_class) == (
        [
            EvalRow(premise="A", hypothesis="A", label=True),
            EvalRow(premise="B", hypothesis="B", label=True),
        ],
        [f"{single_class}: dataset must contain supported and unsupported rows"],
    )


def test_lite_scorer_v2_eval_computes_balanced_accuracy_and_latency(
    tmp_path: Path,
) -> None:
    """Evaluation should write complete score and latency metadata."""
    dataset = tmp_path / "heldout.jsonl"
    output = tmp_path / "result.json"
    _write_dataset(dataset)

    result = _evaluate(
        dataset,
        [0.91, 0.12, 0.73, 0.24],
        threshold=None,
        latency_sample_count=8,
        output=output,
    )

    assert result.rows == 4
    assert result.threshold == 0.5
    assert result.balanced_accuracy == 1.0
    assert result.true_positive_rate == 1.0
    assert result.true_negative_rate == 1.0
    assert result.latency_sample_count == 8
    assert result.latency_p50_ms == 2.0
    assert result.latency_p95_ms == 2.0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset"] == dataset.as_posix()
    assert payload["heldout_eval_dataset"] == dataset.as_posix()
    assert payload["heldout_eval_rows"] == 4
    assert payload["heldout_eval_balanced_accuracy"] == 1.0
    assert payload["heldout_eval_threshold"] == 0.5
    assert payload["latency_p95_ms"] == 2.0


def test_lite_scorer_v2_eval_selects_best_threshold(tmp_path: Path) -> None:
    """Automatic threshold selection should maximise balanced accuracy."""
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    result = _evaluate(
        dataset,
        [0.62, 0.58, 0.61, 0.2],
        threshold=None,
        latency_sample_count=4,
    )

    assert result.threshold == 0.595
    assert result.balanced_accuracy == 1.0


def test_lite_scorer_v2_eval_uses_fixed_threshold(tmp_path: Path) -> None:
    """A caller-supplied threshold should be used without optimisation."""
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    result = _evaluate(
        dataset,
        [0.49, 0.48, 0.95, 0.97],
        threshold=0.9,
        latency_sample_count=1,
    )

    assert result.threshold == 0.9
    assert result.balanced_accuracy == 0.5
    assert result.true_positive_rate == 0.5
    assert result.true_negative_rate == 0.5
    assert result.latency_p50_ms == 2.0
    assert result.latency_p95_ms == 2.0


def test_lite_scorer_v2_eval_rejects_invalid_threshold(tmp_path: Path) -> None:
    """Thresholds outside the probability range should fail."""
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
        _evaluate(dataset, [0.9, 0.1, 0.8, 0.2], threshold=1.1)


def test_lite_scorer_v2_eval_rejects_invalid_scores(tmp_path: Path) -> None:
    """Scorer outputs must stay in the probability range."""
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    with pytest.raises(ValueError, match="scorer returned score outside \\[0, 1\\]"):
        _evaluate(dataset, [1.2, 0.1, 0.8, 0.3])


def test_lite_scorer_v2_eval_rejects_invalid_latency_score(
    tmp_path: Path,
) -> None:
    """Latency samples should use the same score validation as evaluation."""
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    with pytest.raises(ValueError, match="scorer returned score outside \\[0, 1\\]"):
        _evaluate(
            dataset,
            [0.8, 0.2, 0.7, 0.3, 1.2],
            threshold=0.5,
            latency_sample_count=1,
        )


def test_lite_scorer_v2_eval_rejects_invalid_latency_inputs(
    tmp_path: Path,
) -> None:
    """Latency sampling should reject invalid sample counts and clocks."""
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    with pytest.raises(ValueError, match="latency_sample_count must be positive"):
        _evaluate(dataset, [0.8, 0.2, 0.7, 0.3], latency_sample_count=0)

    with pytest.raises(
        ValueError,
        match="latency clock produced non-positive duration",
    ):
        _evaluate(
            dataset,
            [0.8, 0.2, 0.7, 0.3],
            latency_sample_count=1,
            clock=_FrozenClock(),
        )


def test_lite_scorer_v2_eval_rejects_empty_percentile_values() -> None:
    """Percentile calculation requires at least one value."""
    with pytest.raises(ValueError, match="cannot calculate percentile"):
        evaluator._percentile([], 0.95)


def test_lite_scorer_v2_eval_main_prints_success_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The in-process CLI entry point should emit success JSON."""
    dataset = tmp_path / "heldout.jsonl"
    output = tmp_path / "result.json"
    _write_dataset(dataset)
    monkeypatch.setattr(
        evaluator,
        "_build_scorer",
        lambda _args: _StaticScorer([0.9, 0.1, 0.8, 0.2]),
    )

    exit_code = evaluator.main(
        [
            "--dataset",
            str(dataset),
            "--model-path",
            "unused-model",
            "--threshold",
            "0.5",
            "--latency-sample-count",
            "2",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload == {
        "balanced_accuracy": 1.0,
        "dataset": dataset.as_posix(),
        "latency_p50_ms": payload["latency_p50_ms"],
        "latency_p95_ms": payload["latency_p95_ms"],
        "latency_sample_count": 2,
        "rows": 4,
        "threshold": 0.5,
        "true_negative_rate": 1.0,
        "true_positive_rate": 1.0,
    }
    assert payload["latency_p50_ms"] > 0.0
    assert payload["latency_p95_ms"] >= payload["latency_p50_ms"]
    assert output.is_file()


def test_lite_scorer_v2_eval_builds_production_scorer() -> None:
    """The CLI scorer factory should construct the real backend lazily."""
    from director_ai.core.scoring.distilled_scorer import DistilledNLIBackend

    args = evaluator._build_parser().parse_args(
        [
            "--dataset",
            "heldout.jsonl",
            "--model-path",
            "local-model",
            "--device",
            "cpu",
            "--max-length",
            "128",
            "--no-onnx",
        ]
    )

    scorer = evaluator._build_scorer(args)

    assert isinstance(scorer, DistilledNLIBackend)
    assert scorer._model_path == "local-model"
    assert scorer._device == "cpu"
    assert scorer._max_length == 128
    assert scorer._use_onnx is False


def test_lite_scorer_v2_eval_main_reports_errors_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The CLI should keep machine-readable stdout empty on failures."""
    dataset = tmp_path / "heldout.jsonl"
    dataset.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        evaluator,
        "_build_scorer",
        lambda _args: _StaticScorer([0.9, 0.1]),
    )

    exit_code = evaluator.main(
        [
            "--dataset",
            str(dataset),
            "--model-path",
            "unused-model",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == f"{dataset}: dataset must contain at least one row\n"


def test_lite_scorer_v2_eval_main_reports_model_errors_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Model construction failures should be reported as CLI errors."""
    dataset = tmp_path / "heldout.jsonl"
    _write_dataset(dataset)

    def fail_build_scorer(_args: object) -> _StaticScorer:
        """Raise a model-loading failure for the CLI boundary."""
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(evaluator, "_build_scorer", fail_build_scorer)

    exit_code = evaluator.main(
        [
            "--dataset",
            str(dataset),
            "--model-path",
            "unused-model",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "model unavailable\n"
