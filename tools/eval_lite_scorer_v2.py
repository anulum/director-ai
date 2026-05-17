#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 held-out evaluator

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class SupportsScore(Protocol):
    def score(self, premise: str, hypothesis: str) -> float: ...


@dataclass(frozen=True)
class EvalRow:
    premise: str
    hypothesis: str
    label: bool


@dataclass(frozen=True)
class LiteScorerV2EvalResult:
    dataset: str
    rows: int
    threshold: float
    balanced_accuracy: float
    true_positive_rate: float
    true_negative_rate: float
    latency_sample_count: int
    latency_p50_ms: float
    latency_p95_ms: float


def _row_error(path: Path, line_number: int, message: str) -> str:
    return f"{path}:{line_number}: {message}"


def _parse_label(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"supported", "entailment", "true", "1"}:
            return True
        if normalised in {"unsupported", "contradiction", "false", "0"}:
            return False
    return None


def load_lite_scorer_v2_dataset(path: Path) -> tuple[list[EvalRow], list[str]]:
    if not path.exists():
        return [], [f"{path}: dataset does not exist"]
    rows: list[EvalRow] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            errors.append(_row_error(path, line_number, "blank lines are not allowed"))
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(_row_error(path, line_number, f"invalid JSON: {exc}"))
            continue
        if not isinstance(raw, dict):
            errors.append(_row_error(path, line_number, "row must be a JSON object"))
            continue
        premise = raw.get("premise")
        hypothesis = raw.get("hypothesis")
        label = _parse_label(raw.get("label"))
        if not isinstance(premise, str) or not premise.strip():
            errors.append(_row_error(path, line_number, "premise must be a non-empty string"))
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            errors.append(_row_error(path, line_number, "hypothesis must be a non-empty string"))
        if label is None:
            errors.append(_row_error(path, line_number, "label must be boolean or supported/unsupported"))
        if (
            isinstance(premise, str)
            and premise.strip()
            and isinstance(hypothesis, str)
            and hypothesis.strip()
            and label is not None
        ):
            rows.append(EvalRow(premise=premise, hypothesis=hypothesis, label=label))
    if not errors and rows:
        labels = {row.label for row in rows}
        if labels != {False, True}:
            errors.append(f"{path}: dataset must contain supported and unsupported rows")
    if not errors and not rows:
        errors.append(f"{path}: dataset must contain at least one row")
    return rows, errors


def _balanced_accuracy(labels: list[bool], scores: list[float], threshold: float) -> tuple[float, float, float]:
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    true_positive_rate = sum(scores[index] >= threshold for index in positives) / len(positives)
    true_negative_rate = sum(scores[index] < threshold for index in negatives) / len(negatives)
    return (true_positive_rate + true_negative_rate) / 2.0, true_positive_rate, true_negative_rate


def _candidate_thresholds(scores: list[float]) -> list[float]:
    values = sorted(set(scores))
    candidates = {0.5}
    candidates.update(values)
    for left, right in zip(values, values[1:], strict=False):
        candidates.add((left + right) / 2.0)
    return sorted(candidates)


def _select_threshold(labels: list[bool], scores: list[float]) -> tuple[float, float, float, float]:
    ranked: list[tuple[float, float, float, float, float]] = []
    for threshold in _candidate_thresholds(scores):
        balanced_accuracy, true_positive_rate, true_negative_rate = _balanced_accuracy(
            labels,
            scores,
            threshold,
        )
        ranked.append(
            (
                balanced_accuracy,
                -abs(threshold - 0.5),
                threshold,
                true_positive_rate,
                true_negative_rate,
            )
        )
    best = max(ranked)
    return best[2], best[0], best[3], best[4]


def _score_rows(rows: list[EvalRow], scorer: SupportsScore) -> list[float]:
    scores: list[float] = []
    for row in rows:
        score = float(scorer.score(row.premise, row.hypothesis))
        if score < 0.0 or score > 1.0:
            raise ValueError("scorer returned score outside [0, 1]")
        scores.append(score)
    return scores


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate percentile for empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _measure_latency(
    rows: list[EvalRow],
    scorer: SupportsScore,
    sample_count: int,
    clock: Callable[[], float],
) -> tuple[float, float]:
    if sample_count < 1:
        raise ValueError("latency_sample_count must be positive")
    timings_ms: list[float] = []
    for index in range(sample_count):
        row = rows[index % len(rows)]
        start = clock()
        score = float(scorer.score(row.premise, row.hypothesis))
        end = clock()
        if score < 0.0 or score > 1.0:
            raise ValueError("scorer returned score outside [0, 1]")
        elapsed_ms = (end - start) * 1000.0
        if elapsed_ms <= 0.0:
            raise ValueError("latency clock produced non-positive duration")
        timings_ms.append(round(elapsed_ms, 6))
    return round(statistics.median(timings_ms), 6), round(_percentile(timings_ms, 0.95), 6)


def _write_result(path: Path, result: LiteScorerV2EvalResult) -> None:
    payload = asdict(result)
    payload["heldout_eval_dataset"] = result.dataset
    payload["heldout_eval_rows"] = result.rows
    payload["heldout_eval_balanced_accuracy"] = result.balanced_accuracy
    payload["heldout_eval_threshold"] = result.threshold
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_lite_scorer_v2(
    *,
    dataset: Path,
    scorer: SupportsScore,
    threshold: float | None,
    latency_sample_count: int,
    clock: Callable[[], float] = time.perf_counter,
    output: Path | None = None,
) -> LiteScorerV2EvalResult:
    rows, errors = load_lite_scorer_v2_dataset(dataset)
    if errors:
        raise ValueError("; ".join(errors))
    labels = [row.label for row in rows]
    scores = _score_rows(rows, scorer)

    if threshold is None:
        selected_threshold, balanced_accuracy, true_positive_rate, true_negative_rate = _select_threshold(
            labels,
            scores,
        )
    else:
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in [0, 1]")
        selected_threshold = threshold
        balanced_accuracy, true_positive_rate, true_negative_rate = _balanced_accuracy(
            labels,
            scores,
            selected_threshold,
        )

    latency_p50_ms, latency_p95_ms = _measure_latency(
        rows,
        scorer,
        latency_sample_count,
        clock,
    )
    result = LiteScorerV2EvalResult(
        dataset=dataset.as_posix(),
        rows=len(rows),
        threshold=round(selected_threshold, 6),
        balanced_accuracy=round(balanced_accuracy, 6),
        true_positive_rate=round(true_positive_rate, 6),
        true_negative_rate=round(true_negative_rate, 6),
        latency_sample_count=latency_sample_count,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
    )
    if output is not None:
        _write_result(output, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", default=256, type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--latency-sample-count", default=100, type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-onnx", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from director_ai.core.scoring.distilled_scorer import DistilledNLIBackend

    scorer = DistilledNLIBackend(
        model_path=args.model_path,
        use_onnx=not args.no_onnx,
        device=args.device,
        max_length=args.max_length,
    )
    try:
        result = evaluate_lite_scorer_v2(
            dataset=args.dataset,
            scorer=scorer,
            threshold=args.threshold,
            latency_sample_count=args.latency_sample_count,
            output=args.output,
        )
    except ValueError as exc:
        print(str(exc))
        return 1
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
