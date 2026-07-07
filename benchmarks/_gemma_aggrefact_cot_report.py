# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact CoT report helpers
"""Report-building helpers for Gemma AggreFact CoT evaluations."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from _gemma_aggrefact_cot_core import compute_ba
from _gemma_aggrefact_eval_schema import DatasetMetric

logger = logging.getLogger(__name__)

PROMPT_STYLE = "chain-of-thought (1-2 sentences + ANSWER)"


class CoTReport(TypedDict):
    """JSON report schema for the CoT AggreFact evaluator."""

    model: str
    prompt_style: str
    samples: int
    global_balanced_accuracy: float
    per_dataset: dict[str, DatasetMetric]
    unknown_predictions: int
    total_time_seconds: float
    p50_latency_ms: float
    p99_latency_ms: float
    sample_responses: list[str]
    predictions: list[int]
    labels: list[int]
    datasets_per_sample: list[str]


def build_report(
    *,
    model_path: str,
    sample_count: int,
    preds: list[int],
    labels: list[int],
    datasets_per_sample: list[str],
    latencies: list[float],
    raw_responses: list[str],
    unknown: int,
    total_time: float,
) -> CoTReport:
    """Build the deterministic JSON payload for a CoT evaluation run."""
    sorted_latencies = sorted(latencies)
    return {
        "model": model_path,
        "prompt_style": PROMPT_STYLE,
        "samples": sample_count,
        "global_balanced_accuracy": compute_ba(preds, labels),
        "per_dataset": _per_dataset_metrics(preds, labels, datasets_per_sample),
        "unknown_predictions": unknown,
        "total_time_seconds": total_time,
        "p50_latency_ms": 1000 * sorted_latencies[len(sorted_latencies) // 2],
        "p99_latency_ms": 1000
        * sorted_latencies[
            min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)
        ],
        "sample_responses": raw_responses[:20],
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_per_sample,
    }


def write_report(path: Path, report: CoTReport) -> None:
    """Write a CoT evaluation report as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_summary(report: CoTReport, output_path: Path) -> None:
    """Log a concise human-readable summary for operators."""
    logger.info("=" * 60)
    logger.info("Global BA: %.4f", report["global_balanced_accuracy"])
    logger.info(
        "Unknown:   %d (%.1f%%)",
        report["unknown_predictions"],
        100 * report["unknown_predictions"] / report["samples"],
    )
    logger.info(
        "Time:      %.1fmin (%.0fms/sample)",
        report["total_time_seconds"] / 60,
        1000 * report["total_time_seconds"] / report["samples"],
    )
    logger.info("=" * 60)
    for dataset_name, metrics in sorted(report["per_dataset"].items()):
        logger.info(
            "  %-20s %5d  %.4f",
            dataset_name,
            metrics["samples"],
            metrics["balanced_accuracy"],
        )
    logger.info("Saved: %s", output_path)


def _per_dataset_metrics(
    preds: Sequence[int],
    labels: Sequence[int],
    datasets: Sequence[str],
) -> dict[str, DatasetMetric]:
    """Compute balanced accuracy for every represented AggreFact subset."""
    grouped: dict[str, tuple[list[int], list[int]]] = {}
    for prediction, label, dataset_name in zip(preds, labels, datasets, strict=True):
        dataset_preds, dataset_labels = grouped.setdefault(dataset_name, ([], []))
        dataset_preds.append(prediction)
        dataset_labels.append(label)
    return {
        dataset_name: {
            "samples": len(dataset_labels),
            "balanced_accuracy": compute_ba(dataset_preds, dataset_labels),
        }
        for dataset_name, (dataset_preds, dataset_labels) in grouped.items()
    }
