# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact logprob report helpers
"""Report-building helpers for Gemma AggreFact logprob evaluations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from _gemma_aggrefact_logprob_core import (
    compute_balanced_accuracy,
    per_dataset_sweep,
    sweep_threshold,
)

logger = logging.getLogger(__name__)


class LogprobDatasetMetric(TypedDict):
    """Per-dataset logprob threshold metric payload."""

    samples: int
    balanced_accuracy: float
    threshold: float


class LogprobReport(TypedDict):
    """JSON report schema for the logprob AggreFact evaluator."""

    model: str
    samples: int
    global_balanced_accuracy_t05: float
    global_balanced_accuracy_optimal: float
    global_optimal_threshold: float
    per_dataset_avg_balanced_accuracy: float
    per_dataset: dict[str, LogprobDatasetMetric]
    invalid_scores: int
    total_time_seconds: float
    p50_latency_ms: float
    p99_latency_ms: float
    scores: list[float | None]
    labels: list[int]
    datasets: list[str]


def build_report(
    *,
    model_path: str,
    sample_count: int,
    scores: list[float | None],
    labels: list[int],
    datasets: list[str],
    latencies: list[float],
    total_time: float,
) -> LogprobReport:
    """Build the deterministic JSON payload for a logprob evaluation run."""
    sorted_latencies = sorted(latencies)
    best_threshold, best_balanced_accuracy = sweep_threshold(scores, labels)
    per_dataset_raw, per_dataset_average = per_dataset_sweep(scores, labels, datasets)
    per_dataset: dict[str, LogprobDatasetMetric] = {}
    for dataset_name, metric in per_dataset_raw.items():
        per_dataset[dataset_name] = {
            "samples": int(metric["samples"]),
            "balanced_accuracy": float(metric["balanced_accuracy"]),
            "threshold": float(metric["threshold"]),
        }
    return {
        "model": model_path,
        "samples": sample_count,
        "global_balanced_accuracy_t05": compute_balanced_accuracy(scores, labels, 0.5),
        "global_balanced_accuracy_optimal": best_balanced_accuracy,
        "global_optimal_threshold": best_threshold,
        "per_dataset_avg_balanced_accuracy": per_dataset_average,
        "per_dataset": per_dataset,
        "invalid_scores": sum(1 for score in scores if score is None),
        "total_time_seconds": total_time,
        "p50_latency_ms": 1000 * sorted_latencies[len(sorted_latencies) // 2],
        "p99_latency_ms": 1000
        * sorted_latencies[
            min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)
        ],
        "scores": scores,
        "labels": labels,
        "datasets": datasets,
    }


def write_report(path: Path, report: LogprobReport) -> None:
    """Write a logprob evaluation report as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_summary(report: LogprobReport, output_path: Path) -> None:
    """Log a concise human-readable summary for operators."""
    logger.info("=" * 60)
    logger.info("Global BA @ t=0.5:    %.4f", report["global_balanced_accuracy_t05"])
    logger.info(
        "Global BA optimal:    %.4f (t=%.2f)",
        report["global_balanced_accuracy_optimal"],
        report["global_optimal_threshold"],
    )
    logger.info(
        "Per-dataset average:  %.4f",
        report["per_dataset_avg_balanced_accuracy"],
    )
    logger.info(
        "Invalid: %d (%.1f%%)",
        report["invalid_scores"],
        100 * report["invalid_scores"] / report["samples"],
    )
    logger.info("Time: %.1fmin", report["total_time_seconds"] / 60)
    logger.info("=" * 60)
    for dataset_name, metrics in sorted(report["per_dataset"].items()):
        logger.info(
            "  %-20s %5d  BA=%.4f  t=%.2f",
            dataset_name,
            metrics["samples"],
            metrics["balanced_accuracy"],
            metrics["threshold"],
        )
    logger.info("Saved: %s", output_path)
