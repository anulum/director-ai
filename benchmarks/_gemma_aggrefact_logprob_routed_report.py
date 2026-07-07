# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact routed logprob report helpers
"""Report-building helpers for routed Gemma AggreFact logprob evaluations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from _gemma_aggrefact_logprob_routed_core import (
    balanced_accuracy,
    per_dataset_sweep,
    per_family_sweep,
    sweep_threshold,
)
from _judge_common import DATASET_TO_FAMILY

logger = logging.getLogger(__name__)

METHOD_DESCRIPTION = "per-dataset prompt routing (summ/rag/claim) + logprob scores"


class RoutedLogprobMetric(TypedDict):
    """Per-group routed logprob threshold metric payload."""

    samples: int
    balanced_accuracy: float
    threshold: float


class RoutedLogprobReport(TypedDict):
    """JSON report schema for the routed logprob AggreFact evaluator."""

    schema_version: int
    model: str
    method: str
    samples: int
    global_balanced_accuracy_t05: float
    global_balanced_accuracy_optimal: float
    global_optimal_threshold: float
    per_dataset_avg_balanced_accuracy: float
    per_dataset: dict[str, RoutedLogprobMetric]
    per_family: dict[str, RoutedLogprobMetric]
    dataset_to_family: dict[str, str]
    invalid_scores: int
    total_time_seconds: float
    p50_latency_ms: float
    p99_latency_ms: float
    scores: list[float | None]
    labels: list[int]
    datasets_per_sample: list[str]
    families_per_sample: list[str]


def build_report(
    *,
    model_path: str,
    sample_count: int,
    scores: list[float | None],
    labels: list[int],
    datasets: list[str],
    families: list[str],
    latencies: list[float],
    total_time: float,
) -> RoutedLogprobReport:
    """Build the deterministic JSON payload for a routed logprob evaluation."""
    best_threshold, best_balanced_accuracy = sweep_threshold(scores, labels)
    per_dataset_raw, per_dataset_average = per_dataset_sweep(scores, labels, datasets)
    per_family_raw = per_family_sweep(scores, labels, families)
    return {
        "schema_version": 2,
        "model": model_path,
        "method": METHOD_DESCRIPTION,
        "samples": sample_count,
        "global_balanced_accuracy_t05": balanced_accuracy(scores, labels, 0.5),
        "global_balanced_accuracy_optimal": best_balanced_accuracy,
        "global_optimal_threshold": best_threshold,
        "per_dataset_avg_balanced_accuracy": per_dataset_average,
        "per_dataset": _typed_metrics(per_dataset_raw),
        "per_family": _typed_metrics(per_family_raw),
        "dataset_to_family": dict(DATASET_TO_FAMILY),
        "invalid_scores": sum(1 for score in scores if score is None),
        "total_time_seconds": total_time,
        "p50_latency_ms": _latency_percentile_ms(latencies, 0.50),
        "p99_latency_ms": _latency_percentile_ms(latencies, 0.99),
        "scores": scores,
        "labels": labels,
        "datasets_per_sample": datasets,
        "families_per_sample": families,
    }


def write_report(path: Path, report: RoutedLogprobReport) -> None:
    """Write a routed logprob evaluation report as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_summary(report: RoutedLogprobReport, output_path: Path) -> None:
    """Log a concise human-readable summary for operators."""
    sample_count = report["samples"]
    invalid_percent = (
        100 * report["invalid_scores"] / sample_count if sample_count else 0.0
    )
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
        invalid_percent,
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


def _typed_metrics(
    raw_metrics: dict[str, dict[str, float | int]],
) -> dict[str, RoutedLogprobMetric]:
    """Convert generic metric dictionaries into the public report schema."""
    metrics: dict[str, RoutedLogprobMetric] = {}
    for group_name, raw_metric in raw_metrics.items():
        metrics[group_name] = {
            "samples": int(raw_metric["samples"]),
            "balanced_accuracy": float(raw_metric["balanced_accuracy"]),
            "threshold": float(raw_metric["threshold"]),
        }
    return metrics


def _latency_percentile_ms(latencies: list[float], percentile: float) -> float:
    """Return a deterministic latency percentile in milliseconds."""
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    index = min(int(len(sorted_latencies) * percentile), len(sorted_latencies) - 1)
    return 1000 * sorted_latencies[index]
