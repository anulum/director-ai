# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact HiSS report helpers
"""Report-building helpers for Gemma AggreFact HiSS evaluations."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from _gemma_aggrefact_eval_schema import DatasetMetric
from _gemma_aggrefact_hiss_core import HiSSSampleTrace
from _judge_common import compute_balanced_accuracy

logger = logging.getLogger(__name__)

HISS_METHOD = "HiSS (decompose + per-subclaim verify)"


class HiSSReport(TypedDict):
    """JSON report schema for the HiSS AggreFact evaluator."""

    model: str
    method: str
    samples: int
    global_balanced_accuracy: float
    per_dataset: dict[str, DatasetMetric]
    unknown_predictions: int
    mean_subclaims_per_sample: float
    total_time_seconds: float
    p50_latency_ms: float
    p99_latency_ms: float
    predictions: list[int]
    labels: list[int]
    datasets_per_sample: list[str]
    subclaim_counts: list[int]
    first_10_samples: list[HiSSSampleTrace]


def build_report(
    *,
    model_path: str,
    sample_count: int,
    preds: list[int],
    labels: list[int],
    datasets_per_sample: list[str],
    subclaim_counts: list[int],
    latencies: list[float],
    traces: list[HiSSSampleTrace],
    unknown: int,
    total_time: float,
) -> HiSSReport:
    """Build the deterministic JSON payload for a HiSS evaluation run."""
    if sample_count <= 0 or not subclaim_counts or not latencies:
        raise ValueError("HiSS report requires at least one sample")

    sorted_latencies = sorted(latencies)
    return {
        "model": model_path,
        "method": HISS_METHOD,
        "samples": sample_count,
        "global_balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "per_dataset": _per_dataset_metrics(preds, labels, datasets_per_sample),
        "unknown_predictions": unknown,
        "mean_subclaims_per_sample": sum(subclaim_counts) / len(subclaim_counts),
        "total_time_seconds": total_time,
        "p50_latency_ms": 1000 * sorted_latencies[len(sorted_latencies) // 2],
        "p99_latency_ms": 1000
        * sorted_latencies[
            min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)
        ],
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_per_sample,
        "subclaim_counts": subclaim_counts,
        "first_10_samples": traces,
    }


def write_report(path: Path, report: HiSSReport) -> None:
    """Write a HiSS evaluation report as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_summary(report: HiSSReport, output_path: Path) -> None:
    """Log a concise human-readable summary for operators."""
    logger.info("=" * 60)
    logger.info("Global BA:           %.4f", report["global_balanced_accuracy"])
    logger.info("Mean sub-claims:     %.2f", report["mean_subclaims_per_sample"])
    logger.info(
        "Unknown:             %d (%.1f%%)",
        report["unknown_predictions"],
        100 * report["unknown_predictions"] / report["samples"],
    )
    logger.info("Time:                %.1fmin", report["total_time_seconds"] / 60)
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
            "balanced_accuracy": compute_balanced_accuracy(
                dataset_preds,
                dataset_labels,
            ),
        }
        for dataset_name, (dataset_preds, dataset_labels) in grouped.items()
    }
