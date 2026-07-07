# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact self-consistency report helpers
"""Report-building helpers for Gemma AggreFact self-consistency evaluations."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict

from _gemma_aggrefact_eval_schema import DatasetMetric
from _judge_common import DATASET_TO_FAMILY, compute_balanced_accuracy

logger = logging.getLogger(__name__)


class SelfConsistencyReport(TypedDict):
    """JSON report schema for the self-consistency AggreFact evaluator."""

    schema_version: int
    model: str
    method: str
    samples: int
    k: int
    temperature: float
    top_p: float
    global_balanced_accuracy: float
    per_dataset: dict[str, DatasetMetric]
    per_family: dict[str, DatasetMetric]
    dataset_to_family: dict[str, str]
    unknown_predictions: int
    total_time_seconds: float
    p50_latency_ms: float
    p99_latency_ms: float
    predictions: list[int]
    support_fractions: list[float | None]
    labels: list[int]
    datasets_per_sample: list[str]
    families_per_sample: list[str]


def build_report(
    *,
    model_path: str,
    sample_count: int,
    k: int,
    temperature: float,
    top_p: float,
    preds: list[int],
    support_fractions: list[float | None],
    labels: list[int],
    datasets_per_sample: list[str],
    families_per_sample: list[str],
    latencies: list[float],
    unknown: int,
    total_time: float,
) -> SelfConsistencyReport:
    """Build the deterministic JSON payload for a self-consistency run."""
    if sample_count <= 0 or not latencies:
        raise ValueError("self-consistency report requires at least one sample")

    return {
        "schema_version": 2,
        "model": model_path,
        "method": f"self-consistency K={k} T={temperature} on routed prompts",
        "samples": sample_count,
        "k": k,
        "temperature": temperature,
        "top_p": top_p,
        "global_balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "per_dataset": _group_metrics(preds, labels, datasets_per_sample),
        "per_family": _group_metrics(preds, labels, families_per_sample),
        "dataset_to_family": dict(DATASET_TO_FAMILY),
        "unknown_predictions": unknown,
        "total_time_seconds": total_time,
        "p50_latency_ms": _latency_percentile_ms(latencies, 0.50),
        "p99_latency_ms": _latency_percentile_ms(latencies, 0.99),
        "predictions": preds,
        "support_fractions": support_fractions,
        "labels": labels,
        "datasets_per_sample": datasets_per_sample,
        "families_per_sample": families_per_sample,
    }


def write_report(path: Path, report: SelfConsistencyReport) -> None:
    """Write a self-consistency evaluation report as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_summary(report: SelfConsistencyReport, output_path: Path) -> None:
    """Log a concise human-readable summary for operators."""
    logger.info("=" * 60)
    logger.info("K=%d T=%.2f", report["k"], report["temperature"])
    logger.info("Global BA:    %.4f", report["global_balanced_accuracy"])
    logger.info(
        "Unknown:      %d (%.1f%%)",
        report["unknown_predictions"],
        100 * report["unknown_predictions"] / report["samples"],
    )
    logger.info("Time:         %.1fmin", report["total_time_seconds"] / 60)
    logger.info("=" * 60)
    for family, metrics in sorted(report["per_family"].items()):
        logger.info(
            "  %-8s %5d  %.4f",
            family,
            metrics["samples"],
            metrics["balanced_accuracy"],
        )
    for dataset_name, metrics in sorted(report["per_dataset"].items()):
        logger.info(
            "  %-20s %5d  %.4f",
            dataset_name,
            metrics["samples"],
            metrics["balanced_accuracy"],
        )
    logger.info("Saved: %s", output_path)


def _group_metrics(
    preds: Sequence[int],
    labels: Sequence[int],
    groups: Sequence[str],
) -> dict[str, DatasetMetric]:
    """Compute balanced accuracy for every represented group."""
    grouped: dict[str, tuple[list[int], list[int]]] = {}
    for prediction, label, group_name in zip(preds, labels, groups, strict=True):
        group_preds, group_labels = grouped.setdefault(group_name, ([], []))
        group_preds.append(prediction)
        group_labels.append(label)
    return {
        group_name: {
            "samples": len(group_labels),
            "balanced_accuracy": compute_balanced_accuracy(group_preds, group_labels),
        }
        for group_name, (group_preds, group_labels) in grouped.items()
    }


def _latency_percentile_ms(latencies: list[float], percentile: float) -> float:
    """Return a deterministic latency percentile in milliseconds."""
    sorted_latencies = sorted(latencies)
    index = min(int(len(sorted_latencies) * percentile), len(sorted_latencies) - 1)
    return 1000 * sorted_latencies[index]
