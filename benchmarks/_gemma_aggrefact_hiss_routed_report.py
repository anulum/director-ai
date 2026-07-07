# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI - Gemma AggreFact routed HiSS report helpers
"""Report-building helpers for routed Gemma AggreFact HiSS evaluations."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict, cast

from _gemma_aggrefact_eval_schema import DatasetMetric
from _judge_common import (
    DATASET_TO_FAMILY,
    aggregate_per_dataset,
    aggregate_per_family,
    compute_balanced_accuracy,
)

logger = logging.getLogger(__name__)


class HiSSRoutedReport(TypedDict):
    """JSON report schema for the routed HiSS AggreFact evaluator."""

    schema_version: int
    model: str
    method: str
    samples: int
    min_decompose_words: int
    support_frac: float
    max_subclaims: int
    skipped_decompose: int
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
    subclaim_counts: list[int]
    decomposed_flags: list[bool]
    labels: list[int]
    datasets_per_sample: list[str]
    families_per_sample: list[str]


def build_report(
    *,
    model_path: str,
    sample_count: int,
    min_decompose_words: int,
    support_frac: float,
    max_subclaims: int,
    skipped_decompose: int,
    preds: list[int],
    support_fractions: list[float | None],
    labels: list[int],
    datasets_per_sample: list[str],
    families_per_sample: list[str],
    subclaim_counts: list[int],
    decomposed_flags: list[bool],
    latencies: list[float],
    unknown_predictions: int,
    total_time: float,
) -> HiSSRoutedReport:
    """Build the deterministic JSON payload for a routed HiSS evaluation."""
    if sample_count <= 0 or not latencies:
        raise ValueError("routed HiSS report requires at least one sample")

    sorted_latencies = sorted(latencies)
    per_dataset = cast(
        dict[str, DatasetMetric],
        aggregate_per_dataset(preds, labels, datasets_per_sample),
    )
    per_family = cast(
        dict[str, DatasetMetric],
        aggregate_per_family(preds, labels, families_per_sample),
    )
    return {
        "schema_version": 2,
        "model": model_path,
        "method": (
            "HiSS routed: decompose then per-family verify, "
            f"min_words={min_decompose_words} "
            f"support_frac={support_frac} "
            f"max_sub={max_subclaims}"
        ),
        "samples": sample_count,
        "min_decompose_words": min_decompose_words,
        "support_frac": support_frac,
        "max_subclaims": max_subclaims,
        "skipped_decompose": skipped_decompose,
        "global_balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "per_dataset": per_dataset,
        "per_family": per_family,
        "dataset_to_family": DATASET_TO_FAMILY,
        "unknown_predictions": unknown_predictions,
        "total_time_seconds": total_time,
        "p50_latency_ms": 1000 * sorted_latencies[len(sorted_latencies) // 2],
        "p99_latency_ms": 1000
        * sorted_latencies[
            min(int(len(sorted_latencies) * 0.99), len(sorted_latencies) - 1)
        ],
        "predictions": preds,
        "support_fractions": support_fractions,
        "subclaim_counts": subclaim_counts,
        "decomposed_flags": decomposed_flags,
        "labels": labels,
        "datasets_per_sample": datasets_per_sample,
        "families_per_sample": families_per_sample,
    }


def write_report(path: Path, report: HiSSRoutedReport) -> None:
    """Write a routed HiSS evaluation report as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_summary(report: HiSSRoutedReport, output_path: Path) -> None:
    """Log a concise routed HiSS summary for operators."""
    sample_count = report["samples"]
    skipped_decompose = report["skipped_decompose"]
    decomposed = sample_count - skipped_decompose

    logger.info("=" * 60)
    logger.info("Global BA:    %.4f", report["global_balanced_accuracy"])
    logger.info(
        "Decomposed:   %d (%.1f%%)", decomposed, 100 * decomposed / sample_count
    )
    logger.info(
        "Skipped:      %d (%.1f%%, used K=1 routed)",
        skipped_decompose,
        100 * skipped_decompose / sample_count,
    )
    logger.info(
        "Unknown:      %d (%.1f%%)",
        report["unknown_predictions"],
        100 * report["unknown_predictions"] / sample_count,
    )
    logger.info("Time:         %.1fmin", report["total_time_seconds"] / 60)
    logger.info("=" * 60)
    for family, metric in sorted(report["per_family"].items()):
        logger.info(
            "  %-8s %5d  %.4f",
            family,
            metric["samples"],
            metric["balanced_accuracy"],
        )
    for dataset_name, metric in sorted(report["per_dataset"].items()):
        logger.info(
            "  %-20s %5d  %.4f",
            dataset_name,
            metric["samples"],
            metric["balanced_accuracy"],
        )
    logger.info("Saved: %s", output_path)
