# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Gemma AggreFact routed evaluator core
"""Core evaluation utilities for the routed Gemma AggreFact benchmark."""

from __future__ import annotations

import importlib
import json
import logging
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from _gemma_aggrefact_eval_schema import (
    AggreFactDataset,
    DatasetLoader,
    DatasetMetric,
    LlamaFactory,
    LlamaModel,
)
from _judge_common import (
    DATASET_TO_FAMILY,
    PROMPTS,
    aggregate_per_dataset,
    aggregate_per_family,
    compute_balanced_accuracy,
    parse_response,
)

logger = logging.getLogger(__name__)


class RoutedReport(TypedDict):
    """Gemma AggreFact routed evaluator report schema."""

    model: str
    method: str
    samples: int
    global_balanced_accuracy: float
    per_dataset: dict[str, DatasetMetric]
    per_family: dict[str, DatasetMetric]
    dataset_to_family: dict[str, str]
    unknown_predictions: int
    total_time_seconds: float
    p50_latency_ms: float
    p99_latency_ms: float
    predictions: list[int]
    labels: list[int]
    datasets_per_sample: list[str]
    families_per_sample: list[str]


def load_aggrefact(max_samples: int | None) -> AggreFactDataset:
    """Load and optionally truncate the AggreFact test split."""
    datasets_module = importlib.import_module("datasets")
    load_dataset = cast(DatasetLoader, vars(datasets_module)["load_dataset"])
    dataset = load_dataset("lytang/LLM-AggreFact", split="test")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    if len(dataset) == 0:
        raise ValueError("dataset is empty")
    return dataset


def build_llama(model_path: str, *, n_ctx: int, n_threads: int) -> LlamaModel:
    """Construct the llama-cpp backend through its public factory."""
    llama_module = importlib.import_module("llama_cpp")
    llama_factory = cast(LlamaFactory, vars(llama_module)["Llama"])
    return llama_factory(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=512,
        verbose=False,
        logits_all=False,
    )


def _row_text(row: Mapping[str, object], key: str) -> str:
    """Return a required row field as text."""
    try:
        value = row[key]
    except KeyError as exc:
        raise ValueError(f"AggreFact row missing {key!r}") from exc
    return str(value)


def _row_label(row: Mapping[str, object]) -> int:
    """Return a required binary label from an AggreFact row."""
    try:
        value = row["label"]
    except KeyError as exc:
        raise ValueError("AggreFact row missing 'label'") from exc
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError("AggreFact row label must be an integer")


def _chat_content(response: Mapping[str, object]) -> str:
    """Extract the first chat-completion message content."""
    choices = response.get("choices")
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return ""
    if not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return ""
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return ""
    return str(message.get("content", ""))


def evaluate_dataset(
    dataset: AggreFactDataset,
    llm: LlamaModel,
    *,
    log_every: int,
) -> tuple[list[int], list[int], list[str], list[str], list[float], int, float]:
    """Evaluate routed prompts across an AggreFact dataset."""
    preds: list[int] = []
    labels: list[int] = []
    datasets_list: list[str] = []
    families: list[str] = []
    latencies: list[float] = []
    unknown = 0
    total_samples = len(dataset)
    t_start = time.time()

    for i, sample in enumerate(dataset):
        premise = _row_text(sample, "doc")
        hypothesis = _row_text(sample, "claim")
        label = _row_label(sample)
        dataset_name = _row_text(sample, "dataset")
        family = DATASET_TO_FAMILY.get(dataset_name, "claim")
        prompt = PROMPTS[family].format(premise=premise, hypothesis=hypothesis)

        t0 = time.time()
        try:
            out = llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8,
                temperature=0.0,
            )
            text = _chat_content(out)
        except Exception as exc:
            logger.warning("Sample %d failed: %s", i, exc)
            text = "ERROR"
        latencies.append(time.time() - t0)

        pred = parse_response(text)
        if pred < 0:
            unknown += 1
        preds.append(pred)
        labels.append(label)
        datasets_list.append(dataset_name)
        families.append(family)

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t_start
            ba = compute_balanced_accuracy(preds, labels)
            eta = (total_samples - i - 1) * elapsed / (i + 1) / 60
            logger.info(
                "[%d/%d] BA=%.4f unk=%d %.0fms/sample ETA=%.1fmin",
                i + 1,
                total_samples,
                ba,
                unknown,
                1000 * elapsed / (i + 1),
                eta,
            )

    return preds, labels, datasets_list, families, latencies, unknown, t_start


def family_distribution(dataset: AggreFactDataset) -> dict[str, int]:
    """Return routed task-family counts for operator logging."""
    family_counts: defaultdict[str, int] = defaultdict(int)
    for sample in dataset:
        dataset_name = _row_text(sample, "dataset")
        family_counts[DATASET_TO_FAMILY.get(dataset_name, "claim")] += 1
    return dict(family_counts)


def build_report(
    *,
    model_path: str,
    sample_count: int,
    preds: list[int],
    labels: list[int],
    datasets_list: list[str],
    families: list[str],
    latencies: list[float],
    unknown: int,
    total: float,
) -> RoutedReport:
    """Build the JSON report payload for routed AggreFact evaluation."""
    per_ds_metrics = cast(
        dict[str, DatasetMetric],
        aggregate_per_dataset(preds, labels, datasets_list),
    )
    per_family_metrics = cast(
        dict[str, DatasetMetric],
        aggregate_per_family(preds, labels, families),
    )
    return {
        "model": model_path,
        "method": "per-dataset prompt routing (summ/rag/claim families)",
        "samples": sample_count,
        "global_balanced_accuracy": compute_balanced_accuracy(preds, labels),
        "per_dataset": per_ds_metrics,
        "per_family": per_family_metrics,
        "dataset_to_family": DATASET_TO_FAMILY,
        "unknown_predictions": unknown,
        "total_time_seconds": total,
        "p50_latency_ms": 1000 * sorted(latencies)[len(latencies) // 2]
        if latencies
        else 0,
        "p99_latency_ms": (
            1000 * sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
        "predictions": preds,
        "labels": labels,
        "datasets_per_sample": datasets_list,
        "families_per_sample": families,
    }


def write_report(output_path: Path, report: RoutedReport) -> None:
    """Write a routed evaluation report as deterministic UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def log_summary(
    *,
    report: RoutedReport,
    unknown: int,
    total: float,
    output_path: Path,
) -> None:
    """Log the routed evaluation summary for operators."""
    sample_count = report["samples"]

    logger.info("=" * 60)
    logger.info("Global BA:    %.4f", report["global_balanced_accuracy"])
    logger.info("Unknown:      %d (%.1f%%)", unknown, 100 * unknown / sample_count)
    logger.info("Time:         %.1fmin", total / 60)
    logger.info("=" * 60)
    logger.info("Per-family:")
    for fam, m in sorted(report["per_family"].items()):
        logger.info("  %-8s %5d  %.4f", fam, m["samples"], m["balanced_accuracy"])
    logger.info("Per-dataset:")
    for n, m in sorted(report["per_dataset"].items()):
        logger.info("  %-20s %5d  %.4f", n, m["samples"], m["balanced_accuracy"])
    logger.info("Saved: %s", output_path)
