# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — MNLI Benchmark (Held-Out NLI Regression Test)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate NLI model on MNLI validation sets (matched + mismatched).

MNLI was NOT in our fine-tuning dataset. This is a pure held-out
regression test: fine-tuning for hallucination detection should not
degrade general-purpose NLI performance.

Published baselines (DeBERTa-v3-base-mnli-fever-anli):
  matched: ~90.5% acc, mismatched: ~90.1% acc

Usage::

    python -m benchmarks.mnli_eval 500
    python -m benchmarks.mnli_eval --model training/output/deberta-v3-base-hallucination
"""

from __future__ import annotations

import logging
import time

import pytest
from datasets import load_dataset

from benchmarks._common import (
    NLIMetrics,
    NLIPredictor,
    add_common_args,
    print_nli_metrics,
    save_results,
)

logger = logging.getLogger("DirectorAI.Benchmark.MNLI")

# MNLI uses string labels in HuggingFace datasets
_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load_mnli_split(split_name: str) -> list[dict]:
    """Load MNLI validation split from HuggingFace."""
    logger.info("Loading MNLI %s ...", split_name)
    ds = load_dataset("nyu-mll/multi_nli", split=split_name)
    return list(ds)


def run_mnli_benchmark(
    split: str = "validation_matched",
    max_samples: int | None = None,
    model_name: str | None = None,
) -> NLIMetrics:
    predictor = NLIPredictor(model_name=model_name)
    rows = _load_mnli_split(split)
    if max_samples:
        rows = rows[:max_samples]

    metrics = NLIMetrics()
    for row in rows:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        label = row.get("label")
        if label is None or label < 0 or not premise or not hypothesis:
            continue

        t0 = time.perf_counter()
        pred = predictor.predict(premise, hypothesis)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.y_true.append(int(label))
        metrics.y_pred.append(pred)

    return metrics


# ── Pytest ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_mnli_matched_sample():
    m = run_mnli_benchmark("validation_matched", max_samples=200)
    print_nli_metrics(m, "MNLI Matched (200 sample)")
    assert m.accuracy > 0.60, f"MNLI accuracy {m.accuracy:.1%} below 60% — possible regression"


@pytest.mark.slow
def test_mnli_mismatched_sample():
    m = run_mnli_benchmark("validation_mismatched", max_samples=200)
    print_nli_metrics(m, "MNLI Mismatched (200 sample)")
    assert m.accuracy > 0.60


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="MNLI NLI regression benchmark")
    add_common_args(parser)
    parser.add_argument("--split", choices=["matched", "mismatched", "both"],
                        default="both")
    args = parser.parse_args()

    results = {"benchmark": "MNLI"}
    splits = (["validation_matched", "validation_mismatched"]
              if args.split == "both"
              else [f"validation_{args.split}"])

    for split in splits:
        m = run_mnli_benchmark(split, max_samples=args.max_samples, model_name=args.model)
        label = split.replace("validation_", "")
        print_nli_metrics(m, f"MNLI {label}")
        results[label] = m.to_dict()

    save_results(results, "mnli_results.json")
