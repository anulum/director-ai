# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — VitaminC Dev Benchmark (Held-Out Fact Verification)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate NLI model on VitaminC dev/test sets.

Training included VitaminC train split only. Dev and test are held-out.
VitaminC is Wikipedia-based contrastive fact verification: minimal edits
that flip claim truth values. Tests sensitivity to subtle factual
changes — directly relevant to hallucination detection on RAG output.

Usage::

    python -m benchmarks.vitaminc_eval 500
    python -m benchmarks.vitaminc_eval --model training/output/deberta-v3-base-hallucination
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

logger = logging.getLogger("DirectorAI.Benchmark.VitaminC")

# VitaminC uses: SUPPORTS=0, REFUTES=2, NOT ENOUGH INFO=1
# Map to standard NLI ordering
_LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 2, "NOT ENOUGH INFO": 1}


def _load_vitaminc_split(split: str) -> list[dict]:
    logger.info("Loading VitaminC %s ...", split)
    ds = load_dataset("tals/vitaminc", split=split)
    return list(ds)


def run_vitaminc_benchmark(
    split: str = "validation",
    max_samples: int | None = None,
    model_name: str | None = None,
) -> NLIMetrics:
    predictor = NLIPredictor(model_name=model_name)
    rows = _load_vitaminc_split(split)
    if max_samples:
        rows = rows[:max_samples]

    metrics = NLIMetrics()
    for row in rows:
        premise = row.get("evidence", "")
        hypothesis = row.get("claim", "")
        raw_label = row.get("label")

        if isinstance(raw_label, str):
            label = _LABEL_MAP.get(raw_label.upper())
        else:
            continue

        if label is None or not premise or not hypothesis:
            continue

        t0 = time.perf_counter()
        pred = predictor.predict(premise, hypothesis)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.y_true.append(label)
        metrics.y_pred.append(pred)

    return metrics


# ── Pytest ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_vitaminc_dev_sample():
    m = run_vitaminc_benchmark("validation", max_samples=200)
    print_nli_metrics(m, "VitaminC Dev (200 sample)")
    assert m.accuracy > 0.50


@pytest.mark.slow
def test_vitaminc_test_sample():
    m = run_vitaminc_benchmark("test", max_samples=200)
    print_nli_metrics(m, "VitaminC Test (200 sample)")
    assert m.accuracy > 0.50


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="VitaminC fact-verification benchmark")
    add_common_args(parser)
    parser.add_argument("--split", default="validation",
                        choices=["validation", "test"])
    args = parser.parse_args()

    m = run_vitaminc_benchmark(args.split, args.max_samples, args.model)
    print_nli_metrics(m, f"VitaminC {args.split}")

    save_results({"benchmark": f"VitaminC_{args.split}", **m.to_dict()},
                 "vitaminc_results.json")
