# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — ANLI Benchmark (Adversarial NLI, Held-Out Rounds)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate NLI model on ANLI Round 1, 2, and 3 test sets.

Training data included ANLI R3 train only. R1 and R2 are fully held-out.
R3 test is held-out from R3 train. This tests adversarial robustness:
ANLI examples were constructed to fool NLI models.

Published baselines (DeBERTa-v3-base-mnli-fever-anli):
  R1 test: ~73%, R2 test: ~58%, R3 test: ~52%

Usage::

    python -m benchmarks.anli_eval 500
    python -m benchmarks.anli_eval --model training/output/deberta-v3-base-hallucination
    python -m benchmarks.anli_eval --rounds r1 r2
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

logger = logging.getLogger("DirectorAI.Benchmark.ANLI")


def _load_anli_split(round_name: str, split: str) -> list[dict]:
    """Load an ANLI split. round_name: 'r1','r2','r3'. split: 'dev','test'."""
    hf_split = f"{split}_{round_name}"
    logger.info("Loading ANLI %s/%s ...", round_name, split)
    ds = load_dataset("anli", split=hf_split)
    return list(ds)


def run_anli_benchmark(
    round_name: str = "r1",
    split: str = "test",
    max_samples: int | None = None,
    model_name: str | None = None,
) -> NLIMetrics:
    predictor = NLIPredictor(model_name=model_name)
    rows = _load_anli_split(round_name, split)
    if max_samples:
        rows = rows[:max_samples]

    metrics = NLIMetrics()
    for row in rows:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        label = row.get("label")
        if label is None or not premise or not hypothesis:
            continue

        t0 = time.perf_counter()
        pred = predictor.predict(premise, hypothesis)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.y_true.append(int(label))
        metrics.y_pred.append(pred)

    return metrics


# ── Pytest ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_anli_r1_sample():
    m = run_anli_benchmark("r1", "test", max_samples=200)
    print_nli_metrics(m, "ANLI R1 Test (200 sample)")
    assert m.accuracy > 0.40


@pytest.mark.slow
def test_anli_r2_sample():
    m = run_anli_benchmark("r2", "test", max_samples=200)
    print_nli_metrics(m, "ANLI R2 Test (200 sample)")
    assert m.accuracy > 0.35


@pytest.mark.slow
def test_anli_r3_sample():
    m = run_anli_benchmark("r3", "test", max_samples=200)
    print_nli_metrics(m, "ANLI R3 Test (200 sample)")
    assert m.accuracy > 0.35


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="ANLI adversarial NLI benchmark")
    add_common_args(parser)
    parser.add_argument("--rounds", nargs="+", default=["r1", "r2", "r3"],
                        choices=["r1", "r2", "r3"])
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    args = parser.parse_args()

    results = {"benchmark": "ANLI", "split": args.split}
    for rnd in args.rounds:
        m = run_anli_benchmark(rnd, args.split, args.max_samples, args.model)
        print_nli_metrics(m, f"ANLI {rnd.upper()} {args.split}")
        results[rnd] = m.to_dict()

    save_results(results, "anli_results.json")
