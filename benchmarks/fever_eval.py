# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — FEVER Dev Benchmark (Held-Out Fact Verification)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate NLI model on FEVER dev set (fact verification).

Training included FEVER train (pietrolesci/nli_fever train split).
The dev split is held-out. FEVER tests whether the model can verify
claims against evidence passages — directly relevant to RAG
hallucination detection.

Usage::

    python -m benchmarks.fever_eval 500
    python -m benchmarks.fever_eval --model training/output/deberta-v3-base-hallucination
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

logger = logging.getLogger("DirectorAI.Benchmark.FEVER")

_LABEL_MAP = {"entailment": 0, "neutral": 1, "contradiction": 2}


def _load_fever_dev() -> list[dict]:
    logger.info("Loading FEVER NLI dev split ...")
    ds = load_dataset("pietrolesci/nli_fever", split="dev")
    return list(ds)


def run_fever_benchmark(
    max_samples: int | None = None,
    model_name: str | None = None,
) -> NLIMetrics:
    predictor = NLIPredictor(model_name=model_name)
    rows = _load_fever_dev()
    if max_samples:
        rows = rows[:max_samples]

    metrics = NLIMetrics()
    for row in rows:
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        raw_label = row.get("label")

        if isinstance(raw_label, str):
            label = _LABEL_MAP.get(raw_label.lower())
        elif isinstance(raw_label, int):
            label = raw_label
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
def test_fever_dev_sample():
    m = run_fever_benchmark(max_samples=200)
    print_nli_metrics(m, "FEVER Dev (200 sample)")
    assert m.accuracy > 0.60


@pytest.mark.slow
def test_fever_dev_full():
    m = run_fever_benchmark()
    print_nli_metrics(m, "FEVER Dev (full)")
    assert m.accuracy > 0.60


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="FEVER fact-verification benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    m = run_fever_benchmark(max_samples=args.max_samples, model_name=args.model)
    print_nli_metrics(m, "FEVER Dev")

    save_results({"benchmark": "FEVER_dev", **m.to_dict()}, "fever_results.json")
