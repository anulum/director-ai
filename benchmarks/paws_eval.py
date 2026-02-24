# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — PAWS Benchmark (Adversarial Paraphrase Detection)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate NLI model on PAWS (Paraphrase Adversaries from Word Scrambling).

PAWS tests whether the model confuses high lexical overlap with semantic
equivalence. Pairs with nearly identical words can have completely
different meanings. This is critical for hallucination detection: a
response that copies most of the source but changes one key fact should
be flagged as contradiction, not entailment.

NOT in training data — fully held-out.

We map PAWS binary labels to NLI:
  - paraphrase (label=1) → entailment
  - non-paraphrase (label=0) → contradiction/neutral

Usage::

    python -m benchmarks.paws_eval 500
    python -m benchmarks.paws_eval --model training/output/deberta-v3-base-hallucination
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pytest
from datasets import load_dataset

from benchmarks._common import NLIPredictor, RESULTS_DIR, add_common_args, save_results

logger = logging.getLogger("DirectorAI.Benchmark.PAWS")


@dataclass
class PAWSMetrics:
    """Binary paraphrase detection metrics mapped from NLI."""

    total: int = 0
    tp: int = 0  # paraphrase correctly identified as entailment
    fp: int = 0  # non-paraphrase incorrectly identified as entailment
    tn: int = 0  # non-paraphrase correctly identified as non-entailment
    fn: int = 0  # paraphrase missed (predicted non-entailment)
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / max(self.total, 1)

    @property
    def precision(self) -> float:
        return self.tp / max(self.tp + self.fp, 1)

    @property
    def recall(self) -> float:
        return self.tp / max(self.tp + self.fn, 1)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-10)

    @property
    def adversarial_accuracy(self) -> float:
        """Accuracy on adversarial non-paraphrases only (high overlap but different meaning)."""
        return self.tn / max(self.tn + self.fp, 1)

    @property
    def avg_latency_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times)) * 1000

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "adversarial_accuracy": round(self.adversarial_accuracy, 4),
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "latency_ms_avg": round(self.avg_latency_ms, 2),
        }


def run_paws_benchmark(
    max_samples: int | None = None,
    model_name: str | None = None,
) -> PAWSMetrics:
    predictor = NLIPredictor(model_name=model_name)

    logger.info("Loading PAWS test set ...")
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    rows = list(ds)
    if max_samples:
        rows = rows[:max_samples]

    metrics = PAWSMetrics()
    for row in rows:
        sent1 = row.get("sentence1", "")
        sent2 = row.get("sentence2", "")
        label = row.get("label")  # 1 = paraphrase, 0 = non-paraphrase
        if label is None or not sent1 or not sent2:
            continue

        is_paraphrase = label == 1

        t0 = time.perf_counter()
        pred = predictor.predict(sent1, sent2)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.total += 1

        predicted_entailment = pred == 0  # entailment = paraphrase

        if is_paraphrase and predicted_entailment:
            metrics.tp += 1
        elif is_paraphrase and not predicted_entailment:
            metrics.fn += 1
        elif not is_paraphrase and predicted_entailment:
            metrics.fp += 1
        else:
            metrics.tn += 1

    return metrics


def _print_paws_results(m: PAWSMetrics) -> None:
    print(f"\n{'=' * 65}")
    print("  PAWS — Adversarial Paraphrase Detection")
    print(f"{'=' * 65}")
    print(f"  Samples:              {m.total}")
    print(f"  Accuracy:             {m.accuracy:.1%}")
    print(f"  F1:                   {m.f1:.4f}")
    print(f"  Precision:            {m.precision:.4f}")
    print(f"  Recall:               {m.recall:.4f}")
    print(f"  Adversarial Acc:      {m.adversarial_accuracy:.1%}  (non-paraphrase detection)")
    print(f"  (TP={m.tp} FP={m.fp} TN={m.tn} FN={m.fn})")
    if m.inference_times:
        print(f"  Latency:              {m.avg_latency_ms:.1f} ms avg")
    print(f"{'=' * 65}")


# ── Pytest ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_paws_sample():
    m = run_paws_benchmark(max_samples=200)
    _print_paws_results(m)
    assert m.accuracy > 0.50


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="PAWS adversarial paraphrase benchmark")
    add_common_args(parser)
    args = parser.parse_args()

    m = run_paws_benchmark(max_samples=args.max_samples, model_name=args.model)
    _print_paws_results(m)

    save_results({"benchmark": "PAWS", **m.to_dict()}, "paws_results.json")
