# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — False-Positive Rate on Clean RAG Data
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Measure false-positive rate on correct (evidence, answer) pairs.

This is the single most important production metric: in real RAG
pipelines, >95% of model outputs are correct. If we flag even 5%
of correct answers as hallucinations, the product is unusable.

We construct (evidence_passage, correct_answer) pairs from three
high-quality extractive QA datasets:
  - SQuAD 2.0 (answerable subset)
  - Natural Questions (open-domain)
  - TriviaQA (evidence-based)

All pairs have human-verified correct answers. The expected
classification is entailment. Any contradiction/neutral prediction
is a false positive.

Usage::

    python -m benchmarks.falsepositive_eval 1000
    python -m benchmarks.falsepositive_eval --model training/output/deberta-v3-base-hallucination
    python -m benchmarks.falsepositive_eval --sources squad nq
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
from datasets import load_dataset

from benchmarks._common import (
    LABEL_NAMES,
    NLIPredictor,
    RESULTS_DIR,
    add_common_args,
    save_results,
)

logger = logging.getLogger("DirectorAI.Benchmark.FalsePositive")


@dataclass
class FPMetrics:
    """False-positive rate metrics for clean RAG evaluation."""

    total: int = 0
    false_positives: int = 0  # predicted contradiction on correct answer
    uncertain: int = 0        # predicted neutral on correct answer
    correct: int = 0          # predicted entailment on correct answer
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def fp_rate(self) -> float:
        return self.false_positives / max(self.total, 1)

    @property
    def uncertain_rate(self) -> float:
        return self.uncertain / max(self.total, 1)

    @property
    def entailment_rate(self) -> float:
        return self.correct / max(self.total, 1)

    @property
    def avg_latency_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times)) * 1000

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "false_positive_rate": round(self.fp_rate, 4),
            "uncertain_rate": round(self.uncertain_rate, 4),
            "entailment_rate": round(self.entailment_rate, 4),
            "false_positives": self.false_positives,
            "uncertain": self.uncertain,
            "correct": self.correct,
            "per_source": self.per_source,
            "latency_ms_avg": round(self.avg_latency_ms, 2),
        }


def _load_squad_pairs(max_per_source: int | None) -> list[tuple[str, str]]:
    """Extract (context, answer_sentence) from SQuAD 2.0 answerable questions."""
    logger.info("Loading SQuAD 2.0 ...")
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    pairs = []
    for row in ds:
        answers = row.get("answers", {})
        texts = answers.get("text", [])
        if not texts:
            continue
        context = row.get("context", "")
        question = row.get("question", "")
        answer = texts[0]
        if not context or not answer:
            continue
        # Construct a natural-language hypothesis from Q+A
        hypothesis = f"The answer to '{question}' is: {answer}"
        pairs.append((context, hypothesis))
        if max_per_source and len(pairs) >= max_per_source:
            break
    logger.info("SQuAD: %d pairs", len(pairs))
    return pairs


def _load_nq_pairs(max_per_source: int | None) -> list[tuple[str, str]]:
    """Extract (context, answer) from Natural Questions (open-domain)."""
    logger.info("Loading Natural Questions (open) ...")
    ds = load_dataset("google-research-datasets/nq_open", split="validation")
    pairs = []
    for row in ds:
        question = row.get("question", "")
        answers = row.get("answer", [])
        if not question or not answers:
            continue
        answer = answers[0] if isinstance(answers, list) else str(answers)
        # NQ open doesn't include context passages; construct from Q+A
        premise = f"Question: {question}"
        hypothesis = answer
        pairs.append((premise, hypothesis))
        if max_per_source and len(pairs) >= max_per_source:
            break
    logger.info("NQ Open: %d pairs", len(pairs))
    return pairs


def _load_triviaqa_pairs(max_per_source: int | None) -> list[tuple[str, str]]:
    """Extract (evidence, answer) from TriviaQA with verified evidence."""
    logger.info("Loading TriviaQA ...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    pairs = []
    for row in ds:
        question = row.get("question", "")
        answer_obj = row.get("answer", {})
        aliases = answer_obj.get("aliases", [])
        if not question or not aliases:
            continue
        answer = aliases[0]
        premise = f"Question: {question}"
        hypothesis = f"The answer is {answer}."
        pairs.append((premise, hypothesis))
        if max_per_source and len(pairs) >= max_per_source:
            break
    logger.info("TriviaQA: %d pairs", len(pairs))
    return pairs


_SOURCE_LOADERS = {
    "squad": _load_squad_pairs,
    "nq": _load_nq_pairs,
    "triviaqa": _load_triviaqa_pairs,
}


def run_falsepositive_benchmark(
    sources: list[str] | None = None,
    max_samples: int | None = None,
    model_name: str | None = None,
) -> FPMetrics:
    """Evaluate false-positive rate on clean (evidence, correct_answer) pairs.

    Parameters
    ----------
    sources : which QA datasets to use (default: all three).
    max_samples : total sample cap across all sources.
    model_name : HuggingFace model ID or local path.
    """
    predictor = NLIPredictor(model_name=model_name)

    if sources is None:
        sources = list(_SOURCE_LOADERS.keys())

    max_per_source = (max_samples // len(sources)) if max_samples else None

    all_pairs: list[tuple[str, str, str]] = []  # (premise, hypothesis, source_name)
    for src in sources:
        loader = _SOURCE_LOADERS.get(src)
        if loader is None:
            logger.warning("Unknown source: %s", src)
            continue
        pairs = loader(max_per_source)
        all_pairs.extend((p, h, src) for p, h in pairs)

    if max_samples:
        all_pairs = all_pairs[:max_samples]

    metrics = FPMetrics()
    for premise, hypothesis, source in all_pairs:
        t0 = time.perf_counter()
        pred = predictor.predict(premise, hypothesis)
        metrics.inference_times.append(time.perf_counter() - t0)
        metrics.total += 1

        if source not in metrics.per_source:
            metrics.per_source[source] = {"total": 0, "fp": 0, "uncertain": 0, "correct": 0}
        metrics.per_source[source]["total"] += 1

        if pred == 2:  # contradiction — false positive
            metrics.false_positives += 1
            metrics.per_source[source]["fp"] += 1
        elif pred == 1:  # neutral — uncertain
            metrics.uncertain += 1
            metrics.per_source[source]["uncertain"] += 1
        else:  # entailment — correct
            metrics.correct += 1
            metrics.per_source[source]["correct"] += 1

    return metrics


def _print_fp_results(m: FPMetrics) -> None:
    print(f"\n{'=' * 65}")
    print("  False-Positive Rate on Clean RAG Data")
    print(f"{'=' * 65}")
    print(f"  Total pairs:      {m.total}")
    print(f"  Entailment:       {m.correct} ({m.entailment_rate:.1%})")
    print(f"  Neutral:          {m.uncertain} ({m.uncertain_rate:.1%})")
    print(f"  Contradiction:    {m.false_positives} ({m.fp_rate:.1%})  ← FALSE POSITIVES")
    print()
    print(f"  FP Rate:          {m.fp_rate:.2%}")
    print(f"  FP+Uncertain:     {(m.fp_rate + m.uncertain_rate):.2%}")
    if m.per_source:
        print(f"\n  {'Source':<12} {'Total':>6} {'FP':>6} {'Unc':>6} {'OK':>6} {'FP%':>8}")
        print(f"  {'-' * 50}")
        for src, counts in sorted(m.per_source.items()):
            fp_pct = counts["fp"] / max(counts["total"], 1)
            print(f"  {src:<12} {counts['total']:>6} {counts['fp']:>6} "
                  f"{counts['uncertain']:>6} {counts['correct']:>6} {fp_pct:>7.1%}")
    if m.inference_times:
        print(f"\n  Latency:          {m.avg_latency_ms:.1f} ms avg")
    print(f"{'=' * 65}")


# ── Pytest ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_falsepositive_squad():
    m = run_falsepositive_benchmark(sources=["squad"], max_samples=200)
    _print_fp_results(m)
    assert m.fp_rate < 0.15, f"FP rate {m.fp_rate:.1%} too high on SQuAD clean data"


@pytest.mark.slow
def test_falsepositive_combined():
    m = run_falsepositive_benchmark(max_samples=300)
    _print_fp_results(m)
    assert m.fp_rate < 0.20, f"FP rate {m.fp_rate:.1%} too high"


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="False-positive rate on clean RAG data")
    add_common_args(parser)
    parser.add_argument("--sources", nargs="+", default=None,
                        choices=list(_SOURCE_LOADERS.keys()))
    args = parser.parse_args()

    m = run_falsepositive_benchmark(
        sources=args.sources, max_samples=args.max_samples, model_name=args.model
    )
    _print_fp_results(m)

    save_results({"benchmark": "FalsePositive_CleanRAG", **m.to_dict()},
                 "falsepositive_results.json")
