# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — HaluEval Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate CoherenceScorer against HaluEval hallucination detection dataset.

HaluEval provides (context, response) pairs labelled as hallucinated or
not, across QA, summarization, and dialogue tasks.  We score each pair
with ``CoherenceScorer`` and measure precision/recall/F1 for hallucination
detection at a threshold of 0.5.

Usage::

    # Requires DeBERTa NLI model (~2 GB download on first run)
    pytest benchmarks/halueval_eval.py -v -m slow

    # Or run directly
    python -m benchmarks.halueval_eval
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

logger = logging.getLogger("DirectorAI.Benchmark.HaluEval")

_CACHE_DIR = Path(__file__).parent / ".cache"

# HaluEval dataset URLs (HuggingFace parquet files)
_DATASET_URLS = {
    "qa": (
        "https://huggingface.co/datasets/pminervini/HaluEval/resolve/main/"
        "qa/data-00000-of-00001.parquet"
    ),
    "summarization": (
        "https://huggingface.co/datasets/pminervini/HaluEval/resolve/main/"
        "summarization/data-00000-of-00001.parquet"
    ),
    "dialogue": (
        "https://huggingface.co/datasets/pminervini/HaluEval/resolve/main/"
        "dialogue/data-00000-of-00001.parquet"
    ),
}


@dataclass
class ClassificationMetrics:
    """Binary classification metrics for hallucination detection."""

    tp: int = 0  # true positive (correctly flagged hallucination)
    fp: int = 0  # false positive (flagged non-hallucination as hallucination)
    tn: int = 0  # true negative (correctly passed non-hallucination)
    fn: int = 0  # false negative (missed hallucination)

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
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / max(total, 1)

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


@dataclass
class HaluEvalResult:
    """Aggregated HaluEval benchmark results."""

    overall: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    per_task: dict[str, ClassificationMetrics] = field(default_factory=dict)


def _download_task_data(task: str) -> list[dict]:
    """Download a HaluEval task dataset (parquet), caching locally."""
    import pandas as pd

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"halueval_{task}.parquet"

    if cache_path.exists():
        logger.info("Using cached HaluEval %s dataset", task)
        df = pd.read_parquet(cache_path)
        return df.to_dict(orient="records")

    url = _DATASET_URLS.get(task)
    if not url:
        raise ValueError(f"Unknown HaluEval task: {task}")

    import requests

    logger.info("Downloading HaluEval %s dataset...", task)
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    df = pd.read_parquet(cache_path)
    logger.info("Cached %d samples to %s", len(df), cache_path)
    return df.to_dict(orient="records")


def _extract_pairs(task: str, sample: dict) -> list[tuple[str, str, bool]]:
    """Extract (context, response, is_hallucinated) pairs from a sample.

    HaluEval format varies by task:
    - QA: {knowledge, question, right_answer, hallucinated_answer}
    - Summarization: {document, right_summary, hallucinated_summary}
    - Dialogue: {knowledge, dialogue_history, right_response, hallucinated_response}

    Returns a list of (context, response, is_hallucinated) tuples.
    """
    pairs = []

    if task == "qa":
        ctx = sample.get("knowledge", "") or sample.get("question", "")
        right = sample.get("right_answer", "")
        halluc = sample.get("hallucinated_answer", "")
        if right:
            pairs.append((ctx, right, False))
        if halluc:
            pairs.append((ctx, halluc, True))
    elif task == "summarization":
        ctx = sample.get("document", "")
        right = sample.get("right_summary", "")
        halluc = sample.get("hallucinated_summary", "")
        if right:
            pairs.append((ctx, right, False))
        if halluc:
            pairs.append((ctx, halluc, True))
    elif task == "dialogue":
        ctx = sample.get("dialogue_history", "") or sample.get("knowledge", "")
        right = sample.get("right_response", "")
        halluc = sample.get("hallucinated_response", "")
        if right:
            pairs.append((ctx, right, False))
        if halluc:
            pairs.append((ctx, halluc, True))

    return pairs


def run_halueval_benchmark(
    tasks: list[str] | None = None,
    use_nli: bool = True,
    max_samples_per_task: int | None = None,
    coherence_threshold: float = 0.5,
) -> HaluEvalResult:
    """Run HaluEval hallucination detection benchmark.

    For each (context, response) pair, we compute a coherence score.
    If the score is below ``coherence_threshold``, we flag it as
    hallucinated.  We then compare against ground truth labels.

    Parameters
    ----------
    tasks : list of task names (default: all three).
    use_nli : bool — use DeBERTa NLI model.
    max_samples_per_task : int | None — limit for quick testing.
    coherence_threshold : float — below this = flagged as hallucination.
    """
    from director_ai.core import CoherenceScorer

    scorer = CoherenceScorer(threshold=0.5, use_nli=use_nli)

    if tasks is None:
        tasks = ["qa", "summarization", "dialogue"]

    result = HaluEvalResult()

    for task in tasks:
        task_metrics = ClassificationMetrics()
        result.per_task[task] = task_metrics

        try:
            samples = _download_task_data(task)
        except Exception as e:
            logger.warning("Could not load HaluEval %s: %s", task, e)
            continue

        if max_samples_per_task:
            samples = samples[:max_samples_per_task]

        for sample in samples:
            pairs = _extract_pairs(task, sample)
            for context, response, is_hallucinated in pairs:
                if not context or not response:
                    continue

                _, score = scorer.review(context, response)
                predicted_hallucination = score.score < coherence_threshold

                if is_hallucinated and predicted_hallucination:
                    task_metrics.tp += 1
                    result.overall.tp += 1
                elif is_hallucinated and not predicted_hallucination:
                    task_metrics.fn += 1
                    result.overall.fn += 1
                elif not is_hallucinated and predicted_hallucination:
                    task_metrics.fp += 1
                    result.overall.fp += 1
                else:
                    task_metrics.tn += 1
                    result.overall.tn += 1

    return result


def _print_results(result: HaluEvalResult) -> None:
    """Pretty-print HaluEval benchmark results."""
    print("\n" + "=" * 70)
    print("HaluEval Hallucination Detection Benchmark")
    print("=" * 70)

    def _print_metrics(m: ClassificationMetrics, label: str) -> None:
        print(f"\n  {label}:")
        print(f"    Samples:   {m.total}")
        print(f"    Accuracy:  {m.accuracy:.1%}")
        print(f"    Precision: {m.precision:.1%}")
        print(f"    Recall:    {m.recall:.1%}")
        print(f"    F1:        {m.f1:.1%}")
        print(f"    (TP={m.tp}, FP={m.fp}, TN={m.tn}, FN={m.fn})")

    _print_metrics(result.overall, "Overall")

    for task, metrics in sorted(result.per_task.items()):
        _print_metrics(metrics, f"Task: {task}")

    print("=" * 70)


# ── Pytest entry points ───────────────────────────────────────────


@pytest.mark.slow
def test_halueval_qa_sample():
    """Run HaluEval QA benchmark (small sample, requires NLI model)."""
    result = run_halueval_benchmark(
        tasks=["qa"], use_nli=True, max_samples_per_task=25
    )
    _print_results(result)
    assert result.overall.total > 0, "No samples processed"
    logger.info(
        "HaluEval QA F1: %.1f%% (precision=%.1f%%, recall=%.1f%%)",
        result.overall.f1 * 100,
        result.overall.precision * 100,
        result.overall.recall * 100,
    )


@pytest.mark.slow
def test_halueval_full():
    """Full HaluEval benchmark (all tasks, ~10K samples)."""
    result = run_halueval_benchmark(use_nli=True, max_samples_per_task=200)
    _print_results(result)
    assert result.overall.total > 0


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    max_s = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    use_model = "--no-nli" not in sys.argv
    result = run_halueval_benchmark(use_nli=use_model, max_samples_per_task=max_s)
    _print_results(result)

    # Write machine-readable results
    output_path = _CACHE_DIR / "halueval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "benchmark": "HaluEval",
                "overall": {
                    "accuracy": round(result.overall.accuracy, 4),
                    "precision": round(result.overall.precision, 4),
                    "recall": round(result.overall.recall, 4),
                    "f1": round(result.overall.f1, 4),
                    "total": result.overall.total,
                },
                "per_task": {
                    task: {
                        "accuracy": round(m.accuracy, 4),
                        "precision": round(m.precision, 4),
                        "recall": round(m.recall, 4),
                        "f1": round(m.f1, 4),
                        "total": m.total,
                    }
                    for task, m in result.per_task.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")
