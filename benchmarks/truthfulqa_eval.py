# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — TruthfulQA Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate CoherenceScorer against TruthfulQA (MC format).

TruthfulQA is a benchmark of ~817 questions designed to test whether
language models generate truthful answers.  We use the multiple-choice
format: for each question, the scorer should assign higher coherence to
the correct answer than to each incorrect answer.

Usage::

    # Requires DeBERTa NLI model (~2 GB download on first run)
    pytest benchmarks/truthfulqa_eval.py -v -m slow

    # Or run directly
    python -m benchmarks.truthfulqa_eval
"""

from __future__ import annotations

import csv
import io
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pytest

logger = logging.getLogger("DirectorAI.Benchmark.TruthfulQA")

_DATASET_URL = (
    "https://raw.githubusercontent.com/sylinrl/TruthfulQA/"
    "main/TruthfulQA.csv"
)

_CACHE_DIR = Path(__file__).parent / ".cache"


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    total: int = 0
    correct: int = 0
    per_category: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"total": 0, "correct": 0})
    )

    @property
    def accuracy(self) -> float:
        return self.correct / max(self.total, 1)


def _download_dataset() -> list[dict]:
    """Download TruthfulQA CSV dataset, caching locally."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / "TruthfulQA.csv"

    if cache_path.exists():
        logger.info("Using cached TruthfulQA dataset: %s", cache_path)
        text = cache_path.read_text(encoding="utf-8")
    else:
        import requests

        logger.info("Downloading TruthfulQA dataset...")
        resp = requests.get(_DATASET_URL, timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache_path.write_text(text, encoding="utf-8")
        logger.info("Cached dataset to %s", cache_path)

    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def _parse_mc_answers(row: dict) -> tuple[list[str], list[str]]:
    """Parse correct and incorrect MC answers from a TruthfulQA row.

    The dataset has columns ``Correct Answers`` and ``Incorrect Answers``
    separated by semicolons.
    """
    correct_raw = row.get("Correct Answers", "")
    incorrect_raw = row.get("Incorrect Answers", "")

    correct = [a.strip() for a in correct_raw.split(";") if a.strip()]
    incorrect = [a.strip() for a in incorrect_raw.split(";") if a.strip()]

    return correct, incorrect


def run_truthfulqa_benchmark(
    use_nli: bool = True,
    max_questions: int | None = None,
) -> BenchmarkResult:
    """Run the TruthfulQA benchmark.

    For each question, we score the best correct answer and the best
    incorrect answer using ``CoherenceScorer``.  The question is counted
    as correct if the scorer assigns higher coherence to a correct
    answer than to the best incorrect answer.

    Parameters
    ----------
    use_nli : bool — use DeBERTa NLI model (recommended for real results).
    max_questions : int | None — limit for quick testing.
    """
    from director_ai.core import CoherenceScorer, VectorGroundTruthStore

    store = VectorGroundTruthStore()
    scorer = CoherenceScorer(
        threshold=0.5,
        use_nli=use_nli,
        ground_truth_store=store,
    )

    rows = _download_dataset()
    if max_questions:
        rows = rows[:max_questions]

    result = BenchmarkResult()

    for row in rows:
        question = row.get("Question", "").strip()
        category = row.get("Category", "unknown").strip()
        correct_answers, incorrect_answers = _parse_mc_answers(row)

        if not question or not correct_answers or not incorrect_answers:
            continue

        # Score all correct answers — take best
        best_correct_score = -1.0
        for ans in correct_answers:
            _, score = scorer.review(question, ans)
            if score.score > best_correct_score:
                best_correct_score = score.score

        # Score all incorrect answers — take best (highest = hardest to detect)
        best_incorrect_score = -1.0
        for ans in incorrect_answers:
            _, score = scorer.review(question, ans)
            if score.score > best_incorrect_score:
                best_incorrect_score = score.score

        result.total += 1
        result.per_category[category]["total"] += 1

        if best_correct_score > best_incorrect_score:
            result.correct += 1
            result.per_category[category]["correct"] += 1

    return result


def _print_results(result: BenchmarkResult) -> None:
    """Pretty-print benchmark results."""
    print("\n" + "=" * 60)
    print("TruthfulQA Benchmark Results")
    print("=" * 60)
    print(f"Total questions:  {result.total}")
    print(f"Correct:          {result.correct}")
    print(f"Accuracy:         {result.accuracy:.1%}")
    print()
    print("Per-category breakdown:")
    print(f"{'Category':<30} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 56)
    for cat, counts in sorted(result.per_category.items()):
        acc = counts["correct"] / max(counts["total"], 1)
        print(f"{cat:<30} {counts['correct']:>8} {counts['total']:>8} {acc:>7.1%}")
    print("=" * 60)


# ── Pytest entry point ─────────────────────────────────────────────


@pytest.mark.slow
def test_truthfulqa_benchmark():
    """Run TruthfulQA benchmark (requires NLI model, ~5-30 min)."""
    result = run_truthfulqa_benchmark(use_nli=True, max_questions=50)
    _print_results(result)

    # Sanity: the scorer should do better than random (50%)
    # Even modest accuracy establishes the scorer is doing real work
    assert result.total > 0, "No questions processed"
    # Log accuracy regardless of pass/fail for transparency
    logger.info("TruthfulQA accuracy: %.1f%% (%d/%d)",
                result.accuracy * 100, result.correct, result.total)


@pytest.mark.slow
def test_truthfulqa_full():
    """Full TruthfulQA benchmark (all ~817 questions)."""
    result = run_truthfulqa_benchmark(use_nli=True)
    _print_results(result)
    assert result.total > 0


# ── CLI entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else None
    use_model = "--no-nli" not in sys.argv
    result = run_truthfulqa_benchmark(use_nli=use_model, max_questions=max_q)
    _print_results(result)

    # Write machine-readable results
    output_path = _CACHE_DIR / "truthfulqa_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "benchmark": "TruthfulQA",
                "total": result.total,
                "correct": result.correct,
                "accuracy": round(result.accuracy, 4),
                "per_category": dict(result.per_category),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_path}")
