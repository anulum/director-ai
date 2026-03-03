# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — FreshQA Evaluation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate Director-AI on FreshQA — question/answer pairs with validity
labels (valid, false_premise, outdated).

Dataset: ``freshllms/freshqa`` on HuggingFace (or manual CSV).

Usage::

    python -m benchmarks.freshqa_eval --max-samples 100
    python -m benchmarks.freshqa_eval --nli
"""

from __future__ import annotations

import logging
import time

from benchmarks._common import save_results
from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results

logger = logging.getLogger("DirectorAI.Benchmark.FreshQA")


def _load_freshqa(max_samples: int | None = None) -> list[dict]:
    """Load FreshQA dataset via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("freshllms/freshqa", split="train", trust_remote_code=True)
    items = list(ds)
    if max_samples:
        items = items[:max_samples]
    return items


def run_freshqa(
    max_samples: int | None = None,
    threshold: float = 0.5,
    soft_limit: float = 0.6,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> E2EMetrics:
    """Evaluate Director-AI on FreshQA.

    Each sample has a question, answer, and validity label.
    Answers with false_premise or outdated labels are treated as hallucinated.
    """
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    metrics = E2EMetrics()
    items = _load_freshqa(max_samples)
    logger.info("Loaded %d FreshQA samples", len(items))

    for item in items:
        question = item.get("question", "")
        answer = item.get("answer", item.get("best_answer", ""))
        # false_premise or outdated → hallucinated
        validity = item.get("validity", item.get("label", "valid"))
        is_hallucinated = validity in ("false_premise", "outdated", False)

        store = VectorGroundTruthStore()
        scorer = CoherenceScorer(
            threshold=threshold,
            soft_limit=soft_limit,
            use_nli=use_nli,
            ground_truth_store=store,
            nli_model=nli_model,
        )

        t0 = time.perf_counter()
        approved, score = scorer.review(question, answer)
        elapsed = time.perf_counter() - t0

        sample = E2ESample(
            task="freshqa",
            context="",
            question=question,
            response=answer,
            is_hallucinated=is_hallucinated,
            approved=approved,
            score=score.score,
            latency_ms=elapsed * 1000,
        )
        metrics.add(sample)

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="FreshQA benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--nli", action="store_true")
    args = parser.parse_args()

    results = run_freshqa(max_samples=args.max_samples, use_nli=args.nli)
    print_e2e_results(results, "FreshQA")
    save_results(results.to_dict(), "freshqa_results.json")
