# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAGTruth Evaluation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Evaluate Director-AI on RAGTruth (ACL 2024) — per-sentence hallucination
labels on RAG responses.

Dataset: ``yixuantt/ragtruth`` on HuggingFace.

Usage::

    python -m benchmarks.ragtruth_eval --max-samples 100
    python -m benchmarks.ragtruth_eval --nli --max-samples 50
"""

from __future__ import annotations

import logging
import time

from benchmarks._common import save_results
from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results

logger = logging.getLogger("DirectorAI.Benchmark.RAGTruth")


def _load_ragtruth(max_samples: int | None = None) -> list[dict]:
    """Load RAGTruth dataset via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("yixuantt/ragtruth", split="test", trust_remote_code=True)
    items = list(ds)
    if max_samples:
        items = items[:max_samples]
    return items


def run_ragtruth(
    max_samples: int | None = None,
    threshold: float = 0.5,
    soft_limit: float = 0.6,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> E2EMetrics:
    """Evaluate Director-AI scorer on RAGTruth.

    Each sample has context, question, response, and hallucination labels.
    We ingest context and score the response.
    """
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    metrics = E2EMetrics()
    items = _load_ragtruth(max_samples)
    logger.info("Loaded %d RAGTruth samples", len(items))

    for item in items:
        context = item.get("source_text", item.get("context", ""))
        question = item.get("question", "")
        response = item.get("response", "")
        # Label: 1 = hallucinated, 0 = faithful
        is_hallucinated = bool(item.get("label", item.get("is_hallucinated", 0)))

        store = VectorGroundTruthStore()
        scorer = CoherenceScorer(
            threshold=threshold,
            soft_limit=soft_limit,
            use_nli=use_nli,
            ground_truth_store=store,
            nli_model=nli_model,
        )

        if context:
            store.ingest([context])

        t0 = time.perf_counter()
        approved, score = scorer.review(question or response, response)
        elapsed = time.perf_counter() - t0

        sample = E2ESample(
            task="ragtruth",
            context=context,
            question=question,
            response=response,
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
    parser = argparse.ArgumentParser(description="RAGTruth benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--nli", action="store_true")
    args = parser.parse_args()

    results = run_ragtruth(max_samples=args.max_samples, use_nli=args.nli)
    print_e2e_results(results, "RAGTruth")
    save_results(results.to_dict(), "ragtruth_results.json")
