# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth Evaluation

"""Evaluate Director-AI on RAGTruth-style per-sample hallucination labels.

Primary dataset source: ``wandb/RAGTruth-processed`` on HuggingFace.
Fallback source: ``flowaicom/RAGTruth_test``.

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

    candidates: list[tuple[str, str]] = [
        ("wandb/RAGTruth-processed", "test"),
        ("flowaicom/RAGTruth_test", "qa"),
    ]
    items: list[dict] = []
    load_errors: list[str] = []
    for dataset_id, split in candidates:
        try:
            ds = load_dataset(dataset_id, split=split)
            items = list(ds)
            if items:
                break
        except Exception as exc:
            load_errors.append(f"{dataset_id}:{split} -> {exc}")
    if not items:
        raise RuntimeError(
            "Could not load any RAGTruth dataset source. "
            f"Attempts: {' | '.join(load_errors)}"
        )
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
        question = item.get("question", item.get("query", ""))
        response = item.get("response", item.get("output", ""))
        labels = item.get("hallucination_labels_processed", {}) or {}
        # Label semantics:
        # - wandb/RAGTruth-processed: counts in hallucination_labels_processed
        # - fallback sets may expose explicit bool/int labels.
        is_hallucinated = bool(
            item.get("label", item.get("is_hallucinated", 0))
            or labels.get("evident_conflict", 0)
            or labels.get("baseless_info", 0)
        )

        if not response:
            continue

        store = VectorGroundTruthStore()
        if context:
            store.ingest([context])

        scorer = CoherenceScorer(
            threshold=threshold,
            soft_limit=soft_limit,
            use_nli=use_nli,
            ground_truth_store=store,
            nli_model=nli_model,
        )

        t0 = time.perf_counter()
        approved, score = scorer.review(question or response, response)
        elapsed = time.perf_counter() - t0

        sample = E2ESample(
            task="ragtruth",
            context=context,
            response=response,
            is_hallucinated=is_hallucinated,
            approved=approved,
            coherence_score=score.score,
            latency_ms=elapsed * 1000,
        )
        metrics.samples.append(sample)

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="RAGTruth benchmark")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--nli", action="store_true")
    args = parser.parse_args()

    results = run_ragtruth(max_samples=args.max_samples, use_nli=args.nli)
    print_e2e_results(results)
    save_results(results.to_dict(), "ragtruth_results.json")
