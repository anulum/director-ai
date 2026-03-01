# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAGBench Evaluation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Evaluate Director-AI on RAGBench (rungalileo/ragbench) — real RAG pipeline
outputs with grounded/hallucinated labels across 5 domains.

Requires ``datasets`` and optionally ``HF_TOKEN`` env var.

Usage::

    python -m benchmarks.ragbench_eval --max-samples 100
    python -m benchmarks.ragbench_eval --nli --max-samples 50
"""

from __future__ import annotations

import logging
import time

from benchmarks._common import save_results
from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results

logger = logging.getLogger("DirectorAI.Benchmark.RAGBench")

# RAGBench subsets (domains)
RAGBENCH_SUBSETS = [
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
]


def _load_ragbench(subset: str, max_samples: int | None = None) -> list[dict]:
    """Load a RAGBench subset via HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset(
        "rungalileo/ragbench",
        subset,
        split="test",
        trust_remote_code=True,
    )
    items = list(ds)
    if max_samples:
        items = items[:max_samples]
    return items


def run_ragbench(
    subsets: list[str] | None = None,
    max_samples: int | None = None,
    threshold: float = 0.5,
    soft_limit: float = 0.6,
    use_nli: bool = False,
    nli_model: str | None = None,
) -> E2EMetrics:
    """Evaluate Director-AI scorer on RAGBench.

    Each sample has context, response, and a hallucination label.
    We ingest context into VectorGroundTruthStore and score the response.
    """
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    if subsets is None:
        subsets = RAGBENCH_SUBSETS

    store = VectorGroundTruthStore(auto_index=True)
    scorer = CoherenceScorer(
        threshold=threshold,
        soft_limit=soft_limit,
        use_nli=use_nli,
        ground_truth_store=store,
        nli_model=nli_model,
    )

    metrics = E2EMetrics(
        threshold=threshold,
        soft_limit=soft_limit,
    )

    for subset in subsets:
        try:
            samples = _load_ragbench(subset, max_samples=max_samples)
        except (ImportError, OSError, ValueError, KeyError) as e:
            logger.warning("Could not load RAGBench %s: %s", subset, e)
            continue

        logger.info("Evaluating %s: %d samples", subset, len(samples))

        for item in samples:
            context = item.get("context", "") or ""
            response = item.get("response", "") or ""
            # RAGBench labels: 1 = hallucinated, 0 = grounded
            is_hallucinated = bool(item.get("label", 0))

            if not context or not response:
                continue

            store.ingest([context])

            t0 = time.perf_counter()
            approved, score = scorer.review(context, response)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            has_evidence = score.evidence is not None
            n_chunks = len(score.evidence.chunks) if score.evidence else 0

            e2e_sample = E2ESample(
                task=subset,
                context=context[:200],
                response=response[:200],
                is_hallucinated=is_hallucinated,
                coherence_score=score.score,
                approved=approved,
                warning=score.warning,
                has_evidence=has_evidence,
                evidence_chunks=n_chunks,
                latency_ms=elapsed_ms,
            )
            metrics.samples.append(e2e_sample)

    return metrics


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Director-AI RAGBench evaluation")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per domain (default: all)",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--soft-limit", type=float, default=0.6)
    parser.add_argument("--nli", action="store_true", help="Enable NLI model")
    parser.add_argument("--nli-model", type=str, default=None)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        choices=RAGBENCH_SUBSETS,
        help="Which RAGBench subsets to evaluate",
    )
    args = parser.parse_args()

    m = run_ragbench(
        subsets=args.subsets,
        max_samples=args.max_samples,
        threshold=args.threshold,
        soft_limit=args.soft_limit,
        use_nli=args.nli,
        nli_model=args.nli_model,
    )
    print_e2e_results(m)
    save_results(
        {"benchmark": "RAGBench", **m.to_dict()},
        "ragbench_eval.json",
    )
