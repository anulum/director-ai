#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAGTruth + FreshQA Benchmarks (fixed loaders)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
RAGTruth and FreshQA with corrected dataset sources.

- RAGTruth: wandb/RAGTruth-processed (HuggingFace)
- FreshQA: freshllms/freshqa GitHub CSV
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("CloudBench")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _gpu_info() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"gpu": "none", "cuda": False}
    return {
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
        "cuda": True,
        "torch": torch.__version__,
    }


def _save(data: dict, name: str) -> None:
    path = RESULTS_DIR / name
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %s", path)


def bench_ragtruth_nli() -> None:
    """RAGTruth via wandb/RAGTruth-processed."""
    from datasets import load_dataset

    from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    logger.info("=== RAGTruth (NLI, wandb/RAGTruth-processed) ===")
    ds = load_dataset("wandb/RAGTruth-processed", split="test")
    items = list(ds)
    logger.info("Loaded %d RAGTruth samples", len(items))

    metrics = E2EMetrics()
    for i, item in enumerate(items):
        context = item.get("source", item.get("context", item.get("source_text", "")))
        question = item.get("query", item.get("question", ""))
        response = item.get("response", item.get("answer", ""))
        is_hallucinated = bool(item.get("label", item.get("is_hallucinated", 0)))

        if not response:
            continue

        store = VectorGroundTruthStore()
        scorer = CoherenceScorer(
            threshold=0.5,
            soft_limit=0.6,
            use_nli=True,
            ground_truth_store=store,
        )
        if context:
            store.ingest([context])

        t0 = time.perf_counter()
        approved, score = scorer.review(question or response, response)
        elapsed = time.perf_counter() - t0

        sample = E2ESample(
            task="ragtruth",
            context=context[:200] if context else "",
            response=response[:200],
            is_hallucinated=is_hallucinated,
            approved=approved,
            coherence_score=score.score,
            latency_ms=elapsed * 1000,
        )
        metrics.samples.append(sample)

        if (i + 1) % 100 == 0:
            logger.info(
                "RAGTruth progress: %d/%d (catch=%.1f%%)",
                i + 1,
                len(items),
                metrics.catch_rate * 100,
            )

    print_e2e_results(metrics)
    result = {"benchmark": "RAGTruth-NLI", **metrics.to_dict(), "hw": _gpu_info()}
    _save(result, "ragtruth_nli_results.json")


def bench_freshqa_nli() -> None:
    """FreshQA via GitHub CSV."""
    import csv
    import io
    import urllib.request

    from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    logger.info("=== FreshQA (NLI, GitHub CSV) ===")
    url = "https://raw.githubusercontent.com/freshllms/freshqa/main/data/freshqa.csv"
    resp = urllib.request.urlopen(url)
    text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    items = list(reader)
    logger.info("Loaded %d FreshQA samples", len(items))

    metrics = E2EMetrics()
    for i, item in enumerate(items):
        question = item.get("question", "")
        answer = item.get("answer", item.get("best_answer", ""))
        validity = item.get("validity", item.get("label", "valid"))
        is_hallucinated = validity in ("false_premise", "outdated", "False", "false")

        if not answer or not question:
            continue

        store = VectorGroundTruthStore()
        scorer = CoherenceScorer(
            threshold=0.5,
            soft_limit=0.6,
            use_nli=True,
            ground_truth_store=store,
        )

        t0 = time.perf_counter()
        approved, score = scorer.review(question, answer)
        elapsed = time.perf_counter() - t0

        sample = E2ESample(
            task="freshqa",
            context="",
            response=answer[:200],
            is_hallucinated=is_hallucinated,
            approved=approved,
            coherence_score=score.score,
            latency_ms=elapsed * 1000,
        )
        metrics.samples.append(sample)

        if (i + 1) % 50 == 0:
            logger.info(
                "FreshQA progress: %d/%d (catch=%.1f%%)",
                i + 1,
                len(items),
                metrics.catch_rate * 100,
            )

    print_e2e_results(metrics)
    result = {"benchmark": "FreshQA-NLI", **metrics.to_dict(), "hw": _gpu_info()}
    _save(result, "freshqa_nli_results.json")


if __name__ == "__main__":
    logger.info("GPU: %s", json.dumps(_gpu_info()))
    bench_ragtruth_nli()
    bench_freshqa_nli()
    logger.info("=== RAGTruth + FreshQA DONE ===")
