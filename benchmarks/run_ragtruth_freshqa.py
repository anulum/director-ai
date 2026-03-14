#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — RAGTruth + FreshQA Benchmarks (fixed loaders)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""RAGTruth and FreshQA with corrected dataset sources.

- RAGTruth: wandb/RAGTruth-processed (HuggingFace)
  Fields: context, query, output, hallucination_labels_processed
- FreshQA: Google Sheets CSV export (Nov 2025 snapshot)
  Fields: question, false_premise, answer_0..answer_9
"""

from __future__ import annotations

import ast
import csv
import io
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("CloudBench")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FRESHQA_SHEET_ID = "1X6oTXzU1L9PWc2uim1eVzdX8V7y7_4crWfdJhVV08L4"


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

    scorer = CoherenceScorer(threshold=0.5, soft_limit=0.6, use_nli=True)

    metrics = E2EMetrics()
    for i, item in enumerate(items):
        context = item.get("context", "")
        question = item.get("query", "")
        response = item.get("output", "")

        labels_raw = item.get("hallucination_labels_processed", "{}")
        if isinstance(labels_raw, str):
            labels = ast.literal_eval(labels_raw)
        else:
            labels = labels_raw
        is_hallucinated = (labels.get("evident_conflict", 0) > 0) or (
            labels.get("baseless_info", 0) > 0
        )

        if not response:
            continue

        store = VectorGroundTruthStore()
        scorer._ground_truth_store = store
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
    """FreshQA via Google Sheets CSV export."""
    from benchmarks.e2e_eval import E2EMetrics, E2ESample, print_e2e_results
    from director_ai.core.scorer import CoherenceScorer

    logger.info("=== FreshQA (NLI, Google Sheets CSV) ===")
    url = f"https://docs.google.com/spreadsheets/d/{FRESHQA_SHEET_ID}/export?format=csv"
    resp = urllib.request.urlopen(url)
    text = resp.read().decode("utf-8")

    # Skip warning rows (first 2 lines) before the actual header
    lines = text.split("\n")
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("id,") or line.startswith('"id",'):
            header_idx = idx
            break
    if header_idx is None:
        logger.error("Could not find header row in FreshQA CSV")
        return

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text))
    items = list(reader)
    logger.info("Loaded %d FreshQA samples", len(items))

    scorer = CoherenceScorer(threshold=0.5, soft_limit=0.6, use_nli=True)

    metrics = E2EMetrics()
    for i, item in enumerate(items):
        question = item.get("question", "")
        is_false_premise = str(item.get("false_premise", "")).upper() == "TRUE"

        # Collect all non-empty answers
        answers = []
        for k in range(10):
            a = item.get(f"answer_{k}", "")
            if a and a.strip():
                answers.append(a.strip())
        if not answers or not question:
            continue

        answer = answers[0]  # Use first (primary) answer
        is_hallucinated = is_false_premise

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
