# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — HaluEval benchmark evaluation
"""Evaluate Director-AI on the HaluEval benchmark.

HaluEval (Li et al., 2023) is a large-scale hallucination evaluation
benchmark with 35K samples across QA, dialogue, and summarisation.

Usage::

    python benchmarks/eval_halueval.py --backend deberta --split qa
    python benchmarks/eval_halueval.py --backend lite --split all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_halueval(split: str = "qa", max_samples: int | None = None):
    """Load HaluEval dataset from HuggingFace.

    Splits: qa, dialogue, summarization, general, all.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets required")
        sys.exit(1)

    if split == "all":
        splits = ["qa", "dialogue", "summarization", "general"]
    else:
        splits = [split]

    samples = []
    for s in splits:
        ds_name = "pminervini/HaluEval"
        try:
            ds = load_dataset(ds_name, s, split="data")
            for item in ds:
                samples.append(
                    {
                        "split": s,
                        "knowledge": item.get("knowledge", item.get("document", "")),
                        "right_answer": item.get(
                            "right_answer", item.get("right_response", "")
                        ),
                        "hallucinated_answer": item.get(
                            "hallucinated_answer", item.get("hallucinated_response", "")
                        ),
                        "question": item.get(
                            "question", item.get("dialogue_history", "")
                        ),
                    }
                )
                if max_samples and len(samples) >= max_samples:
                    break
        except Exception as exc:
            logger.warning("Failed to load split %s: %s", s, exc)

    logger.info("Loaded %d HaluEval samples (%s)", len(samples), split)
    return samples


def evaluate(samples: list[dict], backend: str, threshold: float) -> dict:
    """Run Director-AI scorer on HaluEval samples.

    Returns accuracy metrics.
    """
    from director_ai.core.scoring.scorer import CoherenceScorer

    scorer = CoherenceScorer(
        threshold=threshold,
        use_nli=(backend != "lite"),
        scorer_backend=backend,
    )

    tp = fp = tn = fn = 0
    latencies: list[float] = []

    for i, s in enumerate(samples):
        context = s["knowledge"]

        # Score right answer (should be approved)
        t0 = time.perf_counter()
        approved_right, _ = scorer.review(context, s["right_answer"])
        latencies.append((time.perf_counter() - t0) * 1000)

        if approved_right:
            tp += 1
        else:
            fn += 1  # false negative: rejected correct answer

        # Score hallucinated answer (should be rejected)
        t0 = time.perf_counter()
        approved_halluc, _ = scorer.review(context, s["hallucinated_answer"])
        latencies.append((time.perf_counter() - t0) * 1000)

        if not approved_halluc:
            tn += 1
        else:
            fp += 1  # false positive: approved hallucination

        if (i + 1) % 500 == 0:
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            logger.info("[%d/%d] acc=%.4f", i + 1, len(samples), acc)

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    ba = (tpr + tnr) / 2
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "backend": backend,
        "threshold": threshold,
        "samples": len(samples),
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(ba, 4),
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4),
        "fpr": round(1 - tnr, 4),
        "fnr": round(1 - tpr, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Director-AI on HaluEval")
    p.add_argument(
        "--backend", default="lite", choices=["deberta", "onnx", "lite", "rules"]
    )
    p.add_argument(
        "--split",
        default="qa",
        choices=["qa", "dialogue", "summarization", "general", "all"],
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    samples = load_halueval(args.split, args.max_samples)
    if not samples:
        logger.error("No samples loaded")
        sys.exit(1)

    results = evaluate(samples, args.backend, args.threshold)

    print("\n" + "=" * 50)
    print(f"HaluEval ({args.split}) — Director-AI ({args.backend})")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
