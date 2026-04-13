# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — FreshQA benchmark evaluation
"""Evaluate Director-AI on the FreshQA benchmark.

FreshQA (Vu et al., 2023) tests factual accuracy on time-sensitive
questions. We use it to measure Director-AI's ability to detect
outdated or fabricated answers against ground-truth facts.

Usage::

    python benchmarks/eval_freshqa.py --backend deberta --max-samples 500
    python benchmarks/eval_freshqa.py --backend lite
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


def load_freshqa(max_samples: int | None = None) -> list[dict]:
    """Load FreshQA dataset.

    FreshQA is distributed as a Google Sheet / CSV. We use the
    HuggingFace mirror when available, or a local CSV.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("freshllms/freshqa", split="train")
        samples = []
        for item in ds:
            question = item.get("question", "")
            answer = item.get("answer", item.get("best_answer", ""))
            if question and answer:
                samples.append(
                    {
                        "question": question,
                        "answer": answer,
                        "category": item.get("fact_type", "unknown"),
                    }
                )
                if max_samples and len(samples) >= max_samples:
                    break
        logger.info("Loaded %d FreshQA samples", len(samples))
        return samples
    except Exception as exc:
        logger.error("Failed to load FreshQA: %s", exc)
        return []


def evaluate(samples: list[dict], backend: str, threshold: float) -> dict:
    """Run Director-AI on FreshQA.

    For each question+answer pair:
    - Score (question, correct_answer) → should approve
    - Score (question, fabricated_answer) → should reject

    We generate fabricated answers by negating the correct one.
    """
    from director_ai.core.scoring.scorer import CoherenceScorer

    scorer = CoherenceScorer(
        threshold=threshold,
        use_nli=(backend != "lite"),
        scorer_backend=backend,
    )

    correct_approved = 0
    fabricated_rejected = 0
    total = 0
    latencies: list[float] = []

    for i, s in enumerate(samples):
        question = s["question"]
        correct = s["answer"]
        # Simple fabrication: prepend "Not true: "
        fabricated = (
            f"Actually, that is incorrect. The answer is definitely not {correct}"
        )

        # Score correct answer
        t0 = time.perf_counter()
        approved, _ = scorer.review(question, correct)
        latencies.append((time.perf_counter() - t0) * 1000)

        if approved:
            correct_approved += 1

        # Score fabricated answer
        t0 = time.perf_counter()
        fab_approved, _ = scorer.review(question, fabricated)
        latencies.append((time.perf_counter() - t0) * 1000)

        if not fab_approved:
            fabricated_rejected += 1

        total += 1

        if (i + 1) % 200 == 0:
            logger.info(
                "[%d/%d] correct_approved=%.1f%% fabricated_rejected=%.1f%%",
                i + 1,
                len(samples),
                correct_approved / total * 100,
                fabricated_rejected / total * 100,
            )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "backend": backend,
        "threshold": threshold,
        "samples": total,
        "correct_approval_rate": round(correct_approved / total * 100, 2)
        if total
        else 0,
        "fabrication_rejection_rate": round(fabricated_rejected / total * 100, 2)
        if total
        else 0,
        "avg_latency_ms": round(avg_latency, 2),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Director-AI on FreshQA")
    p.add_argument(
        "--backend", default="lite", choices=["deberta", "onnx", "lite", "rules"]
    )
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    samples = load_freshqa(args.max_samples)
    if not samples:
        logger.error("No samples loaded")
        sys.exit(1)

    results = evaluate(samples, args.backend, args.threshold)

    print("\n" + "=" * 50)
    print(f"FreshQA — Director-AI ({args.backend})")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
