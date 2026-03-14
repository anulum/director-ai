# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Layer C Claim Coverage FPR Diagnostic
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Benchmark Layer C (claim decomposition + coverage) FPR on HaluEval.

Sweeps alpha values (0.0 = Layer A only, 1.0 = Layer C only) and
support_threshold values to find optimal blending.

Usage:
    python -m benchmarks.claim_coverage_fpr_diag [--samples N] [--out FILE]

Requires: pip install director-ai[nli] datasets
GPU: set nli_device="cuda" for production speed.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np


def run_benchmark(
    n_samples: int = 200,
    threshold: float = 0.15,
    nli_model: str | None = None,
    output_path: str = "benchmarks/results/claim_coverage_fpr_diag.json",
):
    from datasets import load_dataset  # type: ignore[import-untyped]

    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    ds = load_dataset("pminervini/HaluEval", "summarization", split="data")
    samples = list(ds.select(range(min(n_samples, len(ds)))))

    scorer = CoherenceScorer(
        threshold=threshold,
        soft_limit=threshold + 0.05,
        use_nli=True,
        ground_truth_store=VectorGroundTruthStore(),
        nli_model=nli_model,
        scorer_backend="deberta",
        nli_device="cuda",
    )
    scorer._use_prompt_as_premise = True
    scorer._fact_inner_agg = "min"
    scorer._fact_outer_agg = "trimmed_mean"
    scorer._premise_ratio = 0.85
    scorer.W_LOGIC = 0.0
    scorer.W_FACT = 1.0

    profiles = [
        # (name, claim_enabled, alpha, support_threshold, baseline)
        ("layer_a_only", False, 0.0, 0.5, 0.20),
        ("alpha_20_st50", True, 0.2, 0.5, 0.20),
        ("alpha_30_st50", True, 0.3, 0.5, 0.20),
        ("alpha_40_st50", True, 0.4, 0.5, 0.20),
        ("alpha_50_st50", True, 0.5, 0.5, 0.20),
        ("alpha_40_st40", True, 0.4, 0.4, 0.20),
        ("alpha_40_st60", True, 0.4, 0.6, 0.20),
        ("alpha_60_st50", True, 0.6, 0.5, 0.20),
    ]

    results = {}

    for name, claim_en, alpha, st, baseline in profiles:
        scorer._summarization_nli_baseline = baseline
        scorer._claim_coverage_enabled = claim_en
        scorer._claim_coverage_alpha = alpha
        scorer._claim_support_threshold = st

        fp_count = 0
        coherences = []
        h_facts = []
        coverages = []
        t0 = time.monotonic()

        for i, sample in enumerate(samples):
            document = sample["document"]
            right_summary = sample["right_summary"]

            approved, score = scorer.review(document, right_summary)
            coherences.append(score.score)
            h_facts.append(score.h_factual)

            if score.evidence and score.evidence.claim_coverage is not None:
                coverages.append(score.evidence.claim_coverage)

            if not approved:
                fp_count += 1

            if (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                print(
                    f"  [{name}] {i + 1}/{n_samples} "
                    f"FP={fp_count} elapsed={elapsed:.1f}s",
                )

        elapsed = time.monotonic() - t0
        fpr = fp_count / n_samples * 100

        result = {
            "name": name,
            "claim_enabled": claim_en,
            "alpha": alpha,
            "support_threshold": st,
            "baseline": baseline,
            "fpr_pct": fpr,
            "fp_count": fp_count,
            "n_samples": n_samples,
            "mean_coherence": float(np.mean(coherences)),
            "mean_h_fact": float(np.mean(h_facts)),
            "elapsed_seconds": round(elapsed, 1),
        }
        if coverages:
            result["mean_coverage"] = float(np.mean(coverages))

        results[name] = result
        print(
            f"[{name}] FPR={fpr:.1f}% "
            f"coh={np.mean(coherences):.4f} "
            f"h_fact={np.mean(h_facts):.4f} "
            f"time={elapsed:.1f}s",
        )
        if coverages:
            print(f"  mean_coverage={np.mean(coverages):.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Layer C claim coverage FPR diagnostic",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--out",
        default="benchmarks/results/claim_coverage_fpr_diag.json",
    )
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    run_benchmark(
        n_samples=args.samples,
        nli_model=args.model,
        output_path=args.out,
    )


if __name__ == "__main__":
    main()
