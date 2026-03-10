# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Summarization FPR Diagnostic (Layer A)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Diagnostic benchmark: bidirectional NLI baseline sweep for summarization.

Tests the Phase 3 scorer (w_logic=0, direct NLI, trimmed_mean) with
bidirectional scoring at multiple baseline calibration values.

Configurations tested:

- **phase3_fwd_only**: forward-only NLI (current Phase 3, baseline FPR=25.5%)
- **bidir_bl00**: bidirectional NLI, no baseline calibration
- **bidir_bl10**: bidirectional NLI, baseline=0.10
- **bidir_bl15**: bidirectional NLI, baseline=0.15
- **bidir_bl20**: bidirectional NLI, baseline=0.20
- **bidir_bl25**: bidirectional NLI, baseline=0.25

Usage::

    python -m benchmarks.summarization_fpr_diag 50
    python -m benchmarks.summarization_fpr_diag 200 --threshold 0.15
"""

from __future__ import annotations

import json
import logging
import statistics

from benchmarks.halueval_eval import _download_task_data, _extract_pairs

logger = logging.getLogger("DirectorAI.Benchmark.SummFPRDiag")

PROFILES = {
    "phase3_fwd_only": {
        "desc": "forward-only NLI (Phase 3 baseline)",
        "bidir": False,
        "summarization_nli_baseline": 0.0,
    },
    "bidir_bl00": {
        "desc": "bidirectional NLI, no baseline shift",
        "bidir": True,
        "summarization_nli_baseline": 0.0,
    },
    "bidir_bl10": {
        "desc": "bidirectional NLI, baseline=0.10",
        "bidir": True,
        "summarization_nli_baseline": 0.10,
    },
    "bidir_bl15": {
        "desc": "bidirectional NLI, baseline=0.15",
        "bidir": True,
        "summarization_nli_baseline": 0.15,
    },
    "bidir_bl20": {
        "desc": "bidirectional NLI, baseline=0.20",
        "bidir": True,
        "summarization_nli_baseline": 0.20,
    },
    "bidir_bl25": {
        "desc": "bidirectional NLI, baseline=0.25",
        "bidir": True,
        "summarization_nli_baseline": 0.25,
    },
}


def _apply_profile(scorer, profile: dict) -> None:
    """Configure scorer for summarization Phase 3 + bidirectional settings."""
    scorer.W_LOGIC = 0.0
    scorer.W_FACT = 1.0
    scorer._fact_inner_agg = "min"
    scorer._fact_outer_agg = "trimmed_mean"
    scorer._logic_inner_agg = "min"
    scorer._logic_outer_agg = "mean"
    scorer._premise_ratio = 0.85
    scorer._fact_retrieval_top_k = 8

    if profile["bidir"]:
        scorer._use_prompt_as_premise = True
        scorer._summarization_nli_baseline = profile["summarization_nli_baseline"]
    else:
        # Forward-only: use prompt-as-premise but bypass bidir path by
        # temporarily disabling NLI-available check via the old code path.
        # Setting _use_prompt_as_premise=False forces the W_LOGIC<1e-9 branch
        # which calls calculate_factual_divergence_with_evidence directly.
        scorer._use_prompt_as_premise = False
        scorer._summarization_nli_baseline = 0.0


def run_diagnostic(
    max_samples: int | None = None,
    threshold: float = 0.15,
    nli_model: str | None = None,
    profiles: dict | None = None,
) -> dict:
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    if profiles is None:
        profiles = PROFILES

    samples = _download_task_data("summarization")
    correct_pairs: list[tuple[str, str]] = []
    for sample_data in samples:
        for context, response, is_hallucinated in _extract_pairs(
            "summarization", sample_data
        ):
            if not is_hallucinated and context and response:
                correct_pairs.append((context, response))

    if max_samples:
        correct_pairs = correct_pairs[:max_samples]

    all_results: dict[str, dict] = {}

    for profile_name, profile in profiles.items():
        scorer = CoherenceScorer(
            threshold=threshold,
            soft_limit=threshold + 0.05,
            use_nli=True,
            ground_truth_store=VectorGroundTruthStore(),
            nli_model=nli_model,
            scorer_backend="deberta",
            nli_device="cuda",
        )
        _apply_profile(scorer, profile)

        records: list[dict] = []
        for i, (document, summary) in enumerate(correct_pairs):
            if (i + 1) % 25 == 0:
                logger.info(
                    "[%s] Progress: %d / %d", profile_name, i + 1, len(correct_pairs)
                )

            store = VectorGroundTruthStore()
            scorer.ground_truth_store = store
            store.ingest([document])

            h_logic, h_fact, coherence, _ = scorer._heuristic_coherence(
                document, summary
            )
            approved = coherence >= threshold

            records.append(
                {
                    "idx": i,
                    "h_logic": round(h_logic, 4),
                    "h_fact": round(h_fact, 4),
                    "coherence": round(coherence, 4),
                    "approved": approved,
                    "doc_len": len(document),
                    "sum_len": len(summary),
                }
            )

        _print_report(profile_name, profile["desc"], records, threshold)
        all_results[profile_name] = _summarise(records, threshold)

    # Comparison table
    print(f"\n{'=' * 80}")
    print("  PROFILE COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Profile':<30} {'FPR':>8} {'Mean Coh.':>10} {'Mean h_fact':>12}")
    print(f"  {'-' * 62}")
    for name, res in all_results.items():
        print(
            f"  {name:<30} {res['fpr']:>7.1%}"
            f" {res['mean_coherence']:>10.4f} {res['mean_h_fact']:>12.4f}"
        )
    print(f"{'=' * 80}")

    return all_results


def _print_report(name: str, desc: str, records: list[dict], threshold: float) -> None:
    all_fact = [r["h_fact"] for r in records]
    all_coh = [r["coherence"] for r in records]
    fp = [r for r in records if not r["approved"]]
    tp = [r for r in records if r["approved"]]

    print(f"\n{'=' * 80}")
    print(f"  Summarization FPR Diagnostic — {name}")
    print(f"  {desc}")
    print(f"{'=' * 80}")
    print(f"  Samples: {len(records)}  |  Threshold: {threshold}")
    print(
        f"  Approved: {len(tp)}  |  False positives: {len(fp)}"
        f"  |  FPR: {len(fp) / max(len(records), 1):.1%}"
    )
    print()

    def _s(vals):
        return f"{statistics.mean(vals):.4f}" if vals else "N/A"

    print(f"  {'Metric':<30} {'All':>8} {'Approved':>8} {'FP':>8}")
    print(f"  {'-' * 56}")
    print(
        f"  {'h_fact mean':<30} {_s(all_fact):>8}"
        f" {_s([r['h_fact'] for r in tp]):>8}"
        f" {_s([r['h_fact'] for r in fp]):>8}"
    )
    print(
        f"  {'coherence mean':<30} {_s(all_coh):>8}"
        f" {_s([r['coherence'] for r in tp]):>8}"
        f" {_s([r['coherence'] for r in fp]):>8}"
    )
    print()

    print("  FPR at various thresholds:")
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        fp_at_t = sum(1 for r in records if r["coherence"] < t)
        print(
            f"    threshold={t:.2f}:"
            f"  FPR={fp_at_t / len(records) * 100:5.1f}%"
            f"  ({fp_at_t} FP)"
        )
    print(f"{'=' * 80}")


def _summarise(records: list[dict], threshold: float) -> dict:
    all_fact = [r["h_fact"] for r in records]
    all_coh = [r["coherence"] for r in records]
    fp = [r for r in records if not r["approved"]]

    return {
        "total": len(records),
        "fp": len(fp),
        "fpr": len(fp) / max(len(records), 1),
        "mean_h_fact": round(statistics.mean(all_fact), 4),
        "mean_coherence": round(statistics.mean(all_coh), 4),
        "fp_mean_h_fact": round(statistics.mean([r["h_fact"] for r in fp]), 4)
        if fp
        else None,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Summarization FPR diagnostic")
    parser.add_argument("max_samples", nargs="?", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    result = run_diagnostic(
        max_samples=args.max_samples,
        threshold=args.threshold,
        nli_model=args.model,
    )

    outpath = "benchmarks/results/summarization_fpr_diag.json"
    try:
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {outpath}")
    except Exception:
        pass
