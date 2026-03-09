# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Summarization FPR Diagnostic
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Diagnostic benchmark: per-sample h_logic, h_fact, coherence breakdown.

Runs the summ-profile configuration and dumps the score distribution
to identify what drives remaining false positives.

Usage::

    python -m benchmarks.summarization_fpr_diag 50
    python -m benchmarks.summarization_fpr_diag 200 --threshold 0.35
"""

from __future__ import annotations

import json
import logging
import statistics

from benchmarks.halueval_eval import _download_task_data, _extract_pairs

logger = logging.getLogger("DirectorAI.Benchmark.SummarizationDiag")


def run_diagnostic(
    max_samples: int | None = None,
    threshold: float = 0.20,
    nli_model: str | None = None,
) -> dict:
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    scorer = CoherenceScorer(
        threshold=threshold,
        soft_limit=threshold + 0.05,
        use_nli=True,
        ground_truth_store=VectorGroundTruthStore(),
        nli_model=nli_model,
        scorer_backend="deberta",
        w_logic=0.0,
        w_fact=1.0,
    )
    # Apply summ-profile Phase 3 settings
    scorer._fact_inner_agg = "min"
    scorer._fact_outer_agg = "trimmed_mean"
    scorer._logic_inner_agg = "min"
    scorer._logic_outer_agg = "mean"
    scorer._premise_ratio = 0.85
    scorer._fact_retrieval_top_k = 8
    scorer._use_prompt_as_premise = True

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

    records: list[dict] = []
    for i, (document, summary) in enumerate(correct_pairs):
        if (i + 1) % 25 == 0:
            logger.info("Progress: %d / %d", i + 1, len(correct_pairs))

        store = VectorGroundTruthStore()
        scorer.ground_truth_store = store
        store.ingest([document])

        h_logic, h_fact, coherence, _ = scorer._heuristic_coherence(document, summary)
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

    # Analyze
    all_logic = [r["h_logic"] for r in records]
    all_fact = [r["h_fact"] for r in records]
    all_coh = [r["coherence"] for r in records]
    fp = [r for r in records if not r["approved"]]
    tp = [r for r in records if r["approved"]]

    print(f"\n{'=' * 75}")
    print("  Summarization FPR Diagnostic (summ-profile)")
    print(f"{'=' * 75}")
    print(f"  Samples: {len(records)}  |  Threshold: {threshold}")
    print(
        f"  Approved: {len(tp)}  |  False positives: {len(fp)}  |  FPR: {len(fp) / max(len(records), 1):.1%}"
    )
    print()

    print(f"  {'Metric':<30} {'All':>8} {'Approved':>8} {'FP':>8}")
    print(f"  {'-' * 56}")

    def _stats(vals):
        if not vals:
            return "N/A"
        return f"{statistics.mean(vals):.4f}"

    print(
        f"  {'h_logic mean':<30} {_stats(all_logic):>8} {_stats([r['h_logic'] for r in tp]):>8} {_stats([r['h_logic'] for r in fp]):>8}"
    )
    print(
        f"  {'h_fact mean':<30} {_stats(all_fact):>8} {_stats([r['h_fact'] for r in tp]):>8} {_stats([r['h_fact'] for r in fp]):>8}"
    )
    print(
        f"  {'coherence mean':<30} {_stats(all_coh):>8} {_stats([r['coherence'] for r in tp]):>8} {_stats([r['coherence'] for r in fp]):>8}"
    )
    print()

    # Distribution buckets for h_logic
    print("  h_logic distribution:")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        n = sum(1 for v in all_logic if lo <= v < hi)
        print(f"    [{lo:.1f}, {hi:.1f}): {n:>4} ({n / len(all_logic) * 100:5.1f}%)")

    print("  h_fact distribution:")
    for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        n = sum(1 for v in all_fact if lo <= v < hi)
        print(f"    [{lo:.1f}, {hi:.1f}): {n:>4} ({n / len(all_fact) * 100:5.1f}%)")

    print("  coherence distribution:")
    for lo, hi in [(0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        n = sum(1 for v in all_coh if lo <= v < hi)
        label = f"[{lo:.2f}, {hi:.2f})"
        marker = " <-- FP zone" if hi <= threshold + 0.01 else ""
        print(f"    {label:<15} {n:>4} ({n / len(all_coh) * 100:5.1f}%){marker}")

    # FPR at different thresholds
    print()
    print("  FPR at various thresholds:")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        fp_at_t = sum(1 for r in records if r["coherence"] < t)
        print(
            f"    threshold={t:.2f}:  FPR={fp_at_t / len(records) * 100:5.1f}%  ({fp_at_t} FP)"
        )

    print(f"{'=' * 75}")

    return {
        "total": len(records),
        "fp": len(fp),
        "fpr": round(len(fp) / max(len(records), 1), 4),
        "mean_h_logic": round(statistics.mean(all_logic), 4),
        "mean_h_fact": round(statistics.mean(all_fact), 4),
        "mean_coherence": round(statistics.mean(all_coh), 4),
        "fp_mean_h_logic": round(statistics.mean([r["h_logic"] for r in fp]), 4)
        if fp
        else None,
        "fp_mean_h_fact": round(statistics.mean([r["h_fact"] for r in fp]), 4)
        if fp
        else None,
        "records": records,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Summarization FPR diagnostic")
    parser.add_argument("max_samples", nargs="?", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    result = run_diagnostic(
        max_samples=args.max_samples,
        threshold=args.threshold,
        nli_model=args.model,
    )

    outpath = "/root/director-ai/benchmarks/results/summarization_fpr_diag.json"
    try:
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {outpath}")
    except Exception:
        pass
