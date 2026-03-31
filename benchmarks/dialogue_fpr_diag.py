# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Dialogue FPR Diagnostic

"""Diagnostic benchmark: per-sample h_logic, h_fact, coherence breakdown
for dialogue FPR reduction.

Tests multiple scoring profiles on HaluEval dialogue correct responses
to identify the configuration that minimises false-positive rate.

Configurations tested:

- **Phase 0** (baseline): forward-only NLI, no calibration (max-max)
- **Phase 1** (bidir): bidirectional NLI + baseline calibration (auto-profile)
- **Phase 2** (bidir-85): same as phase 1 with baseline=0.85
- **Phase 3** (bidir-75): same as phase 1 with baseline=0.75

Usage::

    python -m benchmarks.dialogue_fpr_diag 50
    python -m benchmarks.dialogue_fpr_diag 200 --threshold 0.50
"""

from __future__ import annotations

import json
import logging
import statistics

from benchmarks.halueval_eval import _download_task_data, _extract_pairs

logger = logging.getLogger("DirectorAI.Benchmark.DialogueDiag")

# Aggregation configurations to sweep
PROFILES = {
    "phase0_baseline": {
        "desc": "forward-only NLI, no calibration (default)",
        "auto_dialogue_profile": False,
        "dialogue_nli_baseline": 0.80,
        "use_prompt_as_premise": False,
    },
    "phase1_bidir_80": {
        "desc": "bidirectional NLI + baseline=0.80 (auto-profile)",
        "auto_dialogue_profile": True,
        "dialogue_nli_baseline": 0.80,
        "use_prompt_as_premise": False,
    },
    "phase2_bidir_85": {
        "desc": "bidirectional NLI + baseline=0.85",
        "auto_dialogue_profile": True,
        "dialogue_nli_baseline": 0.85,
        "use_prompt_as_premise": False,
    },
    "phase3_bidir_75": {
        "desc": "bidirectional NLI + baseline=0.75",
        "auto_dialogue_profile": True,
        "dialogue_nli_baseline": 0.75,
        "use_prompt_as_premise": False,
    },
}


def _apply_profile(scorer, profile: dict) -> None:
    """Apply scoring profile settings to a scorer."""
    scorer._auto_dialogue_profile = profile["auto_dialogue_profile"]
    scorer._dialogue_nli_baseline = profile["dialogue_nli_baseline"]
    scorer._use_prompt_as_premise = profile["use_prompt_as_premise"]
    # Reset aggregation to defaults (the dialogue path doesn't use them)
    scorer._fact_inner_agg = "max"
    scorer._fact_outer_agg = "max"
    scorer._logic_inner_agg = "max"
    scorer._logic_outer_agg = "max"


def run_diagnostic(
    max_samples: int | None = None,
    threshold: float = 0.50,
    nli_model: str | None = None,
    profiles: dict | None = None,
) -> dict:
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    if profiles is None:
        profiles = PROFILES

    # Load dialogue correct responses
    samples = _download_task_data("dialogue")
    correct_pairs: list[tuple[str, str]] = []
    for sample_data in samples:
        for context, response, is_hallucinated in _extract_pairs(
            "dialogue",
            sample_data,
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
        )
        _apply_profile(scorer, profile)

        records: list[dict] = []
        for i, (context, response) in enumerate(correct_pairs):
            if (i + 1) % 25 == 0:
                logger.info(
                    "[%s] Progress: %d / %d",
                    profile_name,
                    i + 1,
                    len(correct_pairs),
                )

            store = VectorGroundTruthStore()
            scorer.ground_truth_store = store
            store.ingest([context])

            h_logic, h_fact, coherence, _ = scorer._heuristic_coherence(
                context,
                response,
            )
            approved = coherence >= threshold

            records.append(
                {
                    "idx": i,
                    "h_logic": round(h_logic, 4),
                    "h_fact": round(h_fact, 4),
                    "coherence": round(coherence, 4),
                    "approved": approved,
                    "ctx_len": len(context),
                    "resp_len": len(response),
                },
            )

        _print_report(profile_name, profile["desc"], records, threshold)
        all_results[profile_name] = _summarise(records, threshold)

    # Comparison table
    print(f"\n{'=' * 75}")
    print("  PROFILE COMPARISON")
    print(f"{'=' * 75}")
    print(f"  {'Profile':<30} {'FPR':>8} {'Mean Coh.':>10} {'Mean h_fact':>12}")
    print(f"  {'-' * 62}")
    for name, res in all_results.items():
        print(
            f"  {name:<30} {res['fpr']:>7.1%} {res['mean_coherence']:>10.4f} {res['mean_h_fact']:>12.4f}",
        )
    print(f"{'=' * 75}")

    return all_results


def _print_report(name: str, desc: str, records: list[dict], threshold: float) -> None:
    all_logic = [r["h_logic"] for r in records]
    all_fact = [r["h_fact"] for r in records]
    all_coh = [r["coherence"] for r in records]
    fp = [r for r in records if not r["approved"]]
    tp = [r for r in records if r["approved"]]

    print(f"\n{'=' * 75}")
    print(f"  Dialogue FPR Diagnostic — {name}")
    print(f"  {desc}")
    print(f"{'=' * 75}")
    print(f"  Samples: {len(records)}  |  Threshold: {threshold}")
    print(
        f"  Approved: {len(tp)}  |  False positives: {len(fp)}"
        f"  |  FPR: {len(fp) / max(len(records), 1):.1%}",
    )
    print()

    def _s(vals):
        return f"{statistics.mean(vals):.4f}" if vals else "N/A"

    print(f"  {'Metric':<30} {'All':>8} {'Approved':>8} {'FP':>8}")
    print(f"  {'-' * 56}")
    print(
        f"  {'h_logic mean':<30} {_s(all_logic):>8}"
        f" {_s([r['h_logic'] for r in tp]):>8}"
        f" {_s([r['h_logic'] for r in fp]):>8}",
    )
    print(
        f"  {'h_fact mean':<30} {_s(all_fact):>8}"
        f" {_s([r['h_fact'] for r in tp]):>8}"
        f" {_s([r['h_fact'] for r in fp]):>8}",
    )
    print(
        f"  {'coherence mean':<30} {_s(all_coh):>8}"
        f" {_s([r['coherence'] for r in tp]):>8}"
        f" {_s([r['coherence'] for r in fp]):>8}",
    )
    print()

    print("  FPR at various thresholds:")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        fp_at_t = sum(1 for r in records if r["coherence"] < t)
        print(
            f"    threshold={t:.2f}:"
            f"  FPR={fp_at_t / len(records) * 100:5.1f}%"
            f"  ({fp_at_t} FP)",
        )
    print(f"{'=' * 75}")


def _summarise(records: list[dict], threshold: float) -> dict:
    all_logic = [r["h_logic"] for r in records]
    all_fact = [r["h_fact"] for r in records]
    all_coh = [r["coherence"] for r in records]
    fp = [r for r in records if not r["approved"]]

    return {
        "total": len(records),
        "fp": len(fp),
        "fpr": len(fp) / max(len(records), 1),
        "mean_h_logic": round(statistics.mean(all_logic), 4),
        "mean_h_fact": round(statistics.mean(all_fact), 4),
        "mean_coherence": round(statistics.mean(all_coh), 4),
        "fp_mean_h_fact": round(statistics.mean([r["h_fact"] for r in fp]), 4)
        if fp
        else None,
        "records": records,
    }


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Dialogue FPR diagnostic")
    parser.add_argument("max_samples", nargs="?", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    result = run_diagnostic(
        max_samples=args.max_samples,
        threshold=args.threshold,
        nli_model=args.model,
    )

    outpath = "benchmarks/results/dialogue_fpr_diag.json"
    try:
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nResults saved to {outpath}")
    except Exception:
        pass
