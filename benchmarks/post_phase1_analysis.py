#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Phase-1 post-experiment analysis
"""Post-Phase-1 ensemble analysis — to be run as each new continuous-score
judge from Phase 1 lands.

This script answers, for any set of judge JSON files where ≥2 of the
judges expose a `scores` field:

1. **Per-judge sample-pooled BA at the judge's own optimal threshold.**
2. **Per-judge per-dataset BA with per-dataset optimal thresholds.**
   This is the FactCG-style +2.18 pp upgrade portable to any
   continuous-score judge.
3. **Pairwise score correlation matrix** (Pearson) across the
   continuous-score judges, to confirm whether the new judges add
   diversity or are correlated with FactCG.
4. **5-fold CV logistic regression fusion** on continuous-score
   judges only. The result is then compared to:
   - the single best continuous-score judge
   - the routed champion's binary predictions (the 82.11 % baseline)
5. **3-feature LR fusion** if a binary-only judge is also given (e.g.
   the routed champion). Mixed continuous + binary features test
   whether the binary addition helps or hurts.

Usage::

    python benchmarks/post_phase1_analysis.py \\
        --score-judges \\
            benchmarks/results/factcg_with_scores.json \\
            benchmarks/results/gemma_e4b_q6_logprob.json \\
            benchmarks/results/gemma_e4b_q6_logprob_routed.json \\
        --binary-judges \\
            benchmarks/results/gemma_e4b_q6_routed.json \\
        --output benchmarks/results/post_phase1_analysis_$(date +%Y%m%d_%H%M).json

If `--binary-judges` is omitted, only the score-only analyses run.
The script is **side-effect free** — it does not write any file other
than the explicit `--output` JSON, and it does not modify any judge
JSON in place.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ── Metrics ──────────────────────────────────────────────────────────────


def balanced_accuracy(
    preds: np.ndarray, labels: np.ndarray
) -> float:
    """Standard two-class balanced accuracy.

    Predictions of -1 (unknown) are dropped from the count.
    Returns 0.0 when either class has zero predictions left.
    """
    valid = preds >= 0
    p = preds[valid]
    lab = labels[valid]
    if p.size == 0:
        return 0.0
    pos = (lab == 1).sum()
    neg = (lab == 0).sum()
    if pos == 0 or neg == 0:
        return 0.0
    tp = ((p == 1) & (lab == 1)).sum()
    tn = ((p == 0) & (lab == 0)).sum()
    return float((tp / pos + tn / neg) / 2)


def sweep_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Sweep 0.05..0.95 in 0.05 steps, return (best_t, best_ba)."""
    best_t, best_ba = 0.5, 0.0
    for t in [0.05 * i for i in range(1, 20)]:
        preds = (scores >= t).astype(int)
        ba = balanced_accuracy(preds, labels)
        if ba > best_ba:
            best_t, best_ba = t, ba
    return best_t, best_ba


def per_dataset_threshold_sweep(
    scores: np.ndarray, labels: np.ndarray, datasets: list[str]
) -> dict[str, dict[str, Any]]:
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(datasets):
        by_ds[d].append(i)
    out: dict[str, dict[str, Any]] = {}
    for ds, idxs in by_ds.items():
        idx_arr = np.array(idxs)
        s = scores[idx_arr]
        lab = labels[idx_arr]
        t, ba = sweep_threshold(s, lab)
        out[ds] = {
            "samples": int(len(idxs)),
            "balanced_accuracy": ba,
            "threshold": t,
        }
    return out


# ── Loaders ──────────────────────────────────────────────────────────────


def _arr(x: Any) -> np.ndarray | None:
    """Return ``x`` as a 1-D array, replacing ``None`` with NaN."""
    if x is None:
        return None
    return np.array(
        [float("nan") if v is None else float(v) for v in x],
        dtype=float,
    )


def load_judge(path: Path) -> dict[str, Any]:
    """Load a judge JSON, normalising field names across schemas."""
    with open(path) as f:
        d = json.load(f)
    name = Path(d.get("model", path.stem)).name or path.stem
    labels = np.array(d["labels"], dtype=int)
    datasets = d.get("datasets_per_sample") or d.get("datasets")

    # Continuous-score field — different scripts call it different things.
    scores = _arr(d.get("scores"))

    # Binary predictions — ditto.
    preds_raw = d.get("predictions")
    preds = None if preds_raw is None else np.array(preds_raw, dtype=int)

    return {
        "name": name,
        "path": str(path),
        "labels": labels,
        "datasets": datasets,
        "scores": scores,
        "preds": preds,
    }


# ── LR fusion ────────────────────────────────────────────────────────────


def lr_fusion_5fold(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, float], float]:
    """Run 5-fold CV LR over the full feature matrix.

    Returns predicted labels (concatenated across folds), the average LR
    coefficients across folds, and the intercept (mean across folds).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    n = feature_matrix.shape[0]
    preds = np.zeros(n, dtype=int)
    coef_sums = np.zeros(feature_matrix.shape[1])
    intercept_sum = 0.0
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for _fold, (train_idx, test_idx) in enumerate(
        kf.split(feature_matrix)
    ):
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(feature_matrix[train_idx], labels[train_idx])
        preds[test_idx] = lr.predict(feature_matrix[test_idx])
        coef_sums += lr.coef_[0]
        intercept_sum += lr.intercept_[0]
    avg_coefs = dict(
        zip(feature_names, (coef_sums / 5).tolist(), strict=True)
    )
    return preds, avg_coefs, intercept_sum / 5


def pairwise_correlation(
    score_arrays: list[np.ndarray], names: list[str]
) -> dict[str, dict[str, float]]:
    """Pairwise Pearson correlation between every pair of score arrays."""
    out: dict[str, dict[str, float]] = {}
    for i, ni in enumerate(names):
        out[ni] = {}
        for j, nj in enumerate(names):
            r = float(np.corrcoef(score_arrays[i], score_arrays[j])[0, 1])
            out[ni][nj] = round(r, 4)
    return out


# ── Driver ───────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--score-judges",
        nargs="+",
        type=Path,
        required=True,
        help="Judge JSONs that expose a continuous `scores` field",
    )
    p.add_argument(
        "--binary-judges",
        nargs="*",
        type=Path,
        default=[],
        help=(
            "Optional binary-only judge JSONs (e.g. the routed champion) "
            "to include as 0/1 features in a mixed LR fusion"
        ),
    )
    p.add_argument(
        "--routed-champion",
        type=Path,
        default=None,
        help="Optional routed-champion JSON for the BA-comparison line",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    score_judges = [load_judge(p) for p in args.score_judges]
    binary_judges = [load_judge(p) for p in args.binary_judges]

    # Sanity: every judge must agree on labels and dataset assignment.
    base_labels = score_judges[0]["labels"]
    base_datasets = score_judges[0]["datasets"]
    for j in score_judges[1:] + binary_judges:
        if not np.array_equal(j["labels"], base_labels):
            raise ValueError(
                f"label mismatch between {score_judges[0]['name']} "
                f"and {j['name']}"
            )
        if j["datasets"] != base_datasets:
            raise ValueError(
                f"dataset mismatch between {score_judges[0]['name']} "
                f"and {j['name']}"
            )
    n = len(base_labels)
    logger.info(
        "Loaded %d score judge(s) and %d binary judge(s); n=%d samples",
        len(score_judges),
        len(binary_judges),
        n,
    )

    # ── 1. Per-judge stats ───────────────────────────────────────────────
    individual: dict[str, dict[str, Any]] = {}
    for j in score_judges:
        if j["scores"] is None:
            logger.warning(
                "Judge %s has no scores field; skipping", j["name"]
            )
            continue
        # Drop NaN samples (from None scores) for the BA calculation
        valid_mask = ~np.isnan(j["scores"])
        s_clean = j["scores"][valid_mask]
        l_clean = base_labels[valid_mask]
        d_clean = [
            base_datasets[i] for i in range(n) if valid_mask[i]
        ]
        n_invalid = int((~valid_mask).sum())

        # Sample-pooled global BA at threshold 0.5 and at the optimum
        preds_t05 = (s_clean >= 0.5).astype(int)
        ba_t05 = balanced_accuracy(preds_t05, l_clean)
        best_t, ba_best = sweep_threshold(s_clean, l_clean)

        # Per-dataset thresholds
        per_ds = per_dataset_threshold_sweep(
            s_clean, l_clean, d_clean
        )
        per_ds_avg = float(
            np.mean([m["balanced_accuracy"] for m in per_ds.values()])
            if per_ds
            else 0.0
        )

        individual[j["name"]] = {
            "ba_t05": ba_t05,
            "ba_optimal_global": ba_best,
            "optimal_global_threshold": best_t,
            "per_dataset_avg_balanced_accuracy": per_ds_avg,
            "per_dataset": per_ds,
            "n_invalid": n_invalid,
        }
        logger.info(
            "%-50s  t=0.5: %.4f  opt: %.4f (t=%.2f)  per-ds avg: %.4f",
            j["name"],
            ba_t05,
            ba_best,
            best_t,
            per_ds_avg,
        )

    # ── 2. Pairwise score correlations ───────────────────────────────────
    score_arrays = []
    score_names = []
    for j in score_judges:
        if j["scores"] is None:
            continue
        # Replace NaN with 0.5 (uninformative) for the correlation calc
        s = np.where(np.isnan(j["scores"]), 0.5, j["scores"])
        score_arrays.append(s)
        score_names.append(j["name"])
    correlations: dict[str, dict[str, float]] = {}
    if len(score_arrays) >= 2:
        correlations = pairwise_correlation(score_arrays, score_names)
        logger.info("Pairwise Pearson correlations:")
        for ni in score_names:
            logger.info(
                "  %-50s %s",
                ni,
                "  ".join(
                    f"{nj[:24]}={correlations[ni][nj]:+.3f}"
                    for nj in score_names
                ),
            )

    # ── 3. LR fusion on continuous scores only ───────────────────────────
    fusion_results: dict[str, Any] = {}
    if len(score_arrays) >= 2:
        feature_matrix = np.column_stack(score_arrays)
        # Replace NaN with 0.5 in the feature matrix
        feature_matrix = np.where(
            np.isnan(feature_matrix), 0.5, feature_matrix
        )
        preds_lr, coefs, intercept = lr_fusion_5fold(
            feature_matrix, base_labels, score_names
        )
        ba_lr = balanced_accuracy(preds_lr, base_labels)
        fusion_results["score_only_lr"] = {
            "n_features": len(score_names),
            "feature_names": score_names,
            "lr_coefs": coefs,
            "lr_intercept": intercept,
            "ba": ba_lr,
        }
        logger.info(
            "Score-only LR fusion (%d features): BA=%.4f",
            len(score_names),
            ba_lr,
        )
        logger.info("  LR coefs: %s", coefs)
        logger.info("  intercept: %.3f", intercept)

    # ── 4. Mixed LR fusion (continuous + binary) ────────────────────────
    if score_arrays and binary_judges:
        binary_arrays = []
        binary_names = []
        for j in binary_judges:
            if j["preds"] is None:
                continue
            # Map -1 (unknown) to 0.5
            arr = np.where(
                j["preds"] < 0, 0.5, j["preds"].astype(float)
            )
            binary_arrays.append(arr)
            binary_names.append(j["name"])
        if binary_arrays:
            mixed_features = score_arrays + binary_arrays
            mixed_names = score_names + binary_names
            mixed_matrix = np.column_stack(mixed_features)
            mixed_matrix = np.where(
                np.isnan(mixed_matrix), 0.5, mixed_matrix
            )
            preds_mixed, coefs_mixed, intercept_mixed = lr_fusion_5fold(
                mixed_matrix, base_labels, mixed_names
            )
            ba_mixed = balanced_accuracy(preds_mixed, base_labels)
            fusion_results["mixed_lr"] = {
                "n_features": len(mixed_names),
                "feature_names": mixed_names,
                "lr_coefs": coefs_mixed,
                "lr_intercept": intercept_mixed,
                "ba": ba_mixed,
            }
            logger.info(
                "Mixed (continuous + binary) LR fusion (%d features): BA=%.4f",
                len(mixed_names),
                ba_mixed,
            )
            logger.info("  LR coefs: %s", coefs_mixed)

    # ── 5. Comparison line ───────────────────────────────────────────────
    if args.routed_champion is not None:
        champ = load_judge(args.routed_champion)
        champ_ba = balanced_accuracy(champ["preds"], base_labels)
        logger.info("Routed champion (%s): %.4f", champ["name"], champ_ba)
        fusion_results["routed_champion_reference"] = {
            "name": champ["name"],
            "ba": champ_ba,
        }

    # ── 6. Save ──────────────────────────────────────────────────────────
    out = {
        "n_samples": n,
        "score_judges": [j["name"] for j in score_judges],
        "binary_judges": [j["name"] for j in binary_judges],
        "individual": individual,
        "pairwise_pearson": correlations,
        "fusion": fusion_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("Saved: %s", args.output)


if __name__ == "__main__":
    main()
