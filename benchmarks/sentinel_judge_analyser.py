# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Sentinel-Judge Ensemble Analyser
"""Combine per-sample predictions from multiple judges and report ensemble metrics.

Loads JSON files produced by ``benchmarks/gemma_aggrefact_eval.py`` and
``benchmarks/aggrefact_eval.py --save-scores`` (the v2 ensemble schema).
Computes:

1. Per-judge global and per-dataset balanced accuracy
2. Voting ensemble (majority of available judges)
3. Per-dataset routing ensemble (use the historically best judge per dataset,
   measured on a held-out half of the samples)
4. Logistic regression fusion on continuous scores + dataset one-hot
   (5-fold cross-validated)
5. Oracle upper bound (pick the judge that got each sample right, if any)

Usage::

    python benchmarks/sentinel_judge_analyser.py \\
        --judges \\
            benchmarks/results/gemma_e4b_q6_with_preds.json \\
            benchmarks/results/factcg_with_scores.json \\
        --output benchmarks/results/sentinel_judge_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Loading ──────────────────────────────────────────────────────────────


def load_judge(path: str) -> dict:
    """Load a judge JSON and normalise the schema."""
    data = json.loads(Path(path).read_text())
    name = data.get("model", path)
    preds = data.get("predictions", [])
    labels = data.get("labels", [])
    datasets = data.get("datasets_per_sample", [])
    scores = data.get("scores")
    if scores is None or (scores and isinstance(scores[0], dict)):
        # legacy v1 layout — derive scores list[float] from list[dict]
        scores = [float(s.get("score", -1)) for s in (data.get("scores") or [])]
    if not (len(preds) == len(labels) == len(datasets)):
        msg = (
            f"{path}: inconsistent list lengths "
            f"preds={len(preds)} labels={len(labels)} datasets={len(datasets)}"
        )
        raise ValueError(msg)
    return {
        "name": Path(path).stem,
        "model": name,
        "preds": [int(p) for p in preds],
        "scores": [float(s) for s in scores] if scores else None,
        "labels": [int(l) for l in labels],
        "datasets": list(datasets),
    }


def align_judges(judges: list[dict]) -> tuple[list[int], list[str], list[list[int]], list[list[float]]]:
    """Verify all judges share the same labels/datasets ordering. Return aligned tensors."""
    base_labels = judges[0]["labels"]
    base_datasets = judges[0]["datasets"]
    for j in judges[1:]:
        if j["labels"] != base_labels:
            msg = f"label mismatch between {judges[0]['name']} and {j['name']}"
            raise ValueError(msg)
        if j["datasets"] != base_datasets:
            msg = f"dataset mismatch between {judges[0]['name']} and {j['name']}"
            raise ValueError(msg)
    preds_matrix = [j["preds"] for j in judges]
    scores_matrix = [j["scores"] if j["scores"] else [-1] * len(base_labels) for j in judges]
    return base_labels, base_datasets, preds_matrix, scores_matrix


# ── Metrics ──────────────────────────────────────────────────────────────


def balanced_accuracy(preds: list[int], labels: list[int]) -> float:
    pos = neg = tp = tn = 0
    for p, l in zip(preds, labels, strict=True):
        if p < 0:
            continue
        if l == 1:
            pos += 1
            if p == 1:
                tp += 1
        else:
            neg += 1
            if p == 0:
                tn += 1
    if pos == 0 or neg == 0:
        return 0.0
    return (tp / pos + tn / neg) / 2


def per_dataset_ba(preds: list[int], labels: list[int], datasets: list[str]) -> dict:
    out: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for p, l, d in zip(preds, labels, datasets, strict=True):
        out[d][0].append(p)
        out[d][1].append(l)
    return {
        d: {"samples": len(l_), "balanced_accuracy": balanced_accuracy(p_, l_)}
        for d, (p_, l_) in out.items()
    }


# ── Ensemble strategies ──────────────────────────────────────────────────


def voting_ensemble(preds_matrix: list[list[int]]) -> list[int]:
    """Majority vote. -1 if all judges abstain. Tie → most-confident judge (#0)."""
    n_samples = len(preds_matrix[0])
    out = []
    for i in range(n_samples):
        votes = [m[i] for m in preds_matrix if m[i] >= 0]
        if not votes:
            out.append(-1)
            continue
        ones = sum(1 for v in votes if v == 1)
        zeros = sum(1 for v in votes if v == 0)
        if ones > zeros:
            out.append(1)
        elif zeros > ones:
            out.append(0)
        else:
            out.append(votes[0])  # break tie with judge #0
    return out


def routed_ensemble(
    preds_matrix: list[list[int]],
    labels: list[int],
    datasets: list[str],
    judge_names: list[str],
) -> tuple[list[int], dict[str, str]]:
    """Per-dataset router: pick the judge with the highest BA on each dataset.

    Uses a 50/50 split — first half of each dataset for selection, second
    half for evaluation. This is the canonical "router learned on val" pattern;
    on AggreFact 29 K halves are large enough to be statistically meaningful.
    """
    rng = np.random.default_rng(0)
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(datasets):
        by_dataset[d].append(i)

    train_idx: set[int] = set()
    for d, idxs in by_dataset.items():
        rng.shuffle(idxs)
        cut = len(idxs) // 2
        train_idx.update(idxs[:cut])

    # Per-dataset best judge on train half
    routing: dict[str, str] = {}
    for d, idxs in by_dataset.items():
        train = [i for i in idxs if i in train_idx]
        if not train:
            routing[d] = judge_names[0]
            continue
        best_ba = -1.0
        best_judge = judge_names[0]
        for j_idx, j_name in enumerate(judge_names):
            j_preds = [preds_matrix[j_idx][i] for i in train]
            j_labels = [labels[i] for i in train]
            ba = balanced_accuracy(j_preds, j_labels)
            if ba > best_ba:
                best_ba = ba
                best_judge = j_name
        routing[d] = best_judge

    # Apply routing to all samples (eval on the held-out half is what matters)
    name_to_idx = {n: i for i, n in enumerate(judge_names)}
    out = []
    for i, d in enumerate(datasets):
        j_idx = name_to_idx[routing[d]]
        out.append(preds_matrix[j_idx][i])
    return out, routing


def lr_fusion_ensemble(
    scores_matrix: list[list[float]],
    labels: list[int],
    datasets: list[str],
) -> list[int]:
    """Logistic regression on [score_j1, score_j2, ..., dataset_onehot].

    5-fold cross-validated to avoid train/test leakage. Returns the
    out-of-fold predictions (each sample is predicted by a model that
    did not see it during training).
    """
    unique_ds = sorted(set(datasets))
    ds_idx = {d: i for i, d in enumerate(unique_ds)}
    n_samples = len(labels)
    n_judges = len(scores_matrix)
    n_features = n_judges + len(unique_ds)

    x = np.zeros((n_samples, n_features), dtype=np.float64)
    for j_idx in range(n_judges):
        x[:, j_idx] = np.array(scores_matrix[j_idx], dtype=np.float64)
    for i, d in enumerate(datasets):
        x[i, n_judges + ds_idx[d]] = 1.0
    y = np.array(labels, dtype=np.int64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    out_pred = np.full(n_samples, -1, dtype=np.int64)
    for fold, (train, test) in enumerate(skf.split(x, y)):
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(x[train], y[train])
        out_pred[test] = clf.predict(x[test])
        logger.info(
            "  fold %d: train=%d test=%d fit_score=%.4f",
            fold + 1, len(train), len(test), clf.score(x[train], y[train]),
        )
    return out_pred.tolist()


def oracle_upper_bound(preds_matrix: list[list[int]], labels: list[int]) -> list[int]:
    """For each sample, pick any judge that got it right; else 0."""
    n = len(labels)
    out = []
    for i in range(n):
        target = labels[i]
        if any(m[i] == target for m in preds_matrix):
            out.append(target)
        else:
            out.append(1 - target)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judges", nargs="+", required=True,
                   help="Paths to judge JSON files (v2 ensemble schema)")
    p.add_argument("--output", type=str,
                   default="benchmarks/results/sentinel_judge_report.json")
    args = p.parse_args()

    judges = [load_judge(p_) for p_ in args.judges]
    judge_names = [j["name"] for j in judges]
    logger.info("Loaded %d judges: %s", len(judges), judge_names)

    labels, datasets, preds_matrix, scores_matrix = align_judges(judges)
    logger.info("Aligned %d samples across %d datasets", len(labels), len(set(datasets)))

    # Per-judge metrics
    individual = {}
    for j_idx, j in enumerate(judges):
        ba = balanced_accuracy(j["preds"], labels)
        per_ds = per_dataset_ba(j["preds"], labels, datasets)
        individual[j["name"]] = {
            "global_balanced_accuracy": ba,
            "per_dataset": per_ds,
        }
        logger.info("  %s: BA=%.4f", j["name"], ba)

    # Ensembles
    logger.info("Voting ensemble...")
    vote_preds = voting_ensemble(preds_matrix)
    vote_ba = balanced_accuracy(vote_preds, labels)
    logger.info("  voting BA: %.4f", vote_ba)

    logger.info("Routed ensemble...")
    routed_preds, routing = routed_ensemble(
        preds_matrix, labels, datasets, judge_names
    )
    routed_ba = balanced_accuracy(routed_preds, labels)
    logger.info("  routed BA: %.4f", routed_ba)
    logger.info("  routing: %s", routing)

    logger.info("LR fusion ensemble (5-fold CV)...")
    if all(j["scores"] is not None for j in judges):
        lr_preds = lr_fusion_ensemble(scores_matrix, labels, datasets)
        lr_ba = balanced_accuracy(lr_preds, labels)
        lr_per_ds = per_dataset_ba(lr_preds, labels, datasets)
        logger.info("  LR fusion BA: %.4f", lr_ba)
    else:
        lr_preds, lr_ba, lr_per_ds = None, None, None
        logger.info("  LR fusion skipped (some judges have no scores)")

    logger.info("Oracle upper bound...")
    oracle_preds = oracle_upper_bound(preds_matrix, labels)
    oracle_ba = balanced_accuracy(oracle_preds, labels)
    logger.info("  oracle BA: %.4f", oracle_ba)

    report = {
        "judges": judge_names,
        "samples": len(labels),
        "individual": individual,
        "voting": {
            "global_balanced_accuracy": vote_ba,
            "per_dataset": per_dataset_ba(vote_preds, labels, datasets),
        },
        "routed": {
            "global_balanced_accuracy": routed_ba,
            "routing_table": routing,
            "per_dataset": per_dataset_ba(routed_preds, labels, datasets),
        },
        "lr_fusion": (
            {
                "global_balanced_accuracy": lr_ba,
                "per_dataset": lr_per_ds,
                "method": "5-fold stratified CV, score+dataset_onehot features",
            }
            if lr_ba is not None
            else None
        ),
        "oracle_upper_bound": {
            "global_balanced_accuracy": oracle_ba,
            "per_dataset": per_dataset_ba(oracle_preds, labels, datasets),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2))

    print()
    print("=" * 60)
    print(f"  SENTINEL-JUDGE ENSEMBLE REPORT — {len(labels)} samples")
    print("=" * 60)
    for n in judge_names:
        print(f"  {n:50s} {individual[n]['global_balanced_accuracy']:.4f}")
    print("-" * 60)
    print(f"  voting (majority)                               {vote_ba:.4f}")
    print(f"  routed (per-dataset best, 50/50 split)          {routed_ba:.4f}")
    if lr_ba is not None:
        print(f"  LR fusion (5-fold CV)                           {lr_ba:.4f}")
    print(f"  oracle upper bound                              {oracle_ba:.4f}")
    print("=" * 60)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
