# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Train dataset-type classifier for per-dataset NLI threshold adaptation.

Loads LLM-AggreFact text + cached NLI scores, trains a LogisticRegression
to predict which of the 11 datasets an input resembles, then maps each
dataset to its optimal NLI threshold.

Usage::

    export HF_TOKEN=hf_...
    python tools/train_dataset_classifier.py \
        --scores gpu_results/2026-03-15/cached_scores_l40s.json \
        --output src/director_ai/data/dataset_classifier.pkl

Requires: scikit-learn, datasets (HuggingFace).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from director_ai.core.scoring.meta_classifier import (  # noqa: E402
    TEXT_FEATURE_COLS,
    extract_text_features,
)


def load_aggrefact() -> list[dict]:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HF_TOKEN env var (gated dataset)")
    logger.info("Loading LLM-AggreFact...")
    ds = load_dataset("lytang/LLM-AggreFact", split="test", token=token)
    rows = list(ds)
    logger.info("Loaded %d samples", len(rows))
    return rows


def load_cached_scores(path: str) -> dict[int, dict]:
    """Load cached scores, keyed by index for alignment with AggreFact rows."""
    data = json.loads(Path(path).read_text())
    by_idx = {}
    for i, entry in enumerate(data["scores"]):
        by_idx[i] = entry
    logger.info("Loaded %d cached scores", len(by_idx))
    return by_idx


def _ba_from_preds(preds: list[int], labels: list[int]) -> float:
    tp = sum(1 for p, lab in zip(preds, labels, strict=True) if p == 1 and lab == 1)
    tn = sum(1 for p, lab in zip(preds, labels, strict=True) if p == 0 and lab == 0)
    pos = sum(1 for lab in labels if lab == 1)
    neg = sum(1 for lab in labels if lab == 0)
    tpr = tp / pos if pos > 0 else 0
    tnr = tn / neg if neg > 0 else 0
    return (tpr + tnr) / 2


def sweep_threshold(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """Find threshold maximizing balanced accuracy. Returns (threshold, ba)."""
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        preds = [1 if s >= t else 0 for s in scores]
        ba = _ba_from_preds(preds, labels)
        if ba > best_ba:
            best_ba = ba
            best_t = round(float(t), 2)
    return best_t, best_ba


def main():
    parser = argparse.ArgumentParser(description="Train dataset-type classifier")
    parser.add_argument(
        "--scores",
        required=True,
        help="Path to cached_scores JSON from aggrefact_eval.py --save-scores",
    )
    parser.add_argument(
        "--output",
        default="src/director_ai/data/dataset_classifier.pkl",
        help="Output pickle path",
    )
    parser.add_argument(
        "--confidence-gate",
        type=float,
        default=0.4,
        help="Min confidence to apply per-dataset threshold (default 0.4)",
    )
    args = parser.parse_args()

    # Load data
    rows = load_aggrefact()
    cached = load_cached_scores(args.scores)

    if len(rows) != len(cached):
        logger.warning(
            "Row count mismatch: AggreFact=%d, cached=%d. Using min.",
            len(rows),
            len(cached),
        )

    n = min(len(rows), len(cached))

    # Extract features + labels
    logger.info("Extracting text features...")
    features = []
    dataset_labels = []
    all_datasets = sorted(set(cached[i]["dataset"] for i in range(n)))
    ds_to_idx = {ds: i for i, ds in enumerate(all_datasets)}

    for i in range(n):
        row = rows[i]
        score_entry = cached[i]
        doc = row.get("doc", "")
        claim = row.get("claim", "")
        if not doc or not claim:
            continue
        feat = extract_text_features(doc, claim)
        features.append([feat[c] for c in TEXT_FEATURE_COLS])
        dataset_labels.append(ds_to_idx[score_entry["dataset"]])

    x_mat = np.array(features)
    y = np.array(dataset_labels)
    logger.info("Feature matrix: %s, %d classes", x_mat.shape, len(all_datasets))

    # Train with stratified 5-fold cross-validation
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_mat)

    clf = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
    )

    cv_scores = cross_val_score(clf, x_scaled, y, cv=5, scoring="balanced_accuracy")
    logger.info(
        "5-fold CV balanced accuracy: %.1f%% ± %.1f%%",
        cv_scores.mean() * 100,
        cv_scores.std() * 100,
    )

    # Train final model on all data
    clf.fit(x_scaled, y)
    train_acc = clf.score(x_scaled, y)
    logger.info("Train accuracy: %.1f%%", train_acc * 100)

    # Per-dataset optimal thresholds from cached NLI scores
    logger.info("Sweeping per-dataset thresholds...")
    by_dataset: dict[str, list[tuple[int, float]]] = {}
    for i in range(n):
        entry = cached[i]
        ds = entry["dataset"]
        if ds not in by_dataset:
            by_dataset[ds] = []
        # score is entailment probability; label 1=supported, 0=hallucinated
        by_dataset[ds].append((entry["label"], entry["score"]))

    dataset_thresholds = {}
    global_labels = []
    global_scores = []
    for ds_name in sorted(by_dataset):
        labels = [pair[0] for pair in by_dataset[ds_name]]
        scores = [pair[1] for pair in by_dataset[ds_name]]
        best_t, best_ba = sweep_threshold(labels, scores)
        dataset_thresholds[ds_name] = best_t
        logger.info(
            "  %s: t=%.2f, BA=%.1f%% (%d samples)",
            ds_name,
            best_t,
            best_ba * 100,
            len(labels),
        )
        global_labels.extend(labels)
        global_scores.extend(scores)

    global_t, global_ba = sweep_threshold(global_labels, global_scores)
    logger.info("Global threshold: t=%.2f, BA=%.1f%%", global_t, global_ba * 100)

    # Compute oracle BA (per-dataset thresholds)
    per_ds_ba = {}
    for ds_name in sorted(by_dataset):
        t = dataset_thresholds[ds_name]
        labels = [pair[0] for pair in by_dataset[ds_name]]
        scores = [pair[1] for pair in by_dataset[ds_name]]
        preds = [1 if s >= t else 0 for s in scores]
        per_ds_ba[ds_name] = _ba_from_preds(preds, labels)

    oracle_ba = np.mean(list(per_ds_ba.values()))
    logger.info("Oracle BA (per-dataset thresholds): %.2f%%", oracle_ba * 100)
    logger.info("Improvement over global: +%.2fpp", (oracle_ba - global_ba) * 100)

    # Save bundle
    bundle = {
        "classifier": clf,
        "scaler": scaler,
        "feature_cols": TEXT_FEATURE_COLS,
        "mode": "dataset_type",
        "label_names": all_datasets,
        "dataset_thresholds": dataset_thresholds,
        "confidence_gate": args.confidence_gate,
        "metadata": {
            "train_samples": len(x_mat),
            "n_classes": len(all_datasets),
            "cv_ba_mean": float(cv_scores.mean()),
            "cv_ba_std": float(cv_scores.std()),
            "global_threshold": global_t,
            "global_ba": float(global_ba),
            "oracle_ba": float(oracle_ba),
            "per_dataset_thresholds": dataset_thresholds,
            "per_dataset_ba": {k: float(v) for k, v in per_ds_ba.items()},
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = out.stat().st_size / 1024
    logger.info("Saved bundle to %s (%.1f KB)", out, size_kb)
    logger.info("Datasets: %s", all_datasets)
    logger.info("Thresholds: %s", dataset_thresholds)


if __name__ == "__main__":
    main()
