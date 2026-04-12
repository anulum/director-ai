# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shared numpy-based analysis utilities
"""Numpy-based metrics, threshold sweeps, and LR fusion used by
``post_phase1_analysis.py`` and related ensemble analysis scripts.

These are the numpy-vectorised equivalents of the list-based helpers in
``_judge_common.py``, designed for post-hoc analysis of saved results
rather than inline benchmark loops.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ── Metrics ──────────────────────────────────────────────────────────────


def balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Standard two-class balanced accuracy with -1 (unknown) filtering.

    Returns 0.0 when either class has zero predictions after filtering.
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
    scores: np.ndarray, labels: np.ndarray,
) -> tuple[float, float]:
    """Sweep 0.05..0.95 in 0.05 steps, return ``(best_t, best_ba)``."""
    best_t, best_ba = 0.5, 0.0
    for t in [0.05 * i for i in range(1, 20)]:
        preds = (scores >= t).astype(int)
        ba = balanced_accuracy(preds, labels)
        if ba > best_ba:
            best_t, best_ba = t, ba
    return best_t, best_ba


def per_dataset_threshold_sweep(
    scores: np.ndarray, labels: np.ndarray, datasets: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-dataset threshold optimisation."""
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
    """Return *x* as a 1-D float array, replacing ``None`` with NaN."""
    if x is None:
        return None
    return np.array(
        [float("nan") if v is None else float(v) for v in x],
        dtype=float,
    )


def load_judge(path: Path) -> dict[str, Any]:
    """Load a judge JSON, normalising field names across schemas."""
    with open(path) as f:  # noqa: PTH123
        d = json.load(f)
    name = Path(d.get("model", path.stem)).name or path.stem
    labels = np.array(d["labels"], dtype=int)
    datasets = d.get("datasets_per_sample") or d.get("datasets")
    scores = _arr(d.get("scores"))
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
    """5-fold CV logistic regression. Returns predictions, avg coefs, avg intercept."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    n = feature_matrix.shape[0]
    preds = np.zeros(n, dtype=int)
    coef_sums = np.zeros(feature_matrix.shape[1])
    intercept_sum = 0.0
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for _fold, (train_idx, test_idx) in enumerate(kf.split(feature_matrix)):
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
    score_arrays: list[np.ndarray], names: list[str],
) -> dict[str, dict[str, float]]:
    """Pairwise Pearson correlation between every pair of score arrays."""
    out: dict[str, dict[str, float]] = {}
    for i, ni in enumerate(names):
        out[ni] = {}
        for j, nj in enumerate(names):
            r = float(np.corrcoef(score_arrays[i], score_arrays[j])[0, 1])
            out[ni][nj] = round(r, 4)
    return out
