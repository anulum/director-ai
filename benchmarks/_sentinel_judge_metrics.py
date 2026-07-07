# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Sentinel-Judge metrics
"""Metric and report builders for the Sentinel-Judge analyser."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt
from _sentinel_judge_schema import (
    DatasetMetrics,
    EnsembleMetrics,
    LrFusionMetrics,
    SentinelReport,
    align_judges,
    load_judge,
)
from sklearn.linear_model import (  # type: ignore[import-untyped] # scikit-learn has no py.typed metadata in this environment.
    LogisticRegression,
)
from sklearn.model_selection import (  # type: ignore[import-untyped] # scikit-learn has no py.typed metadata in this environment.
    StratifiedKFold,
)

logger = logging.getLogger(__name__)

FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]


class _LogisticRegressionModel(Protocol):
    """Typed subset of the scikit-learn logistic-regression estimator."""

    def fit(self, x: FloatMatrix, y: IntVector) -> _LogisticRegressionModel:
        """Fit the classifier and return the estimator."""

    def predict(self, x: FloatMatrix) -> IntVector:
        """Predict binary labels for a feature matrix."""

    def score(self, x: FloatMatrix, y: IntVector) -> float:
        """Return training accuracy for logging."""


class _LogisticRegressionFactory(Protocol):
    """Factory protocol for constructing a logistic-regression estimator."""

    def __call__(self, **params: int | float | str) -> _LogisticRegressionModel:
        """Construct a typed logistic-regression estimator."""


class _StratifiedKFoldModel(Protocol):
    """Typed subset of scikit-learn's stratified split iterator."""

    def split(
        self,
        x: FloatMatrix,
        y: IntVector,
    ) -> Iterable[tuple[IntVector, IntVector]]:
        """Yield train/test index arrays."""


class _StratifiedKFoldFactory(Protocol):
    """Factory protocol for constructing a stratified split iterator."""

    def __call__(
        self,
        *,
        n_splits: int,
        shuffle: bool,
        random_state: int,
    ) -> _StratifiedKFoldModel:
        """Construct a typed stratified split iterator."""


_logistic_regression = cast(_LogisticRegressionFactory, LogisticRegression)
_stratified_kfold = cast(_StratifiedKFoldFactory, StratifiedKFold)


def balanced_accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    """Return binary balanced accuracy while ignoring ``-1`` abstentions."""
    pos = neg = tp = tn = 0
    for pred, label in zip(preds, labels, strict=True):
        if pred < 0:
            continue
        if label == 1:
            pos += 1
            if pred == 1:
                tp += 1
        else:
            neg += 1
            if pred == 0:
                tn += 1
    if pos == 0 or neg == 0:
        return 0.0
    return (tp / pos + tn / neg) / 2


def per_dataset_ba(
    preds: Sequence[int],
    labels: Sequence[int],
    datasets: Sequence[str],
) -> dict[str, DatasetMetrics]:
    """Return balanced accuracy grouped by dataset name."""
    grouped: dict[str, tuple[list[int], list[int]]] = {}
    for pred, label, dataset in zip(preds, labels, datasets, strict=True):
        grouped.setdefault(dataset, ([], []))
        grouped[dataset][0].append(pred)
        grouped[dataset][1].append(label)
    return {
        dataset: {
            "samples": len(dataset_labels),
            "balanced_accuracy": balanced_accuracy(dataset_preds, dataset_labels),
        }
        for dataset, (dataset_preds, dataset_labels) in grouped.items()
    }


def voting_ensemble(preds_matrix: Sequence[Sequence[int]]) -> list[int]:
    """Return majority-vote predictions, using ``-1`` when all judges abstain."""
    if not preds_matrix:
        raise ValueError("at least one judge prediction vector is required")
    out: list[int] = []
    for sample_idx in range(len(preds_matrix[0])):
        votes = [
            matrix[sample_idx] for matrix in preds_matrix if matrix[sample_idx] >= 0
        ]
        if not votes:
            out.append(-1)
            continue
        ones = sum(1 for vote in votes if vote == 1)
        zeros = sum(1 for vote in votes if vote == 0)
        out.append(1 if ones > zeros else 0 if zeros > ones else votes[0])
    return out


def routed_ensemble(
    preds_matrix: Sequence[Sequence[int]],
    labels: Sequence[int],
    datasets: Sequence[str],
    judge_names: Sequence[str],
) -> tuple[list[int], dict[str, str]]:
    """Route each dataset to the judge with highest train-half accuracy."""
    rng = np.random.default_rng(0)
    by_dataset: dict[str, list[int]] = {}
    for index, dataset in enumerate(datasets):
        by_dataset.setdefault(dataset, []).append(index)

    train_idx: set[int] = set()
    for idxs in by_dataset.values():
        rng.shuffle(idxs)
        train_idx.update(idxs[: len(idxs) // 2])

    routing: dict[str, str] = {}
    for dataset, idxs in by_dataset.items():
        train = [index for index in idxs if index in train_idx]
        if not train:
            routing[dataset] = judge_names[0]
            continue
        best_ba = -1.0
        best_judge = judge_names[0]
        for judge_index, judge_name in enumerate(judge_names):
            ba = balanced_accuracy(
                [preds_matrix[judge_index][index] for index in train],
                [labels[index] for index in train],
            )
            if ba > best_ba:
                best_ba = ba
                best_judge = judge_name
        routing[dataset] = best_judge

    name_to_idx = {name: index for index, name in enumerate(judge_names)}
    return (
        [
            preds_matrix[name_to_idx[routing[dataset]]][index]
            for index, dataset in enumerate(datasets)
        ],
        routing,
    )


def lr_fusion_ensemble(
    scores_matrix: Sequence[Sequence[float]],
    labels: Sequence[int],
    datasets: Sequence[str],
) -> list[int]:
    """Return out-of-fold logistic-regression fusion predictions."""
    if len(labels) < 5:
        raise ValueError("LR fusion requires at least 5 samples")
    if len(set(labels)) < 2:
        raise ValueError("LR fusion requires both binary classes")

    unique_ds = sorted(set(datasets))
    ds_idx = {dataset: index for index, dataset in enumerate(unique_ds)}
    n_samples = len(labels)
    n_judges = len(scores_matrix)
    x = np.zeros((n_samples, n_judges + len(unique_ds)), dtype=np.float64)
    for judge_index, scores in enumerate(scores_matrix):
        if len(scores) != n_samples:
            raise ValueError(
                f"score vector {judge_index} has {len(scores)} samples; "
                f"expected {n_samples}",
            )
        x[:, judge_index] = np.asarray(scores, dtype=np.float64)
    for sample_index, dataset in enumerate(datasets):
        x[sample_index, n_judges + ds_idx[dataset]] = 1.0
    y = np.asarray(labels, dtype=np.int64)

    out_pred = np.full(n_samples, -1, dtype=np.int64)
    for fold, (train, test) in enumerate(
        _stratified_kfold(n_splits=5, shuffle=True, random_state=0).split(x, y),
    ):
        clf = _logistic_regression(max_iter=2000, C=1.0, solver="liblinear")
        clf.fit(x[train], y[train])
        out_pred[test] = clf.predict(x[test])
        logger.info(
            "  fold %d: train=%d test=%d fit_score=%.4f",
            fold + 1,
            len(train),
            len(test),
            clf.score(x[train], y[train]),
        )
    return [int(value) for value in out_pred.tolist()]


def oracle_upper_bound(
    preds_matrix: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> list[int]:
    """Return the theoretical prediction if any judge could be selected per sample."""
    out: list[int] = []
    for sample_index, target in enumerate(labels):
        out.append(
            target
            if any(matrix[sample_index] == target for matrix in preds_matrix)
            else 1 - target,
        )
    return out


def _ensemble_metrics(
    preds: Sequence[int], labels: Sequence[int], datasets: Sequence[str]
) -> EnsembleMetrics:
    """Build a common ensemble metric payload."""
    return {
        "global_balanced_accuracy": balanced_accuracy(preds, labels),
        "per_dataset": per_dataset_ba(preds, labels, datasets),
    }


def build_report(judge_paths: Sequence[str]) -> SentinelReport:
    """Build the Sentinel-Judge ensemble report for judge result files."""
    judges = [load_judge(path) for path in judge_paths]
    judge_names = [judge["name"] for judge in judges]
    logger.info("Loaded %d judges: %s", len(judges), judge_names)

    labels, datasets, preds_matrix, scores_matrix = align_judges(judges)
    logger.info(
        "Aligned %d samples across %d datasets", len(labels), len(set(datasets))
    )

    individual = {
        judge["name"]: _ensemble_metrics(judge["preds"], labels, datasets)
        for judge in judges
    }
    for judge in judges:
        logger.info(
            "  %s: BA=%.4f",
            judge["name"],
            individual[judge["name"]]["global_balanced_accuracy"],
        )

    vote_preds = voting_ensemble(preds_matrix)
    routed_preds, routing = routed_ensemble(preds_matrix, labels, datasets, judge_names)
    oracle_preds = oracle_upper_bound(preds_matrix, labels)
    lr_fusion: LrFusionMetrics | None = None
    if all(judge["scores"] is not None for judge in judges):
        lr_preds = lr_fusion_ensemble(scores_matrix, labels, datasets)
        lr_fusion = {
            **_ensemble_metrics(lr_preds, labels, datasets),
            "method": "5-fold stratified CV, score+dataset_onehot features",
        }

    return {
        "judges": judge_names,
        "samples": len(labels),
        "individual": individual,
        "voting": _ensemble_metrics(vote_preds, labels, datasets),
        "routed": {
            **_ensemble_metrics(routed_preds, labels, datasets),
            "routing_table": routing,
        },
        "lr_fusion": lr_fusion,
        "oracle_upper_bound": _ensemble_metrics(oracle_preds, labels, datasets),
    }
