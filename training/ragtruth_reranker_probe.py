# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — RAGTruth exploratory example-level reranker probe

"""Probe whether cached token scores contain enough signal for a reranker."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from training.eval_ragtruth_token import (
        _confusion_metrics,
        _enrich_cached_records,
    )
except ModuleNotFoundError:  # pragma: no cover - flat script mode
    from eval_ragtruth_token import _confusion_metrics, _enrich_cached_records


DEFAULT_CACHE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/token_eval_probs.json"
)
DEFAULT_OUTPUT_JSON = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_reranker_probe.json"
)
DEFAULT_TEST_CACHE = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/token_eval_probs.json"
)
DEFAULT_OUTPUT_MD = Path(
    "/media/anulum/GOTM/_scratch/director_ragtruth/jarvis_20260619/"
    "fp_a/diagnostics/ragtruth_reranker_probe.md"
)

PROBABILITY_THRESHOLDS = (0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99)
TASK_TYPES = ("QA", "Summary", "Data2txt")


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def feature_names(*, include_metadata: bool) -> list[str]:
    """Return the ordered feature names used by `record_features`."""
    names = [
        "max_probability",
        "mean_probability",
        "std_probability",
        "q50_probability",
        "q75_probability",
        "q90_probability",
        "q95_probability",
        "q99_probability",
    ]
    for threshold in PROBABILITY_THRESHOLDS:
        suffix = str(threshold).replace(".", "")
        names.extend([f"count_ge_{suffix}", f"density_ge_{suffix}"])
    names.extend(["response_tokens", "log_response_tokens"])
    if include_metadata:
        names.extend(
            [
                "log_context_tokens",
                "log_context_chars",
                "log_response_chars",
                *[f"task_is_{task_type}" for task_type in TASK_TYPES],
            ]
        )
    return names


def record_features(record: dict[str, Any], *, include_metadata: bool) -> list[float]:
    """Extract deterministic example-level features from cached token scores."""
    probabilities = np.array(record.get("resp_probs", []), dtype=float)
    if probabilities.size == 0:
        probabilities = np.array([0.0], dtype=float)
    token_count = max(1, int(probabilities.size))
    features = [
        float(np.max(probabilities)),
        float(np.mean(probabilities)),
        float(np.std(probabilities)),
        *[
            float(value)
            for value in np.quantile(probabilities, [0.5, 0.75, 0.9, 0.95, 0.99])
        ],
    ]
    for threshold in PROBABILITY_THRESHOLDS:
        count = float(np.sum(probabilities >= threshold))
        features.extend([count, count / token_count])
    features.extend([float(token_count), math.log1p(token_count)])
    if include_metadata:
        task_type = str(record.get("task_type", "unknown"))
        features.extend(
            [
                math.log1p(float(record.get("context_tokens") or 0)),
                math.log1p(float(record.get("context_chars") or 0)),
                math.log1p(float(record.get("response_chars") or 0)),
                *[float(task_type == known_task) for known_task in TASK_TYPES],
            ]
        )
    return features


def _feature_matrix(
    records: list[dict[str, Any]], *, include_metadata: bool
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.array([int(record["label"]) for record in records], dtype=int)
    matrix = np.array(
        [
            record_features(record, include_metadata=include_metadata)
            for record in records
        ],
        dtype=float,
    )
    return matrix, labels


def _candidate_models() -> dict[str, Callable[[], Any]]:
    return {
        "logreg_c0_1": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.1,
                random_state=17,
            ),
        ),
        "logreg_c0_3": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.3,
                random_state=17,
            ),
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=17,
            n_jobs=-1,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=17,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.04,
            max_depth=2,
            random_state=17,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=80,
            l2_regularization=2.0,
            learning_rate=0.04,
            max_leaf_nodes=7,
            random_state=17,
        ),
    }


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    max_fpr: float,
    min_recall: float,
    min_f1: float,
) -> dict[str, Any]:
    """Choose a fold-local operating threshold with FPR-first ranking."""
    candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.05, 0.95, 181):
        metrics = _confusion_metrics(labels, probabilities >= threshold)
        candidates.append({"threshold": float(threshold), **metrics})
    gate = [
        row
        for row in candidates
        if row["fpr"] <= max_fpr and row["recall"] >= min_recall and row["f1"] >= min_f1
    ]
    if not gate:
        gate = [
            row
            for row in candidates
            if row["fpr"] <= max_fpr and row["recall"] >= min_recall
        ]
    if not gate:
        gate = [row for row in candidates if row["fpr"] <= max_fpr]
    return max(
        gate or candidates,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
            -float(row["fpr"]),
        ),
    )


def aggregate_confusions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate fold confusion counts and recompute metrics."""
    tp = int(sum(int(row["tp"]) for row in rows))
    fp = int(sum(int(row["fp"]) for row in rows))
    tn = int(sum(int(row["tn"]) for row in rows))
    fn = int(sum(int(row["fn"]) for row in rows))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "balanced_accuracy": (recall + tnr) / 2,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def cross_validate_probe(
    records: list[dict[str, Any]],
    *,
    include_metadata: bool,
    n_splits: int,
    max_fpr: float,
    min_recall: float,
    min_f1: float,
) -> list[dict[str, Any]]:
    """Run deterministic stratified CV for several lightweight rerankers."""
    matrix, labels = _feature_matrix(records, include_metadata=include_metadata)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=17)
    results: list[dict[str, Any]] = []
    for model_name, model_factory in _candidate_models().items():
        fold_rows: list[dict[str, Any]] = []
        for fold_index, (train_idx, test_idx) in enumerate(
            splitter.split(matrix, labels), start=1
        ):
            model = model_factory()
            model.fit(matrix[train_idx], labels[train_idx])
            train_probabilities = model.predict_proba(matrix[train_idx])[:, 1]
            test_probabilities = model.predict_proba(matrix[test_idx])[:, 1]
            selected = select_threshold(
                labels[train_idx],
                train_probabilities,
                max_fpr=max_fpr,
                min_recall=min_recall,
                min_f1=min_f1,
            )
            test_metrics = _confusion_metrics(
                labels[test_idx],
                test_probabilities >= float(selected["threshold"]),
            )
            fold_rows.append(
                {
                    "fold": fold_index,
                    "threshold": selected["threshold"],
                    "train_selected": selected,
                    **test_metrics,
                }
            )
        aggregate = aggregate_confusions(fold_rows)
        results.append(
            {
                "model": model_name,
                "include_metadata": include_metadata,
                "aggregate": aggregate,
                "folds": fold_rows,
            }
        )
    return results


def evaluate_calibration_to_test(
    calibration_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    *,
    include_metadata: bool,
    max_fpr: float,
    min_recall: float,
    min_f1: float,
) -> list[dict[str, Any]]:
    """Fit rerankers on calibration cache and score a separate test cache."""
    calibration_matrix, calibration_labels = _feature_matrix(
        calibration_records,
        include_metadata=include_metadata,
    )
    test_matrix, test_labels = _feature_matrix(
        test_records,
        include_metadata=include_metadata,
    )
    results: list[dict[str, Any]] = []
    for model_name, model_factory in _candidate_models().items():
        model = model_factory()
        model.fit(calibration_matrix, calibration_labels)
        calibration_probabilities = model.predict_proba(calibration_matrix)[:, 1]
        test_probabilities = model.predict_proba(test_matrix)[:, 1]
        selected = select_threshold(
            calibration_labels,
            calibration_probabilities,
            max_fpr=max_fpr,
            min_recall=min_recall,
            min_f1=min_f1,
        )
        test_metrics = _confusion_metrics(
            test_labels,
            test_probabilities >= float(selected["threshold"]),
        )
        results.append(
            {
                "model": model_name,
                "include_metadata": include_metadata,
                "selected_threshold": selected,
                "aggregate": test_metrics,
                "folds": [],
            }
        )
    return results


def _rank_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            float(row["aggregate"]["f1"]),
            -float(row["aggregate"]["fpr"]),
            float(row["aggregate"]["precision"]),
            float(row["aggregate"]["recall"]),
        ),
        reverse=True,
    )


def build_probe(
    *,
    cache_path: Path,
    calibration_cache_path: Path | None = None,
    test_cache_path: Path | None = None,
    n_splits: int = 5,
    max_fpr: float = 0.08,
    min_recall: float = 0.70,
    min_f1: float = 0.763,
) -> dict[str, Any]:
    """Build an exploratory or calibration-to-test reranker packet."""
    all_results = []
    method = "exploratory_stratified_cv_on_test_cache_not_promotion_evidence"
    if calibration_cache_path is not None and test_cache_path is not None:
        method = "calibration_cache_threshold_selection_then_test_cache_evaluation"
        calibration_records = _enrich_cached_records(_load_json(calibration_cache_path))
        test_records = _enrich_cached_records(_load_json(test_cache_path))
        for include_metadata in (False, True):
            all_results.extend(
                evaluate_calibration_to_test(
                    calibration_records,
                    test_records,
                    include_metadata=include_metadata,
                    max_fpr=max_fpr,
                    min_recall=min_recall,
                    min_f1=min_f1,
                )
            )
    else:
        records = _enrich_cached_records(_load_json(cache_path))
        for include_metadata in (False, True):
            all_results.extend(
                cross_validate_probe(
                    records,
                    include_metadata=include_metadata,
                    n_splits=n_splits,
                    max_fpr=max_fpr,
                    min_recall=min_recall,
                    min_f1=min_f1,
                )
            )
    ranked = _rank_results(all_results)
    best = ranked[0]
    best_gate = [
        row
        for row in ranked
        if row["aggregate"]["f1"] > min_f1
        and row["aggregate"]["fpr"] <= max_fpr
        and row["aggregate"]["recall"] >= min_recall
    ]
    return {
        "cache_path": str(cache_path),
        "calibration_cache_path": (
            str(calibration_cache_path) if calibration_cache_path is not None else None
        ),
        "test_cache_path": str(test_cache_path)
        if test_cache_path is not None
        else None,
        "method": method,
        "feature_sets": {
            "token_score_only": feature_names(include_metadata=False),
            "token_score_plus_metadata": feature_names(include_metadata=True),
        },
        "gates": {
            "min_f1": min_f1,
            "max_fpr": max_fpr,
            "min_recall": min_recall,
        },
        "best": best,
        "best_gate_candidate": best_gate[0] if best_gate else None,
        "results": ranked,
        "decision": (
            "eligible_for_one_confirmation_eval"
            if best_gate
            else "do_not_launch_jarvis_from_reranker_probe"
        ),
    }


def _metric_line(row: dict[str, Any] | None) -> str:
    if not row:
        return "none"
    metrics = row["aggregate"]
    return (
        f"{row['model']} metadata={row['include_metadata']}: "
        f"F1 {metrics['f1']:.4f}, precision {metrics['precision']:.4f}, "
        f"recall {metrics['recall']:.4f}, FPR {metrics['fpr']:.4f}"
    )


def write_markdown(packet: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown report for the reranker probe."""
    lines = [
        "# RAGTruth Reranker Probe",
        "",
        f"Method: `{packet['method']}`",
        f"Decision: `{packet['decision']}`",
        f"Calibration cache: `{packet['calibration_cache_path']}`",
        f"Test cache: `{packet['test_cache_path']}`",
        "",
        "## Best Candidate",
        "",
        f"- {_metric_line(packet['best'])}",
        f"- Gate-passing candidate: {_metric_line(packet['best_gate_candidate'])}",
        "",
        "## Ranked Results",
        "",
    ]
    for row in packet["results"]:
        lines.append(f"- {_metric_line(row)}")
    lines.extend(
        [
            "",
            "Interpretation: this probe uses cross-validation over the existing "
            "test cache, so it is exploratory only. A cloud run is justified only "
            "after the same reranker family clears the gates on a separate train "
            "or calibration cache and then evaluates once on the held-out test split.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_TEST_CACHE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-fpr", type=float, default=0.08)
    parser.add_argument("--min-recall", type=float, default=0.70)
    parser.add_argument("--min-f1", type=float, default=0.763)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packet = build_probe(
        cache_path=args.cache,
        calibration_cache_path=args.calibration_cache,
        test_cache_path=args.test_cache if args.calibration_cache else None,
        n_splits=args.folds,
        max_fpr=args.max_fpr,
        min_recall=args.min_recall,
        min_f1=args.min_f1,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(packet, indent=2, sort_keys=True))
    write_markdown(packet, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    print(f"decision: {packet['decision']}")
    print(f"best: {_metric_line(packet['best'])}")
    print(f"best gate candidate: {_metric_line(packet['best_gate_candidate'])}")


if __name__ == "__main__":
    main()
