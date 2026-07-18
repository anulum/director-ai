# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AggreFact Metrics

"""Metric primitives for the AggreFact benchmark: balanced accuracy,
per-label precision/recall/F1, the two aggregate conventions
(per-dataset mean vs sample-pooled), and the metrics container.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


def balanced_accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Return balanced accuracy for binary or categorical labels.

    The cached-score replay path intentionally works without the optional
    ``scikit-learn`` training stack. This implementation covers the benchmark
    contract directly: balanced accuracy is the mean recall over labels present
    in ``y_true``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        return 0.0

    recalls: list[float] = []
    for label in sorted(set(y_true)):
        total = 0
        correct = 0
        for truth, prediction in zip(y_true, y_pred, strict=True):
            if truth != label:
                continue
            total += 1
            if prediction == label:
                correct += 1
        if total:
            recalls.append(correct / total)
    return float(np.mean(recalls)) if recalls else 0.0


def _precision_recall_f1_for_label(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label: int,
) -> tuple[float, float, float]:
    """Return zero-division-safe precision, recall, and F1 for one label."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    true_positive = 0
    false_positive = 0
    false_negative = 0
    for truth, prediction in zip(y_true, y_pred, strict=True):
        if prediction == label:
            if truth == label:
                true_positive += 1
            else:
                false_positive += 1
        elif truth == label:
            false_negative += 1

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


@dataclass
class AggreFactMetrics:
    """Per-dataset balanced accuracy plus the two aggregate metrics
    that the AggreFact community uses interchangeably.

    Two distinct aggregates are exposed and **must not be confused**:

    - ``per_dataset_mean_balanced_acc`` — unweighted mean of the
      per-dataset balanced accuracies. **This is the AggreFact
      leaderboard convention** (verified verbatim from
      https://llm-aggrefact.github.io/ on 2026-04-12). Heterogeneous
      benchmarks are evaluated this way to prevent the largest
      dataset (RAGTruth, ~16 K samples) from dominating the score.
    - ``sample_pooled_balanced_acc`` — balanced accuracy computed
      once across the flat sample pool, weighted by dataset size.
      Convenient for fast comparison of judges on the same data
      distribution but **not the leaderboard metric**.

    Historical note: the legacy ``avg_balanced_acc`` property
    returned the per-dataset mean. It is preserved as an alias for
    backwards compatibility but new code should call
    ``per_dataset_mean_balanced_acc`` explicitly so the metric is
    unambiguous in every call site.
    """

    per_dataset: dict[str, dict] = field(default_factory=dict)
    threshold: float = 0.5
    per_dataset_thresholds: dict[str, float] = field(default_factory=dict)
    inference_times: list[float] = field(default_factory=list, repr=False)

    @property
    def per_dataset_mean_balanced_acc(self) -> float:
        """Unweighted mean of per-dataset BAs (AggreFact leaderboard
        convention)."""
        accs = [d["balanced_acc"] for d in self.per_dataset.values() if d["total"] > 0]
        return float(np.mean(accs)) if accs else 0.0

    @property
    def avg_balanced_acc(self) -> float:
        """**Deprecated alias** — same value as
        ``per_dataset_mean_balanced_acc``.

        Kept so that pre-2026-04-12 callers don't break, but new
        code should use the explicit name. Every call site of
        ``avg_balanced_acc`` in the repo should be migrated to
        either ``per_dataset_mean_balanced_acc`` (when the
        leaderboard metric is intended) or
        ``sample_pooled_balanced_acc`` (when sample-pooled is
        intended).
        """
        return self.per_dataset_mean_balanced_acc

    @property
    def total_samples(self) -> int:
        return sum(d["total"] for d in self.per_dataset.values())

    @property
    def avg_latency_ms(self) -> float:
        if not self.inference_times:
            return 0.0
        return float(np.mean(self.inference_times)) * 1000

    def to_dict(self) -> dict:
        return {
            "avg_balanced_accuracy": round(self.avg_balanced_acc, 4),
            "avg_balanced_accuracy_pct": round(self.avg_balanced_acc * 100, 1),
            "threshold": self.threshold,
            "total_samples": self.total_samples,
            "per_dataset": {
                k: {
                    kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()
                }
                for k, v in self.per_dataset.items()
            },
            "latency_ms_avg": round(self.avg_latency_ms, 2),
            **(
                {"per_dataset_thresholds": self.per_dataset_thresholds}
                if self.per_dataset_thresholds
                else {}
            ),
        }


def _binary_class_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Precision/recall/F1 for both supported (1) and hallucination (0) classes."""
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) < 2:
        return {}
    hall_prec, hall_rec, hall_f1 = _precision_recall_f1_for_label(y_true, y_pred, 0)
    supp_prec, supp_rec, supp_f1 = _precision_recall_f1_for_label(y_true, y_pred, 1)
    return {
        "hallucination_precision": float(hall_prec),
        "hallucination_recall": float(hall_rec),
        "hallucination_f1": float(hall_f1),
        "supported_precision": float(supp_prec),
        "supported_recall": float(supp_rec),
        "supported_f1": float(supp_f1),
    }


def _compute_sample_pooled_ba(predictions: list[int], labels: list[int]) -> float:
    """True sample-pooled balanced accuracy on the flat (preds, labels)
    pool. Predictions of -1 (unknown) are dropped from the count.

    Returns 0.0 when either class has zero predictions left.
    Distinct from ``AggreFactMetrics.per_dataset_mean_balanced_acc``
    which averages BAs across datasets — see the docstring of
    ``AggreFactMetrics`` for the discussion of when to use which.
    """
    pos = neg = tp = tn = 0
    for p, lab in zip(predictions, labels, strict=True):
        if p < 0:
            continue
        if lab == 1:
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
