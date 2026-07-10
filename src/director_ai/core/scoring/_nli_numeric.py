# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Numeric Mapping (softmax, divergence, confidence)
"""Numeric mapping helpers for the NLI scorer.

Pure functions that turn model logits into calibrated scores: row-wise
softmax, divergence and confidence mapping from softmax rows, label-index
resolution from a model config, and the small float-list reducers the
chunk aggregation paths share. Every function with a Rust fast lane
dispatches through :mod:`._nli_accel`, so forcing the pure-Python floor in
a test means patching that one binding module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..mandatory import mandatory_execution
from . import _nli_accel

__all__ = [
    "_count_below_threshold",
    "_mean_float",
    "_probs_to_confidence",
    "_probs_to_divergence",
    "_resolve_label_indices",
    "_softmax_np",
    "_sum_float_list",
    "_weighted_sum_float",
]


def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for 2D numpy array.

    Uses Rust accelerator for large batches when available.
    """
    if _nli_accel._RUST_NLI and x.size >= 100:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            flat = x.flatten().tolist()
            cols = x.shape[1]
            result = _nli_accel.rust_softmax(flat, cols)
            return np.array(result, dtype=np.float64).reshape(x.shape)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    denom = np.asarray(
        [_sum_float_list(row.tolist()) for row in e],
        dtype=np.float64,
    ).reshape(-1, 1)
    s: np.ndarray = e / denom
    return s


def _sum_float_list(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values))


def _mean_float(values: list[float]) -> float:
    if not values:
        return 0.0
    return _sum_float_list(values) / len(values)


def _count_below_threshold(values: list[float], threshold: float) -> int:
    if not values:
        return 0
    flags = np.asarray(values, dtype=np.float64) < float(threshold)
    return int(np.count_nonzero(flags))


def _weighted_sum_float(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    vec = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return _sum_float_list((vec * w).tolist())


def _resolve_label_indices(model: Any) -> tuple[int, int]:
    """Read model.config.id2label to find contradiction and neutral indices.

    Returns (contradiction_idx, neutral_idx). Falls back to (2, 1) if
    id2label is missing or labels are unrecognisable.
    """
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if not id2label:
        return (2, 1)
    contra_idx = 2
    neutral_idx = 1
    for idx, label in id2label.items():
        normed = str(label).lower().strip()
        if normed in ("contradiction", "contradict"):
            contra_idx = int(idx)
        elif normed == "neutral":
            neutral_idx = int(idx)
    return (contra_idx, neutral_idx)


def _probs_to_divergence(
    probs: np.ndarray,
    label_indices: tuple[int, int] | None = None,
) -> list[float]:
    """Convert softmax rows to divergence scores.

    2-class: divergence = 1 - P(supported).
    3-class: divergence = P(contradiction) + 0.5 * P(neutral).
    ``label_indices`` is (contradiction_idx, neutral_idx) from
    ``_resolve_label_indices``; defaults to (2, 1).

    Uses Rust accelerator for large batches when available.
    """
    ncols = probs.shape[1]
    ci, ni = label_indices or (2, 1)
    if _nli_accel._RUST_NLI and probs.shape[0] >= 10:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            flat = probs.flatten().tolist()
            return [
                float(v)
                for v in _nli_accel.rust_probs_to_divergence(flat, ncols, ci, ni)
            ]
    if ncols == 2:
        return [float(1.0 - row[1]) for row in probs]
    return [float(row[ci]) + float(row[ni]) * 0.5 for row in probs]


def _probs_to_confidence(probs: np.ndarray) -> list[float]:
    """Convert softmax rows to confidence scores.

    Confidence = 1 - H(p)/log(K) where H is entropy and K is num classes.
    Returns values in [0, 1]: 1 = maximally confident (one-hot),
    0 = maximally uncertain (uniform).

    Uses Rust accelerator for large batches when available.
    """
    ncols = probs.shape[1]
    if _nli_accel._RUST_NLI and probs.shape[0] >= 10:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            flat = probs.flatten().tolist()
            return [float(v) for v in _nli_accel.rust_probs_to_confidence(flat, ncols)]
    log_k = float(np.log(ncols)) if ncols > 1 else 1.0
    result: list[float] = []
    for row in probs:
        clipped = np.clip(row, 1e-10, 1.0)
        entropy = -_sum_float_list((clipped * np.log(clipped)).tolist())
        result.append(max(0.0, 1.0 - entropy / log_k))
    return result
