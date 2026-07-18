# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Probability Calibration

"""Post-hoc probability calibration for the coherence score.

The raw coherence score separates grounded from hallucinated responses well
(the 2026-07-18 grounded-operating-point campaign measured good median 0.93 vs
bad median 0.06 on 20 000 samples) but it is not a *calibrated* probability:
a score of 0.7 does not mean "70 % likely grounded". A red-team review flagged
the missing per-task calibration and an Expected Calibration Error worse than
climatology on one slice.

This module supplies the two standard post-hoc maps and their reliability
metrics, learned from held-out ``(score, label)`` pairs so a published number
carries a calibrated confidence:

* :func:`expected_calibration_error` / :func:`brier_score` / :func:`reliability_bins`
  measure how far a probability is from the observed frequency;
* :class:`IsotonicCalibrator` fits a monotone step map by pool-adjacent-violators
  — non-parametric, exact on the training grid, ideal when enough data exists;
* :class:`PlattCalibrator` fits a two-parameter sigmoid by the regularised
  Newton method of Lin, Lin & Weng (2007) — robust on smaller samples.

All computation is NumPy on pre-collected arrays; the calibrators are frozen
value objects so a fitted map can be serialised into a versioned artefact and
re-loaded without re-fitting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def _as_prob_label_arrays(
    probabilities: Sequence[float], labels: Sequence[int | bool]
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs to aligned float/binary arrays."""
    probs = np.asarray(probabilities, dtype=np.float64)
    ys = np.asarray(labels, dtype=np.float64)
    if probs.shape != ys.shape:
        raise ValueError("probabilities and labels must have the same length")
    if probs.ndim != 1:
        raise ValueError("probabilities and labels must be one-dimensional")
    if probs.size == 0:
        raise ValueError("probabilities and labels must be non-empty")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.all((ys == 0.0) | (ys == 1.0)):
        raise ValueError("labels must be binary (0/1 or bool)")
    return probs, ys


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``1 / (1 + exp(z))`` (evaluates ``exp`` on one side).

    The naive form overflows for large positive *z*; splitting on the sign
    keeps every ``exp`` argument non-positive so no overflow or NaN can enter.
    """
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0.0
    ez = np.exp(-z[pos])
    out[pos] = ez / (1.0 + ez)
    ez = np.exp(z[~pos])
    out[~pos] = 1.0 / (1.0 + ez)
    return out


def reliability_bins(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Return ``(mean_confidence, empirical_accuracy, count)`` per occupied bin.

    Probabilities are grouped into *n_bins* equal-width bins over ``[0, 1]``;
    the top bin is closed on the right so ``p == 1.0`` lands in the last bin.
    Empty bins are omitted.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    probs, ys = _as_prob_label_arrays(probabilities, labels)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=False puts p==1.0 in an overflow bin; clip it back.
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[tuple[float, float, int]] = []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append((float(probs[mask].mean()), float(ys[mask].mean()), count))
    return bins


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[int | bool],
    *,
    n_bins: int = 10,
) -> float:
    """Return the sample-weighted mean confidence-accuracy gap over bins.

    The Expected Calibration Error is
    ``sum_b (n_b / N) * |accuracy_b - confidence_b|``. Zero means every bin's
    predicted confidence matches its observed accuracy; the score lies in
    ``[0, 1]`` and is lower-is-better.
    """
    probs, _ys = _as_prob_label_arrays(probabilities, labels)
    total = probs.size
    ece = 0.0
    for mean_conf, empirical_acc, count in reliability_bins(
        probabilities, labels, n_bins=n_bins
    ):
        ece += (count / total) * abs(empirical_acc - mean_conf)
    return ece


def brier_score(probabilities: Sequence[float], labels: Sequence[int | bool]) -> float:
    """Mean squared error between probability and outcome (lower is better)."""
    probs, ys = _as_prob_label_arrays(probabilities, labels)
    return float(np.mean((probs - ys) ** 2))


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Monotone non-decreasing step calibration fitted by pool-adjacent-violators.

    ``x_thresholds`` and ``y_values`` are the (ascending) fitted step points;
    :meth:`transform` linearly interpolates between them and clamps to the
    endpoints outside the training range.
    """

    x_thresholds: tuple[float, ...]
    y_values: tuple[float, ...]

    @classmethod
    def fit(
        cls, scores: Sequence[float], labels: Sequence[int | bool]
    ) -> IsotonicCalibrator:
        """Fit a monotone map from *scores* to the binary *labels* by PAV.

        Points are sorted by score; ties are averaged; the pool-adjacent-
        violators algorithm produces the least-squares monotone fit, which is
        then thinned to the distinct breakpoints.
        """
        xs, ys = _as_prob_label_arrays(scores, labels)
        order = np.argsort(xs, kind="mergesort")
        xs = xs[order]
        ys = ys[order]

        # Merge equal x into a single weighted point before PAV so the map is
        # a well-defined function of the score.
        uniq_x, inverse = np.unique(xs, return_inverse=True)
        sums = np.zeros(uniq_x.size, dtype=np.float64)
        counts = np.zeros(uniq_x.size, dtype=np.float64)
        np.add.at(sums, inverse, ys)
        np.add.at(counts, inverse, 1.0)
        values = sums / counts

        fitted = _pool_adjacent_violators(values, counts)
        return cls(
            x_thresholds=tuple(float(x) for x in uniq_x),
            y_values=tuple(float(v) for v in fitted),
        )

    def transform(self, scores: Sequence[float]) -> list[float]:
        """Map raw *scores* to calibrated probabilities by linear interpolation."""
        xs = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(xs)):
            raise ValueError("scores must be finite")
        knots_x = np.asarray(self.x_thresholds, dtype=np.float64)
        knots_y = np.asarray(self.y_values, dtype=np.float64)
        calibrated = np.interp(xs, knots_x, knots_y)
        return [float(v) for v in calibrated]


def _pool_adjacent_violators(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Least-squares isotonic (non-decreasing) fit via pool-adjacent-violators."""
    level_values = list(values.astype(np.float64))
    level_weights = list(weights.astype(np.float64))
    level_sizes = [1] * len(level_values)

    i = 0
    while i < len(level_values) - 1:
        if level_values[i] <= level_values[i + 1]:
            i += 1
            continue
        # Pool the violating pair into a weighted mean, then back up so an
        # earlier level that now violates monotonicity is re-pooled.
        w = level_weights[i] + level_weights[i + 1]
        merged = (
            level_values[i] * level_weights[i]
            + level_values[i + 1] * level_weights[i + 1]
        ) / w
        level_values[i] = merged
        level_weights[i] = w
        level_sizes[i] += level_sizes[i + 1]
        del level_values[i + 1]
        del level_weights[i + 1]
        del level_sizes[i + 1]
        if i > 0:
            i -= 1

    out = np.empty(int(sum(level_sizes)), dtype=np.float64)
    pos = 0
    for value, size in zip(level_values, level_sizes, strict=True):
        out[pos : pos + size] = value
        pos += size
    return out


@dataclass(frozen=True)
class PlattCalibrator:
    """Two-parameter sigmoid calibration ``P = 1 / (1 + exp(a * score + b))``.

    Fitted by the regularised Newton method of Lin, Lin & Weng (2007), which
    replaces the 0/1 targets with ``(N+ + 1)/(N+ + 2)`` and ``1/(N- + 2)`` to
    avoid overfitting and adds a line search for numerical robustness.
    """

    a: float
    b: float

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[int | bool],
        *,
        max_iter: int = 100,
        min_step: float = 1e-10,
        sigma: float = 1e-12,
    ) -> PlattCalibrator:
        """Fit the sigmoid parameters to *scores* and binary *labels*.

        Follows Lin, Lin & Weng (2007) Algorithm 3: regularised targets, the
        negative-log-likelihood objective, a Newton direction with Hessian
        stabilisation *sigma*, and a backtracking line search so the fit stays
        finite even on (near-)separable data instead of diverging.
        """
        xs, ys = _as_prob_label_arrays(scores, labels)
        prior1 = float(ys.sum())
        prior0 = float(ys.size) - prior1
        hi = (prior1 + 1.0) / (prior1 + 2.0)
        lo = 1.0 / (prior0 + 2.0)
        targets = np.where(ys > 0.0, hi, lo)

        a = 0.0
        b = math.log((prior0 + 1.0) / (prior1 + 1.0))

        def nll(a_: float, b_: float) -> float:
            fapb = a_ * xs + b_
            # log(1 + exp(fApB)) computed stably on both signs of fApB.
            log1p_exp = np.where(
                fapb >= 0.0,
                fapb + np.log1p(np.exp(-np.abs(fapb))),
                np.log1p(np.exp(-np.abs(fapb))),
            )
            return float(np.sum((targets - 1.0) * fapb + log1p_exp))

        obj = nll(a, b)
        for _ in range(max_iter):
            fapb = a * xs + b
            p = _sigmoid(fapb)
            q = 1.0 - p
            h11 = float(np.sum(xs * xs * p * q)) + sigma
            h22 = float(np.sum(p * q)) + sigma
            h21 = float(np.sum(xs * p * q))
            g1 = float(np.sum(xs * (targets - p)))
            g2 = float(np.sum(targets - p))
            if abs(g1) < 1e-5 and abs(g2) < 1e-5:
                break
            det = h11 * h22 - h21 * h21
            # Newton direction (minimising the NLL: gradient is -g).
            da = -(h22 * g1 - h21 * g2) / det
            db = -(-h21 * g1 + h11 * g2) / det
            gd = g1 * da + g2 * db  # directional derivative of the NLL

            step = 1.0
            while step >= min_step:
                new_a = a + step * da
                new_b = b + step * db
                new_obj = nll(new_a, new_b)
                if new_obj < obj + 1e-4 * step * gd:
                    a, b, obj = new_a, new_b, new_obj
                    break
                step /= 2.0
            else:
                # Line search could not improve — the fit has converged.
                break
        return cls(a=float(a), b=float(b))

    def transform(self, scores: Sequence[float]) -> list[float]:
        """Map raw *scores* to calibrated probabilities through the sigmoid."""
        xs = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(xs)):
            raise ValueError("scores must be finite")
        calibrated = _sigmoid(self.a * xs + self.b)
        return [float(v) for v in calibrated]
