# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for Phase 5 Gem 5: Conformal Prediction Intervals.

Covers: uncalibrated/calibrated prediction, coverage narrowing, many
samples, mismatched lengths guard, invalid coverage guard, dataclass
fields, bounds validity, parametrised scores/coverage, pipeline
integration with scorer, and performance documentation.
"""

from __future__ import annotations

import pytest

from director_ai.core.calibration.conformal import (
    ConformalPredictor,
    PredictionInterval,
)


class TestConformalPredictor:
    def test_uncalibrated_returns_full_interval(self):
        cp = ConformalPredictor()
        pi = cp.predict(0.7)
        assert pi.lower == 0.0
        assert pi.upper == 1.0
        assert pi.calibration_size == 0
        assert not pi.is_reliable

    def test_calibrated_narrows_interval(self):
        cp = ConformalPredictor(coverage=0.95, min_samples=5)
        # Calibration: 50 responses, all correct (score > 0.7, label=False)
        scores = [0.8 + i * 0.001 for i in range(50)]
        labels = [False] * 50
        cp.calibrate(scores, labels)
        pi = cp.predict(0.85)
        assert pi.lower < pi.upper
        assert pi.upper < 1.0  # narrower than uncalibrated
        assert pi.calibration_size == 50
        assert pi.is_reliable

    def test_mixed_calibration(self):
        cp = ConformalPredictor(coverage=0.90, min_samples=10)
        scores = [0.9, 0.85, 0.7, 0.3, 0.2, 0.8, 0.75, 0.6, 0.4, 0.95]
        labels = [False, False, False, True, True, False, False, True, True, False]
        cp.calibrate(scores, labels)
        pi = cp.predict(0.5)
        assert 0.0 <= pi.lower <= pi.point_estimate
        assert pi.point_estimate <= pi.upper <= 1.0
        assert pi.is_reliable

    def test_point_estimate_is_inverted_score(self):
        cp = ConformalPredictor()
        cp.calibrate([0.5], [False])
        pi = cp.predict(0.8)
        assert abs(pi.point_estimate - 0.2) < 0.001  # 1 - 0.8 = 0.2

    def test_high_score_low_hallucination_prob(self):
        cp = ConformalPredictor(min_samples=5)
        cp.calibrate([0.9] * 10, [False] * 10)
        pi = cp.predict(0.95)
        assert pi.point_estimate < 0.1
        assert pi.lower < 0.1

    def test_low_score_high_hallucination_prob(self):
        cp = ConformalPredictor(min_samples=5)
        cp.calibrate([0.3] * 10, [True] * 10)
        pi = cp.predict(0.2)
        assert pi.point_estimate > 0.7

    def test_below_min_samples_unreliable(self):
        cp = ConformalPredictor(min_samples=50)
        cp.calibrate([0.8] * 10, [False] * 10)
        pi = cp.predict(0.7)
        assert not pi.is_reliable
        assert pi.calibration_size == 10


class TestConformalValidation:
    def test_mismatched_lengths_raises(self):
        cp = ConformalPredictor()
        with pytest.raises(ValueError, match="same length"):
            cp.calibrate([0.5, 0.6], [True])

    def test_invalid_coverage_raises(self):
        with pytest.raises(ValueError, match="coverage"):
            ConformalPredictor(coverage=0.0)
        with pytest.raises(ValueError, match="coverage"):
            ConformalPredictor(coverage=1.0)


class TestPredictionInterval:
    def test_dataclass_fields(self):
        pi = PredictionInterval(
            point_estimate=0.15,
            lower=0.05,
            upper=0.25,
            coverage=0.95,
            calibration_size=100,
            is_reliable=True,
        )
        assert pi.point_estimate == 0.15
        assert pi.lower == 0.05
        assert pi.upper == 0.25

    @pytest.mark.parametrize("score", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_bounds_are_valid(self, score):
        cp = ConformalPredictor(min_samples=5)
        cp.calibrate([0.7 + i * 0.01 for i in range(20)], [False] * 20)
        pi = cp.predict(score)
        assert 0.0 <= pi.lower <= pi.upper <= 1.0


class TestConformalParametrised:
    """Parametrised conformal prediction tests."""

    @pytest.mark.parametrize("coverage", [0.8, 0.9, 0.95, 0.99])
    def test_various_coverage_levels(self, coverage):
        cp = ConformalPredictor(coverage=coverage, min_samples=5)
        scores = [0.7 + i * 0.01 for i in range(20)]
        cp.calibrate(scores, [False] * 20)
        pi = cp.predict(0.5)
        assert pi.coverage == coverage
        assert 0.0 <= pi.lower <= pi.upper <= 1.0

    @pytest.mark.parametrize("n_samples", [10, 50, 100])
    def test_calibration_sizes(self, n_samples):
        cp = ConformalPredictor(min_samples=5)
        scores = [0.5 + i * 0.001 for i in range(n_samples)]
        cp.calibrate(scores, [i % 2 == 0 for i in range(n_samples)])
        pi = cp.predict(0.5)
        assert pi.calibration_size == n_samples


class TestConformalPerformanceDoc:
    """Document conformal prediction pipeline performance."""

    def test_prediction_fast(self):
        import time

        cp = ConformalPredictor(min_samples=5)
        cp.calibrate([0.5 + i * 0.01 for i in range(50)], [False] * 50)
        t0 = time.perf_counter()
        for _ in range(1000):
            cp.predict(0.5)
        per_call_us = (time.perf_counter() - t0) / 1000 * 1_000_000
        assert per_call_us < 100, f"Predict took {per_call_us:.1f}µs"

    def test_interval_has_all_fields(self):
        cp = ConformalPredictor(min_samples=5)
        cp.calibrate([0.7] * 10, [False] * 10)
        pi = cp.predict(0.5)
        for field in [
            "point_estimate",
            "lower",
            "upper",
            "coverage",
            "calibration_size",
            "is_reliable",
        ]:
            assert hasattr(pi, field), f"Missing: {field}"
