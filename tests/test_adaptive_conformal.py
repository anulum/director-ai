# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Gibbs-Candès adaptive conformal inference.

Covers gamma/coverage construction guards, the ACI update direction (covering
widens alpha, missing shrinks it), clamping to the unit interval, the
effective-coverage reporting, reset, predict before/at the boundaries and after
calibration, and a drift scenario where sustained misses drive coverage back up
(the covariate-shift correction the static predictor cannot make).
"""

from __future__ import annotations

import pytest

from director_ai.core.calibration.adaptive_conformal import (
    AdaptiveConformalPredictor,
)


class TestConstruction:
    def test_starts_at_target_alpha(self):
        p = AdaptiveConformalPredictor(coverage=0.9)
        assert p.current_alpha == pytest.approx(0.1)
        assert p.effective_coverage == pytest.approx(0.9)
        assert p.gamma == 0.05

    @pytest.mark.parametrize("gamma", [0.0, -0.1, 1.5])
    def test_invalid_gamma(self, gamma):
        with pytest.raises(ValueError, match="gamma"):
            AdaptiveConformalPredictor(gamma=gamma)

    def test_inherits_coverage_guard(self):
        with pytest.raises(ValueError, match="coverage"):
            AdaptiveConformalPredictor(coverage=1.5)


class TestUpdateDynamics:
    def test_covering_widens_alpha(self):
        # err=0 -> alpha rises toward 1 (narrower intervals).
        p = AdaptiveConformalPredictor(coverage=0.9, gamma=0.1)
        a0 = p.current_alpha
        p.update(covered=True)
        assert p.current_alpha > a0
        assert p.current_alpha == pytest.approx(0.1 + 0.1 * 0.1)

    def test_missing_shrinks_alpha(self):
        # err=1 -> alpha falls toward 0 (wider intervals, more coverage).
        p = AdaptiveConformalPredictor(coverage=0.9, gamma=0.1)
        a0 = p.current_alpha
        p.update(covered=False)
        assert p.current_alpha < a0
        assert p.current_alpha == pytest.approx(0.1 + 0.1 * (0.1 - 1.0))

    def test_alpha_clamps_to_unit_interval(self):
        p = AdaptiveConformalPredictor(coverage=0.9, gamma=1.0)
        for _ in range(50):
            p.update(covered=True)
        assert p.current_alpha == 1.0
        assert p.effective_coverage == 0.0
        for _ in range(50):
            p.update(covered=False)
        assert p.current_alpha == 0.0
        assert p.effective_coverage == 1.0

    def test_reset_adaptation(self):
        p = AdaptiveConformalPredictor(coverage=0.9, gamma=0.2)
        for _ in range(5):
            p.update(covered=False)
        assert p.current_alpha < 0.1
        p.reset_adaptation()
        assert p.current_alpha == pytest.approx(0.1)


class TestPredict:
    def _calibrate(self, p):
        # Spread of coherence scores with mixed hallucination labels.
        scores = [0.1 * (i % 10) for i in range(60)]
        labels = [i % 3 == 0 for i in range(60)]
        p.calibrate(scores, labels)

    def test_predict_without_calibration_is_unreliable(self):
        p = AdaptiveConformalPredictor()
        interval = p.predict(0.5)
        assert interval.calibration_size == 0
        assert interval.is_reliable is False
        assert (interval.lower, interval.upper) == (0.0, 1.0)

    def test_predict_reports_effective_coverage(self):
        p = AdaptiveConformalPredictor(coverage=0.9, gamma=0.1)
        self._calibrate(p)
        p.update(covered=True)  # alpha rises, coverage drops below 0.9
        interval = p.predict(0.5)
        assert interval.coverage == pytest.approx(p.effective_coverage)
        assert interval.coverage < 0.9
        assert interval.is_reliable is True

    def test_misses_widen_the_interval(self):
        # Sustained misses drop alpha -> coverage rises -> half-width grows.
        p = AdaptiveConformalPredictor(coverage=0.8, gamma=0.2)
        self._calibrate(p)
        narrow = p.predict(0.5)
        for _ in range(10):
            p.update(covered=False)
        wide = p.predict(0.5)
        assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower)
        assert wide.coverage > narrow.coverage

    def test_degenerate_zero_coverage_gives_point_interval(self):
        p = AdaptiveConformalPredictor(coverage=0.9, gamma=1.0)
        self._calibrate(p)
        for _ in range(50):
            p.update(covered=True)  # drive alpha to 1 -> eff coverage 0
        interval = p.predict(0.5)
        assert interval.coverage == 0.0
        assert (
            interval.lower == interval.upper == pytest.approx(interval.point_estimate)
        )

    def test_full_coverage_uses_max_residual(self):
        p = AdaptiveConformalPredictor(coverage=0.9, gamma=1.0)
        self._calibrate(p)
        for _ in range(50):
            p.update(covered=False)  # drive alpha to 0 -> eff coverage 1
        interval = p.predict(0.5)
        assert interval.coverage == 1.0
        # widest interval available from the calibration residuals
        assert interval.upper - interval.lower > 0.0


class TestStaticBaselineUnaffected:
    def test_base_predictor_quantile_unchanged(self):
        # The refactor must not change the static predictor's behaviour.
        from director_ai.core.calibration.conformal import ConformalPredictor

        scores = [0.1 * (i % 10) for i in range(60)]
        labels = [i % 3 == 0 for i in range(60)]
        base = ConformalPredictor(coverage=0.9)
        base.calibrate(scores, labels)
        adaptive = AdaptiveConformalPredictor(coverage=0.9)
        adaptive.calibrate(scores, labels)
        # Before any update, adaptive predicts at the same coverage as static.
        assert adaptive.predict(0.5).coverage == pytest.approx(0.9)
        assert adaptive.predict(0.5).lower == pytest.approx(base.predict(0.5).lower)
