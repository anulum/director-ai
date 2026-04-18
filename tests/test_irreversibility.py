# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — irreversibility forecaster tests

"""Multi-angle coverage: ReversibilityScore validation,
RuleReversibility marker logic, custom estimator via Protocol,
IrreversibilityForecaster seeded determinism, Wilson interval
bounds, parameter validation, Beasley-Springer quantile
accuracy."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pytest

from director_ai.core.irreversibility import (
    Forecast,
    IrreversibilityForecaster,
    ReversibilityScore,
    RuleReversibility,
)
from director_ai.core.irreversibility.forecaster import (
    _standard_normal_quantile,
    _wilson_score,
)

# --- ReversibilityScore ---------------------------------------------


class TestReversibilityScore:
    def test_valid(self):
        s = ReversibilityScore(score=0.5, reason="ok")
        assert s.score == 0.5

    def test_score_out_of_range(self):
        with pytest.raises(ValueError, match="score"):
            ReversibilityScore(score=1.5, reason="bad")

    def test_empty_reason_rejected(self):
        with pytest.raises(ValueError, match="reason"):
            ReversibilityScore(score=0.5, reason="")


# --- RuleReversibility ---------------------------------------------


class TestRuleReversibility:
    def test_irreversible_marker_scores_low(self):
        est = RuleReversibility()
        s = est.score("DELETE FROM users WHERE id = 1")
        assert s.score < 0.2

    def test_reversible_marker_scores_high(self):
        est = RuleReversibility()
        s = est.score("preview the change in staging")
        assert s.score > 0.8

    def test_no_markers_returns_baseline(self):
        est = RuleReversibility(baseline=0.42)
        s = est.score("adjust the thermostat slightly")
        assert s.score == 0.42

    def test_conflicting_markers_return_baseline(self):
        est = RuleReversibility()
        s = est.score("preview and then delete the draft")
        assert "both" in s.reason

    def test_empty_action_returns_baseline(self):
        est = RuleReversibility(baseline=0.3)
        s = est.score("")
        assert s.score == 0.3

    def test_invalid_baseline_rejected(self):
        with pytest.raises(ValueError, match="baseline"):
            RuleReversibility(baseline=1.5)

    def test_context_accepted(self):
        est = RuleReversibility()
        s = est.score("preview change", context={"tenant_id": "t1"})
        assert s.score > 0.5


# --- IrreversibilityForecaster --------------------------------------


class _AlwaysIrreversible:
    def score(
        self, action: str, *, context: Mapping[str, object] | None = None
    ) -> ReversibilityScore:
        return ReversibilityScore(score=0.01, reason="stub irreversible")


class _AlwaysReversible:
    def score(
        self, action: str, *, context: Mapping[str, object] | None = None
    ) -> ReversibilityScore:
        return ReversibilityScore(score=0.99, reason="stub reversible")


class TestForecaster:
    def test_irreversible_sequence_crosses_often(self):
        f = IrreversibilityForecaster(
            estimator=_AlwaysIrreversible(), n_samples=512, threshold=0.5
        )
        forecast = f.forecast(["delete everything", "drop table"], seed=42)
        assert forecast.p_irreversible > 0.9
        assert forecast.ci_low > 0.8

    def test_reversible_sequence_rarely_crosses(self):
        f = IrreversibilityForecaster(
            estimator=_AlwaysReversible(), n_samples=512, threshold=0.5
        )
        forecast = f.forecast(["preview", "stage", "dry-run"], seed=42)
        assert forecast.p_irreversible < 0.1

    def test_seed_is_deterministic(self):
        f = IrreversibilityForecaster(estimator=_AlwaysIrreversible(), n_samples=128)
        a = f.forecast(["delete"], seed=7)
        b = f.forecast(["delete"], seed=7)
        assert a == b

    def test_different_seeds_diverge(self):
        # Threshold of 0.5 and per-action score of 0.3 puts the
        # cumulative around the boundary so two different seeds
        # produce different crossing counts.
        class _Mid:
            def score(
                self, action: str, *, context: Mapping[str, object] | None = None
            ) -> ReversibilityScore:
                return ReversibilityScore(score=0.3, reason="mid")

        f = IrreversibilityForecaster(estimator=_Mid(), n_samples=64, threshold=0.5)
        a = f.forecast(["a"], seed=1)
        b = f.forecast(["a"], seed=2)
        assert a.crossed != b.crossed or a.p_irreversible != b.p_irreversible

    def test_empty_action_list_rejected(self):
        f = IrreversibilityForecaster()
        with pytest.raises(ValueError, match="actions"):
            f.forecast([])

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="threshold"):
            IrreversibilityForecaster(threshold=1.5)

    def test_invalid_samples(self):
        with pytest.raises(ValueError, match="n_samples"):
            IrreversibilityForecaster(n_samples=0)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            IrreversibilityForecaster(confidence=2.0)

    def test_forecast_structure(self):
        f = IrreversibilityForecaster(estimator=_AlwaysIrreversible())
        forecast = f.forecast(["delete"], seed=0)
        assert isinstance(forecast, Forecast)
        assert (
            0.0 <= forecast.ci_low <= forecast.p_irreversible <= forecast.ci_high <= 1.0
        )
        assert forecast.crossed <= forecast.samples

    def test_default_estimator_runs(self):
        """No explicit estimator — forecaster constructs a default."""
        f = IrreversibilityForecaster(n_samples=64)
        forecast = f.forecast(["delete all staging data"], seed=0)
        assert isinstance(forecast, Forecast)


# --- Internal helpers -----------------------------------------------


class TestWilsonScore:
    def test_bounds_contain_phat(self):
        low, high = _wilson_score(0.5, n=100, confidence=0.95)
        assert low < 0.5 < high

    def test_zero_samples(self):
        assert _wilson_score(0.0, n=0, confidence=0.95) == (0.0, 0.0)

    def test_edge_probabilities(self):
        low, high = _wilson_score(0.0, n=100, confidence=0.95)
        assert low == 0.0 and high < 0.05
        low, high = _wilson_score(1.0, n=100, confidence=0.95)
        assert high == 1.0 and low > 0.95


class TestStandardNormalQuantile:
    def test_symmetry_around_median(self):
        q = _standard_normal_quantile(0.5)
        assert math.isclose(q, 0.0, abs_tol=1e-3)

    def test_known_quantile_95(self):
        # z_{0.975} ≈ 1.9600; the Beasley-Springer approximation
        # should be within 1e-3.
        q = _standard_normal_quantile(0.975)
        assert math.isclose(q, 1.9600, abs_tol=1e-3)

    def test_known_quantile_99(self):
        q = _standard_normal_quantile(0.995)
        assert math.isclose(q, 2.5758, abs_tol=1e-3)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            _standard_normal_quantile(0.0)
        with pytest.raises(ValueError):
            _standard_normal_quantile(1.0)
