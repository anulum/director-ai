# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — sustainability budget tests

"""Multi-angle coverage: ComputeQuota daily ceiling + rolling
window, CarbonIntensityTracker readings + percentile,
ConformalDemandForecaster EMA + residual-based interval,
SustainabilityBudget allow/block flow across all three
throttle branches."""

from __future__ import annotations

import pytest

from director_ai.core.sustainability import (
    BudgetVerdict,
    CarbonIntensityTracker,
    CarbonReading,
    ComputeQuota,
    ConformalDemandForecaster,
    DailyUsage,
    PredictionInterval,
    QuotaError,
    SustainabilityBudget,
)


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


# --- ComputeQuota --------------------------------------------------


class TestComputeQuota:
    def _quota(self, limit: float = 100.0, clock: _FakeClock | None = None) -> ComputeQuota:
        return ComputeQuota(
            daily_limit=limit, window_days=7, clock=clock or _FakeClock()
        )

    def test_initial_remaining_is_limit(self):
        quota = self._quota(limit=50.0)
        assert quota.remaining_today("t1") == 50.0

    def test_consume_decreases_remaining(self):
        quota = self._quota(limit=50.0)
        quota.consume(tenant_id="t1", amount=10.0)
        assert quota.remaining_today("t1") == 40.0

    def test_daily_ceiling_enforced(self):
        quota = self._quota(limit=10.0)
        quota.consume(tenant_id="t1", amount=10.0)
        with pytest.raises(QuotaError, match="daily limit"):
            quota.consume(tenant_id="t1", amount=1.0)

    def test_per_tenant_independent(self):
        quota = self._quota(limit=10.0)
        quota.consume(tenant_id="t1", amount=10.0)
        # t2 still has full budget.
        assert quota.remaining_today("t2") == 10.0

    def test_day_resets_on_new_day(self):
        clock = _FakeClock(start=0.0)
        quota = self._quota(limit=10.0, clock=clock)
        quota.consume(tenant_id="t1", amount=10.0)
        clock.now = 86_400.0  # next day
        assert quota.remaining_today("t1") == 10.0

    def test_window_trims_old_days(self):
        clock = _FakeClock(start=0.0)
        quota = ComputeQuota(
            daily_limit=10.0, window_days=3, clock=clock
        )
        for day in range(5):
            clock.now = float(day * 86_400)
            quota.consume(tenant_id="t1", amount=1.0)
        clock.now = 5 * 86_400.0
        history = quota.usage_history("t1")
        # Only the last 3 days should remain.
        assert len(history) <= 3

    def test_usage_history_aggregates_same_day(self):
        clock = _FakeClock(start=0.0)
        quota = self._quota(clock=clock)
        quota.consume(tenant_id="t1", amount=3.0)
        quota.consume(tenant_id="t1", amount=2.0)
        history = quota.usage_history("t1")
        assert len(history) == 1
        assert history[0].consumed == 5.0

    def test_reset_single_tenant(self):
        quota = self._quota(limit=10.0)
        quota.consume(tenant_id="t1", amount=10.0)
        quota.reset("t1")
        assert quota.remaining_today("t1") == 10.0

    def test_reset_all(self):
        quota = self._quota(limit=10.0)
        quota.consume(tenant_id="t1", amount=5.0)
        quota.consume(tenant_id="t2", amount=5.0)
        quota.reset()
        assert quota.remaining_today("t1") == 10.0
        assert quota.remaining_today("t2") == 10.0

    def test_consume_rejects_empty_tenant(self):
        quota = self._quota()
        with pytest.raises(QuotaError, match="tenant_id"):
            quota.consume(tenant_id="", amount=1.0)

    def test_consume_rejects_non_positive_amount(self):
        quota = self._quota()
        with pytest.raises(QuotaError, match="amount"):
            quota.consume(tenant_id="t", amount=0.0)

    def test_usage_history_empty_tenant(self):
        quota = self._quota()
        with pytest.raises(QuotaError, match="tenant_id"):
            quota.usage_history("")

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"daily_limit": 0.0}, "daily_limit"),
            ({"daily_limit": 10.0, "window_days": 0}, "window_days"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(QuotaError, match=match):
            ComputeQuota(**kwargs)

    def test_daily_usage_validation(self):
        with pytest.raises(QuotaError, match="tenant_id"):
            DailyUsage(tenant_id="", day=0, consumed=1.0)
        with pytest.raises(QuotaError, match="consumed"):
            DailyUsage(tenant_id="t", day=0, consumed=-1.0)


# --- CarbonIntensityTracker ---------------------------------------


class TestCarbonTracker:
    def test_fallback_when_empty(self):
        tracker = CarbonIntensityTracker(fallback_intensity=600.0)
        assert tracker.current() == 600.0

    def test_records_latest(self):
        tracker = CarbonIntensityTracker()
        tracker.record(CarbonReading(timestamp=0.0, intensity=100.0))
        tracker.record(CarbonReading(timestamp=1.0, intensity=200.0))
        assert tracker.current() == 200.0

    def test_record_many(self):
        tracker = CarbonIntensityTracker()
        tracker.record_many(
            [
                CarbonReading(timestamp=0.0, intensity=50.0),
                CarbonReading(timestamp=1.0, intensity=150.0),
            ]
        )
        assert tracker.current() == 150.0

    def test_percentile(self):
        tracker = CarbonIntensityTracker()
        for i in range(10):
            tracker.record(
                CarbonReading(timestamp=float(i), intensity=float(i * 10))
            )
        # 40 is above 0, 10, 20, 30, 40 (5 readings) out of 10 → 0.5.
        assert tracker.percentile(40.0) == 0.5

    def test_percentile_empty_returns_one(self):
        tracker = CarbonIntensityTracker()
        assert tracker.percentile(100.0) == 1.0

    def test_mean(self):
        tracker = CarbonIntensityTracker()
        tracker.record_many(
            [
                CarbonReading(timestamp=0.0, intensity=100.0),
                CarbonReading(timestamp=1.0, intensity=200.0),
            ]
        )
        assert tracker.mean() == 150.0

    def test_window_eviction(self):
        tracker = CarbonIntensityTracker(window_size=3)
        for i in range(5):
            tracker.record(
                CarbonReading(timestamp=float(i), intensity=float(i))
            )
        assert len(tracker.window()) == 3

    def test_reading_validation(self):
        with pytest.raises(ValueError, match="timestamp"):
            CarbonReading(timestamp=-1.0, intensity=0.0)
        with pytest.raises(ValueError, match="intensity"):
            CarbonReading(timestamp=0.0, intensity=-1.0)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"window_size": 0}, "window_size"),
            ({"fallback_intensity": -1.0}, "fallback_intensity"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(ValueError, match=match):
            CarbonIntensityTracker(**kwargs)


# --- ConformalDemandForecaster -------------------------------------


class TestForecaster:
    def test_first_observation_sets_ema(self):
        fc = ConformalDemandForecaster(alpha=0.5)
        fc.observe(100.0)
        assert fc.last_forecast == 100.0

    def test_predict_without_observation_raises(self):
        fc = ConformalDemandForecaster()
        with pytest.raises(ValueError, match="no observations"):
            fc.predict()

    def test_interval_collapses_before_min_samples(self):
        fc = ConformalDemandForecaster(min_samples=5)
        for value in (10.0, 12.0, 11.0):
            fc.observe(value)
        interval = fc.predict()
        assert interval.lower == interval.upper == interval.point

    def test_interval_widens_after_samples(self):
        fc = ConformalDemandForecaster(alpha=0.3, min_samples=5)
        for value in (10.0, 12.0, 9.0, 11.0, 10.5, 13.0, 8.0, 12.5):
            fc.observe(value)
        interval = fc.predict(coverage=0.9)
        assert isinstance(interval, PredictionInterval)
        assert interval.width > 0
        assert interval.lower <= interval.point <= interval.upper

    def test_coverage_validation(self):
        fc = ConformalDemandForecaster()
        fc.observe(1.0)
        with pytest.raises(ValueError, match="coverage"):
            fc.predict(coverage=0.0)

    def test_negative_demand_rejected(self):
        fc = ConformalDemandForecaster()
        with pytest.raises(ValueError, match="demand"):
            fc.observe(-1.0)

    def test_reset_clears_state(self):
        fc = ConformalDemandForecaster()
        fc.observe(10.0)
        fc.reset()
        with pytest.raises(ValueError):
            fc.predict()

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"alpha": 0.0}, "alpha"),
            ({"alpha": 1.5}, "alpha"),
            ({"residual_window": 0}, "residual_window"),
            ({"min_samples": 0}, "min_samples"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        with pytest.raises(ValueError, match=match):
            ConformalDemandForecaster(**kwargs)


# --- SustainabilityBudget ------------------------------------------


class TestBudget:
    def _budget(
        self,
        *,
        daily_limit: float = 100.0,
        carbon_threshold: float = 400.0,
    ) -> tuple[SustainabilityBudget, ComputeQuota, ConformalDemandForecaster, CarbonIntensityTracker]:
        clock = _FakeClock()
        quota = ComputeQuota(daily_limit=daily_limit, clock=clock)
        fc = ConformalDemandForecaster(alpha=0.5, min_samples=2)
        carbon = CarbonIntensityTracker(
            fallback_intensity=100.0, clock=clock
        )
        budget = SustainabilityBudget(
            quota=quota,
            forecaster=fc,
            carbon=carbon,
            carbon_throttle_intensity=carbon_threshold,
        )
        return budget, quota, fc, carbon

    def test_allow_happy_path(self):
        budget, *_ = self._budget()
        verdict = budget.allow(tenant_id="t1", amount=10.0)
        assert isinstance(verdict, BudgetVerdict)
        assert verdict.allowed
        assert verdict.reason == "allowed"
        assert verdict.remaining_today == 90.0

    def test_blocks_when_quota_exceeded(self):
        budget, *_ = self._budget(daily_limit=10.0)
        budget.allow(tenant_id="t1", amount=10.0)
        verdict = budget.allow(tenant_id="t1", amount=1.0)
        assert not verdict.allowed
        assert verdict.reason == "quota_exceeded"

    def test_blocks_on_high_carbon(self):
        budget, _, _, carbon = self._budget(carbon_threshold=200.0)
        carbon.record(CarbonReading(timestamp=0.0, intensity=500.0))
        verdict = budget.allow(tenant_id="t1", amount=1.0)
        assert not verdict.allowed
        assert verdict.reason == "high_carbon"
        assert "500" in verdict.detail

    def test_allow_during_high_carbon_override(self):
        budget, _, _, carbon = self._budget(carbon_threshold=200.0)
        carbon.record(CarbonReading(timestamp=0.0, intensity=500.0))
        verdict = budget.allow(
            tenant_id="t1", amount=1.0, allow_during_high_carbon=True
        )
        assert verdict.allowed

    def test_forecast_headroom_blocks(self):
        budget, quota, fc, _ = self._budget(daily_limit=100.0)
        # Prime the forecaster: historical days consumed 30 / 30.
        fc.observe(30.0)
        fc.observe(30.0)
        fc.observe(30.0)
        # Use up 40 today → projected = 40 + 50 = 90 vs forecaster
        # conformal upper, which is around 30 + residual 0 = 30 (since
        # EMA has stabilised at 30 and residuals are 0). Request
        # should blow past the forecast upper.
        quota.consume(tenant_id="t1", amount=40.0)
        verdict = budget.allow(tenant_id="t1", amount=50.0)
        assert not verdict.allowed
        assert verdict.reason == "forecast_headroom"

    def test_allows_even_without_forecast(self):
        """When the forecaster has not observed anything the budget
        treats the upper bound as 0 (no constraint) so the
        quota / carbon branches govern."""
        budget, *_ = self._budget()
        verdict = budget.allow(tenant_id="t1", amount=1.0)
        assert verdict.allowed

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"carbon_throttle_intensity": -1.0}, "carbon_throttle"),
            ({"coverage": 1.5}, "coverage"),
        ],
    )
    def test_constructor_validation(self, kwargs: dict, match: str):
        clock = _FakeClock()
        quota = ComputeQuota(daily_limit=10.0, clock=clock)
        fc = ConformalDemandForecaster()
        carbon = CarbonIntensityTracker(clock=clock)
        with pytest.raises(ValueError, match=match):
            SustainabilityBudget(
                quota=quota, forecaster=fc, carbon=carbon, **kwargs
            )

    def test_empty_tenant_rejected(self):
        budget, *_ = self._budget()
        with pytest.raises(ValueError, match="tenant_id"):
            budget.allow(tenant_id="", amount=1.0)

    def test_non_positive_amount(self):
        budget, *_ = self._budget()
        with pytest.raises(ValueError, match="amount"):
            budget.allow(tenant_id="t", amount=0.0)
