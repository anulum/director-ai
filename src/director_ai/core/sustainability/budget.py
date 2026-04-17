# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SustainabilityBudget

"""Compose the quota + forecaster + carbon tracker into one
allow/block decision.

Three branches fire:

* **Daily quota** — reject when the tenant has hit the per-day
  ceiling.
* **Forecast headroom** — reject when adding the requested
  amount to the running day-total would cross the forecaster's
  conformal upper bound. Prevents quiet exhaustion before the
  explicit daily limit hits.
* **Carbon throttle** — degrade or reject when the current grid
  intensity is above the throttle threshold. Callers can pass
  ``allow_during_high_carbon=True`` for traffic that cannot be
  deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .carbon import CarbonIntensityTracker
from .forecaster import ConformalDemandForecaster, PredictionInterval
from .quota import ComputeQuota, QuotaError

ThrottleReason = Literal[
    "allowed", "quota_exceeded", "forecast_headroom", "high_carbon"
]


@dataclass(frozen=True)
class BudgetVerdict:
    """Outcome of one :meth:`SustainabilityBudget.allow` call."""

    allowed: bool
    reason: ThrottleReason
    remaining_today: float
    forecast_upper: float
    carbon_intensity: float
    detail: str = ""


class SustainabilityBudget:
    """Compose the three signals into one decision.

    Parameters
    ----------
    quota :
        The per-tenant :class:`ComputeQuota`.
    forecaster :
        A :class:`ConformalDemandForecaster`. The forecaster is
        updated from outside (via its :meth:`observe` method) so
        the budget does not take on the responsibility of
        measuring day-end demand.
    carbon :
        The :class:`CarbonIntensityTracker`.
    carbon_throttle_intensity :
        Above this gCO₂/kWh value the budget considers the grid
        "high carbon" and blocks deferrable traffic. Default 400.
    coverage :
        Conformal coverage target for the forecaster interval.
        Default 0.9.
    """

    def __init__(
        self,
        *,
        quota: ComputeQuota,
        forecaster: ConformalDemandForecaster,
        carbon: CarbonIntensityTracker,
        carbon_throttle_intensity: float = 400.0,
        coverage: float = 0.9,
    ) -> None:
        if carbon_throttle_intensity < 0:
            raise ValueError(
                "carbon_throttle_intensity must be non-negative"
            )
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0, 1)")
        self._quota = quota
        self._forecaster = forecaster
        self._carbon = carbon
        self._carbon_threshold = carbon_throttle_intensity
        self._coverage = coverage

    def allow(
        self,
        *,
        tenant_id: str,
        amount: float,
        allow_during_high_carbon: bool = False,
    ) -> BudgetVerdict:
        """Return a :class:`BudgetVerdict` for the request."""
        if not tenant_id:
            raise ValueError("tenant_id must be non-empty")
        if amount <= 0:
            raise ValueError("amount must be positive")
        remaining = self._quota.remaining_today(tenant_id)
        carbon_now = self._carbon.current()
        forecast = self._resolve_forecast()
        if amount > remaining:
            return BudgetVerdict(
                allowed=False,
                reason="quota_exceeded",
                remaining_today=remaining,
                forecast_upper=forecast.upper,
                carbon_intensity=carbon_now,
                detail=(
                    f"amount {amount:.2f} > remaining_today "
                    f"{remaining:.2f}"
                ),
            )
        projected_today = self._projected_today_usage(tenant_id, amount)
        if projected_today > forecast.upper > 0:
            return BudgetVerdict(
                allowed=False,
                reason="forecast_headroom",
                remaining_today=remaining,
                forecast_upper=forecast.upper,
                carbon_intensity=carbon_now,
                detail=(
                    f"projected {projected_today:.2f} > "
                    f"forecast upper {forecast.upper:.2f}"
                ),
            )
        if (
            carbon_now > self._carbon_threshold
            and not allow_during_high_carbon
        ):
            return BudgetVerdict(
                allowed=False,
                reason="high_carbon",
                remaining_today=remaining,
                forecast_upper=forecast.upper,
                carbon_intensity=carbon_now,
                detail=(
                    f"carbon {carbon_now:.1f} > threshold "
                    f"{self._carbon_threshold:.1f}"
                ),
            )
        try:
            self._quota.consume(tenant_id=tenant_id, amount=amount)
        except QuotaError as exc:
            return BudgetVerdict(
                allowed=False,
                reason="quota_exceeded",
                remaining_today=remaining,
                forecast_upper=forecast.upper,
                carbon_intensity=carbon_now,
                detail=str(exc),
            )
        return BudgetVerdict(
            allowed=True,
            reason="allowed",
            remaining_today=max(0.0, remaining - amount),
            forecast_upper=forecast.upper,
            carbon_intensity=carbon_now,
        )

    def _projected_today_usage(
        self, tenant_id: str, amount: float
    ) -> float:
        used_today = self._quota.daily_limit - self._quota.remaining_today(
            tenant_id
        )
        return used_today + amount

    def _resolve_forecast(self) -> PredictionInterval:
        try:
            return self._forecaster.predict(coverage=self._coverage)
        except ValueError:
            return PredictionInterval(
                point=0.0,
                lower=0.0,
                upper=0.0,
                coverage=self._coverage,
                residual_sample_size=0,
            )
