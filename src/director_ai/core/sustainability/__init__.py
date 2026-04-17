# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — long-horizon sustainability budget

"""Multi-day compute quota with conformal demand forecasting and
carbon-aware throttling.

* :class:`ComputeQuota` — per-tenant multi-day budget. Day
  boundaries come from a caller-supplied clock so the quota
  works in any timezone.
* :class:`CarbonIntensityTracker` — rolling window of
  grid-carbon-intensity observations (gCO₂/kWh) supplied by the
  deployment's data provider. Exposes current intensity + a
  clean percentile query for relative ranking.
* :class:`ConformalDemandForecaster` — inductive conformal
  prediction on a per-day demand time series. Keeps a recent
  history of residuals between predicted and observed demand,
  and returns a prediction interval at the caller's coverage
  target.
* :class:`SustainabilityBudget` — the orchestrator. Composes
  the quota + forecaster + carbon tracker into one
  :meth:`allow` entry point that returns a
  :class:`BudgetVerdict`. Blocks when the projected day-end
  demand would exceed the quota, throttles when the carbon
  intensity is above the operator's high-carbon threshold.
"""

from .budget import BudgetVerdict, SustainabilityBudget, ThrottleReason
from .carbon import CarbonIntensityTracker, CarbonReading
from .forecaster import ConformalDemandForecaster, PredictionInterval
from .quota import ComputeQuota, DailyUsage, QuotaError

__all__ = [
    "BudgetVerdict",
    "CarbonIntensityTracker",
    "CarbonReading",
    "ComputeQuota",
    "ConformalDemandForecaster",
    "DailyUsage",
    "PredictionInterval",
    "QuotaError",
    "SustainabilityBudget",
    "ThrottleReason",
]
