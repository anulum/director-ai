# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Sustainability Forecaster Tests
"""Module-specific tests for conformal demand forecasting."""

from __future__ import annotations

import pytest

import director_ai.core.sustainability.forecaster as forecaster_mod
from director_ai.core.sustainability import (
    ConformalDemandForecaster,
    PredictionInterval,
)


def test_python_ema_path_sets_first_observation_and_updates(monkeypatch) -> None:
    monkeypatch.setattr(forecaster_mod, "_RUST_FORECAST", False)
    forecaster = ConformalDemandForecaster(alpha=0.25, min_samples=2)

    forecaster.observe(100.0)
    assert forecaster.last_forecast == pytest.approx(100.0)

    forecaster.observe(140.0)
    assert forecaster.last_forecast == pytest.approx(110.0)


def test_python_quantile_path_selects_conformal_width(monkeypatch) -> None:
    monkeypatch.setattr(forecaster_mod, "_RUST_FORECAST", False)
    forecaster = ConformalDemandForecaster(alpha=0.5, min_samples=3)
    for demand in (10.0, 20.0, 10.0, 20.0, 10.0):
        forecaster.observe(demand)

    interval = forecaster.predict(coverage=0.75)

    assert isinstance(interval, PredictionInterval)
    assert interval.point == pytest.approx(13.125)
    assert interval.residual_sample_size == 4
    assert interval.width == pytest.approx(15.0)
    assert interval.lower == pytest.approx(5.625)
    assert interval.upper == pytest.approx(20.625)


def test_python_quantile_lower_bound_is_clamped_to_zero(monkeypatch) -> None:
    monkeypatch.setattr(forecaster_mod, "_RUST_FORECAST", False)
    forecaster = ConformalDemandForecaster(alpha=1.0, min_samples=2)
    for demand in (1.0, 10.0, 1.0):
        forecaster.observe(demand)

    interval = forecaster.predict(coverage=0.9)

    assert interval.point == pytest.approx(1.0)
    assert interval.lower == 0.0
    assert interval.upper == pytest.approx(10.0)
