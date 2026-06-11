# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — carbon tests

from __future__ import annotations

import pytest

from director_ai.core.sustainability.carbon import (
    CarbonIntensityTracker,
    CarbonReading,
    _sum_float,
    _sum_int,
)


def test_carbon_reading_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp must be non-negative"):
        CarbonReading(timestamp=-0.1, intensity=1.0)


def test_carbon_reading_rejects_negative_intensity() -> None:
    with pytest.raises(ValueError, match="intensity must be non-negative"):
        CarbonReading(timestamp=0.0, intensity=-1.0)


def test_carbon_tracker_rejects_invalid_init_args() -> None:
    with pytest.raises(ValueError, match="window_size must be positive"):
        CarbonIntensityTracker(window_size=0)
    with pytest.raises(ValueError, match="fallback_intensity must be non-negative"):
        CarbonIntensityTracker(fallback_intensity=-1.0)


def test_carbon_tracker_current_and_window_when_empty() -> None:
    tracker = CarbonIntensityTracker(window_size=3, fallback_intensity=42.0)
    assert tracker.current() == 42.0
    assert tracker.window() == ()


def test_carbon_tracker_record_and_window_rolls() -> None:
    tracker = CarbonIntensityTracker(window_size=2)
    tracker.record(CarbonReading(timestamp=1.0, intensity=100.0))
    tracker.record(CarbonReading(timestamp=2.0, intensity=200.0))
    tracker.record(CarbonReading(timestamp=3.0, intensity=300.0))

    readings = tracker.window()
    assert len(readings) == 2
    assert readings[0].intensity == 200.0
    assert readings[1].intensity == 300.0
    assert tracker.current() == 300.0


def test_carbon_tracker_record_many_uses_insertion_order() -> None:
    tracker = CarbonIntensityTracker(window_size=4)
    tracker.record_many(
        [
            CarbonReading(timestamp=1.0, intensity=1.0),
            CarbonReading(timestamp=2.0, intensity=2.0),
        ]
    )

    assert tuple(reading.intensity for reading in tracker.window()) == (1.0, 2.0)


def test_carbon_tracker_percentile_and_mean_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("director_ai.core.sustainability.carbon._RUST_CARBON", False)

    tracker = CarbonIntensityTracker(window_size=4)
    tracker.record_many(
        [
            CarbonReading(timestamp=1.0, intensity=10.0),
            CarbonReading(timestamp=2.0, intensity=20.0),
            CarbonReading(timestamp=3.0, intensity=30.0),
        ]
    )

    assert tracker.percentile(15.0) == pytest.approx(1 / 3)
    assert tracker.percentile(25.0) == pytest.approx(2 / 3)
    assert tracker.percentile(100.0) == 1.0
    assert tracker.mean() == pytest.approx(20.0)
    assert tracker.percentile(20.0) == pytest.approx(2 / 3)


def test_carbon_tracker_percentile_uses_rust_path_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[float], float]] = []

    def fake_percentile(values: list[float], value: float) -> float:
        calls.append((values, value))
        return 0.75

    monkeypatch.setattr("director_ai.core.sustainability.carbon._RUST_CARBON", True)
    monkeypatch.setattr(
        "director_ai.core.sustainability.carbon.rust_percentile_rank",
        fake_percentile,
    )
    tracker = CarbonIntensityTracker(window_size=2)
    tracker.record_many(
        [
            CarbonReading(timestamp=1.0, intensity=100.0),
            CarbonReading(timestamp=2.0, intensity=200.0),
        ]
    )

    assert tracker.percentile(150.0) == pytest.approx(0.75)
    assert calls == [([100.0, 200.0], 150.0)]


def test_carbon_tracker_mean_uses_rust_path_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[float]] = []

    def fake_mean(values: list[float]) -> float:
        calls.append(values)
        return 151.5

    monkeypatch.setattr("director_ai.core.sustainability.carbon._RUST_CARBON", True)
    monkeypatch.setattr("director_ai.core.sustainability.carbon.rust_mean", fake_mean)
    tracker = CarbonIntensityTracker(window_size=2)
    tracker.record_many(
        [
            CarbonReading(timestamp=1.0, intensity=100.0),
            CarbonReading(timestamp=2.0, intensity=200.0),
        ]
    )

    assert tracker.mean() == pytest.approx(151.5)
    assert calls == [[100.0, 200.0]]


def test_carbon_tracker_percentile_empty_returns_fallback() -> None:
    tracker = CarbonIntensityTracker(window_size=2)
    assert tracker.percentile(99.0) == 1.0


def test_carbon_tracker_mean_empty_uses_fallback() -> None:
    tracker = CarbonIntensityTracker(window_size=2, fallback_intensity=77.0)
    assert tracker.mean() == 77.0


def test_sum_int_uses_rust_path_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_sum(values: list[int]) -> int:
        calls.append(len(values))
        return 123

    monkeypatch.setattr("director_ai.core.sustainability.carbon._RUST_CARBON", True)
    monkeypatch.setattr("director_ai.core.sustainability.carbon.rust_sum_i64", fake_sum)
    assert _sum_int([1, 2, 3]) == 123
    assert calls == [3]


def test_sum_float_uses_rust_path_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_sum(values: list[float]) -> float:
        calls.append(len(values))
        return 6.0

    monkeypatch.setattr("director_ai.core.sustainability.carbon._RUST_CARBON", True)
    monkeypatch.setattr("director_ai.core.sustainability.carbon.rust_sum_f64", fake_sum)
    assert _sum_float([1.0, 2.0, 3.0]) == pytest.approx(6.0)
    assert calls == [3]
