# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the empirical-miscoverage drift monitor.

Covers the Hoeffding bound (monotonicity, confidence widening, zero-sample
ceiling, clamping), per-slice accounting over the rolling window, the
fail-closed verdicts (insufficient samples, bound over alpha, unseen slice,
empty monitor), the aggregate health report, construction guards, the dataclass
serialisers, reset, and a covariate-shift drift scenario where a slice that
starts within target later trips the RED verdict.
"""

from __future__ import annotations

import logging
import math

import pytest

from director_ai.core.calibration.miscoverage import (
    CoverageHealthReport,
    MiscoverageMonitor,
    SegmentCoverageHealth,
    hoeffding_upper_bound,
)


class TestHoeffdingBound:
    def test_zero_samples_returns_ceiling(self):
        assert hoeffding_upper_bound(0, 0, 0.05) == 1.0

    def test_matches_closed_form(self):
        # 5 misses in 100 trials, delta=0.05.
        expected = 0.05 + math.sqrt(math.log(1.0 / 0.05) / (2.0 * 100))
        assert hoeffding_upper_bound(5, 100, 0.05) == pytest.approx(expected)

    def test_clamped_to_one(self):
        # Tiny n with a high miss count + wide slack saturates at 1.0.
        assert hoeffding_upper_bound(1, 1, 0.01) == 1.0

    def test_more_samples_tighten_bound(self):
        # Same point estimate, more evidence -> the bound shrinks toward p_hat.
        loose = hoeffding_upper_bound(5, 50, 0.05)
        tight = hoeffding_upper_bound(50, 500, 0.05)
        assert tight < loose

    def test_higher_confidence_widens_bound(self):
        # Smaller delta (more confidence) -> larger slack -> larger bound.
        lo_conf = hoeffding_upper_bound(10, 200, 0.10)
        hi_conf = hoeffding_upper_bound(10, 200, 0.01)
        assert hi_conf > lo_conf


class TestConstructionGuards:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_alpha(self, alpha):
        with pytest.raises(ValueError, match="target_alpha"):
            MiscoverageMonitor(alpha)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window"):
            MiscoverageMonitor(0.1, window=0)

    def test_invalid_min_samples(self):
        with pytest.raises(ValueError, match="min_samples must be >= 1"):
            MiscoverageMonitor(0.1, min_samples=0)

    def test_min_samples_exceeds_window(self):
        with pytest.raises(ValueError, match="must be <="):
            MiscoverageMonitor(0.1, window=10, min_samples=20)

    @pytest.mark.parametrize("conf", [0.0, 1.0, 2.0])
    def test_invalid_confidence(self, conf):
        with pytest.raises(ValueError, match="confidence"):
            MiscoverageMonitor(0.1, confidence=conf)

    def test_target_alpha_property(self):
        assert MiscoverageMonitor(0.2).target_alpha == 0.2


class TestObservationAccounting:
    def test_empty_slice_reports_none(self):
        mon = MiscoverageMonitor(0.1)
        assert mon.miscoverage("qa") is None
        assert mon.miscoverage_upper_bound("qa") is None

    def test_blank_segment_rejected(self):
        mon = MiscoverageMonitor(0.1)
        with pytest.raises(ValueError, match="non-empty"):
            mon.observe("   ", covered=True)

    def test_segment_is_stripped(self):
        mon = MiscoverageMonitor(0.1)
        mon.observe("  qa  ", covered=True)
        assert mon.miscoverage("qa") == 0.0

    def test_point_miscoverage(self):
        mon = MiscoverageMonitor(0.1, min_samples=1)
        for _ in range(8):
            mon.observe("qa", covered=True)
        for _ in range(2):
            mon.observe("qa", covered=False)
        assert mon.miscoverage("qa") == pytest.approx(0.2)

    def test_window_evicts_oldest(self):
        # Window of 4: the two early misses roll out, leaving all-covered.
        mon = MiscoverageMonitor(0.1, window=4, min_samples=1)
        mon.observe("qa", covered=False)
        mon.observe("qa", covered=False)
        for _ in range(4):
            mon.observe("qa", covered=True)
        assert mon.miscoverage("qa") == 0.0
        assert mon.miscoverage_upper_bound("qa") is not None


class TestFailClosedVerdicts:
    def test_insufficient_samples_fails_closed(self):
        mon = MiscoverageMonitor(0.1, min_samples=30)
        for _ in range(10):
            mon.observe("qa", covered=True)  # perfect, but too few
        health = mon.segment_health("qa")
        assert health.healthy is False
        assert health.reason == "insufficient_samples"

    def test_within_target_is_healthy(self):
        mon = MiscoverageMonitor(0.2, min_samples=30, window=500)
        # 1000 observations, ~2% miss rate -> bound well under alpha=0.2.
        for i in range(1000):
            mon.observe("qa", covered=(i % 50 != 0))
        health = mon.segment_health("qa")
        assert health.healthy is True
        assert health.reason == "within_target"
        assert health.upper_bound <= 0.2

    def test_drift_trips_red(self):
        mon = MiscoverageMonitor(0.05, min_samples=30, window=200)
        # 30% realised miss rate, far above alpha=0.05.
        for i in range(200):
            mon.observe("qa", covered=(i % 10 >= 3))
        health = mon.segment_health("qa")
        assert health.healthy is False
        assert health.reason == "miscoverage_exceeds_alpha"
        assert health.upper_bound > 0.05

    def test_unseen_segment_is_unhealthy(self):
        mon = MiscoverageMonitor(0.1)
        assert mon.is_healthy("never-seen") is False


class TestAggregateHealth:
    def test_empty_monitor_is_red(self):
        report = MiscoverageMonitor(0.1).health()
        assert report.healthy is False
        assert report.segments == ()
        assert report.unhealthy_segments == ()

    def test_all_healthy_is_green(self):
        mon = MiscoverageMonitor(0.2, min_samples=30, window=500)
        for seg in ("qa", "rag"):
            for i in range(500):
                mon.observe(seg, covered=(i % 100 != 0))
        report = mon.health()
        assert report.healthy is True
        assert report.unhealthy_segments == ()
        assert [s.segment for s in report.segments] == ["qa", "rag"]

    def test_one_bad_slice_reds_overall(self, caplog):
        # alpha=0.2 so the low-miss slice can actually certify at n=200 (the
        # Hoeffding bound is conservative: tighter alpha needs far more samples).
        mon = MiscoverageMonitor(0.2, min_samples=30, window=200)
        for i in range(200):
            mon.observe("good", covered=(i % 100 != 0))  # ~1% miss
        for i in range(200):
            mon.observe("bad", covered=(i % 2 == 0))  # ~50% miss
        with caplog.at_level(logging.WARNING):
            report = mon.health()
        assert report.healthy is False
        assert report.unhealthy_segments == ("bad",)
        assert "coverage health RED" in caplog.text

    def test_empty_monitor_logs_no_slices(self, caplog):
        with caplog.at_level(logging.WARNING):
            MiscoverageMonitor(0.1).health()
        assert "no observed slices" in caplog.text


class TestSerialisationAndReset:
    def test_segment_health_to_dict(self):
        health = SegmentCoverageHealth(
            segment="qa",
            samples=100,
            miscoverage=0.123456,
            upper_bound=0.234567,
            target_alpha=0.2,
            healthy=True,
            reason="within_target",
        )
        d = health.to_dict()
        assert d["segment"] == "qa"
        assert d["miscoverage"] == 0.1235
        assert d["upper_bound"] == 0.2346
        assert d["healthy"] is True

    def test_report_to_dict_and_snapshot(self):
        mon = MiscoverageMonitor(0.2, min_samples=1)
        mon.observe("qa", covered=True)
        snap = mon.snapshot()
        assert isinstance(snap, dict)
        assert "segments" in snap
        report = mon.health()
        assert isinstance(report, CoverageHealthReport)
        assert report.to_dict()["segments"][0]["segment"] == "qa"

    def test_reset_single_segment(self):
        mon = MiscoverageMonitor(0.1, min_samples=1)
        mon.observe("qa", covered=True)
        mon.observe("rag", covered=True)
        mon.reset("qa")
        assert mon.miscoverage("qa") is None
        assert mon.miscoverage("rag") == 0.0

    def test_reset_all(self):
        mon = MiscoverageMonitor(0.1, min_samples=1)
        mon.observe("qa", covered=True)
        mon.reset()
        assert mon.miscoverage("qa") is None


class TestCovariateShiftScenario:
    def test_slice_starts_green_then_drifts_red(self):
        # The exact failure the monitor exists to catch: a slice calibrated fine,
        # then covariate shift pushes the realised miss rate past alpha while a
        # static quantile would still look confident.
        # alpha=0.2: 100 clean samples certify (Hoeffding bound ~0.12 <= 0.2),
        # then the window fully turns over to misses and the bound blows past 0.2.
        mon = MiscoverageMonitor(0.2, min_samples=30, window=100)
        for _ in range(100):  # clean regime: 0% miss
            mon.observe("qa", covered=True)
        assert mon.is_healthy("qa") is True
        for _ in range(100):  # shifted regime: 100% miss, window fully turns over
            mon.observe("qa", covered=False)
        health = mon.segment_health("qa")
        assert health.healthy is False
        assert health.reason == "miscoverage_exceeds_alpha"
