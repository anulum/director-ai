# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime self-protection tests

"""Multi-angle tests for behavioural runtime self-protection (RASP).

Covers the streaming robust detector (cold start, MAD scoring, the MAD-zero
mean-absolute-deviation fallback, the truly-constant baseline, validation) and
the monitor (severity bands ok/watch/alert, per-metric detectors, recent-anomaly
tracking, under_attack, tenant-safe serialisation, ProductionGuard wiring).
"""

from __future__ import annotations

import math

import pytest

from director_ai.core.rasp import (
    AnomalyScore,
    RuntimeSelfProtection,
    StreamingRobustDetector,
)


class TestDetectorValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"window_size": 1}, "window_size"),
            ({"min_samples": 0}, "min_samples"),
            ({"min_samples": 999}, "min_samples"),
            ({"z_threshold": 0}, "z_threshold"),
        ],
    )
    def test_invalid_params_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            StreamingRobustDetector(**kwargs)

    def test_z_threshold_exposed(self):
        assert StreamingRobustDetector(z_threshold=4.0).z_threshold == 4.0


class TestDetectorScoring:
    def test_cold_start_records_without_judging(self):
        d = StreamingRobustDetector(min_samples=5)
        scores = [d.update(10.0) for _ in range(5)]
        assert all(s.cold_start for s in scores)
        assert all(not s.anomalous for s in scores)
        assert all(s.robust_z == 0.0 for s in scores)

    def test_in_range_value_not_flagged(self):
        d = StreamingRobustDetector(min_samples=5, z_threshold=3.5)
        for v in (10.0, 11.0, 9.0, 10.0, 10.0):
            d.update(v)
        # 11 is inside the observed range; the MAD-zero mean-AD fallback keeps it
        # from being scored as an infinite outlier.
        score = d.update(11.0)
        assert score.anomalous is False
        assert math.isfinite(score.robust_z)

    def test_spike_flagged(self):
        d = StreamingRobustDetector(min_samples=5, z_threshold=3.5)
        for v in (10.0, 11.0, 9.0, 10.0, 10.0, 11.0):
            d.update(v)
        score = d.update(500.0)
        assert score.anomalous is True
        assert score.robust_z > 3.5

    def test_constant_baseline_deviation_is_infinite(self):
        d = StreamingRobustDetector(min_samples=4)
        for _ in range(4):
            d.update(5.0)
        assert d.update(5.0).robust_z == 0.0  # exact match
        assert d.update(6.0).robust_z == math.inf  # any deviation
        assert d.update(6.0).anomalous is True

    def test_mad_positive_path(self):
        d = StreamingRobustDetector(min_samples=8, z_threshold=3.5)
        for v in (10, 10, 10, 10, 12, 12, 12, 12):  # median 11, MAD 1
            d.update(v)
        score = d.update(11.0)
        assert score.robust_z == pytest.approx(0.0, abs=1e-9)

    def test_score_is_frozen_dataclass(self):
        score = AnomalyScore(value=1.0, robust_z=0.0, anomalous=False, cold_start=True)
        with pytest.raises(AttributeError):
            score.value = 2.0  # type: ignore[misc]


class TestMonitorValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"watch_fraction": 0.0}, "watch_fraction"),
            ({"watch_fraction": 1.0}, "watch_fraction"),
            ({"recent_window": 0}, "recent_window"),
        ],
    )
    def test_invalid_params_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            RuntimeSelfProtection(**kwargs)

    def test_empty_metric_rejected(self):
        with pytest.raises(ValueError, match="metric name"):
            RuntimeSelfProtection().observe("  ", 1.0)

    def test_under_attack_min_anomalies_validated(self):
        with pytest.raises(ValueError, match="min_anomalies"):
            RuntimeSelfProtection().under_attack(min_anomalies=0)


class TestMonitorSeverity:
    def _primed(self) -> RuntimeSelfProtection:
        rasp = RuntimeSelfProtection(min_samples=8, z_threshold=3.5, watch_fraction=0.7)
        for v in (10, 10, 10, 10, 12, 12, 12, 12):  # median 11, MAD 1, scale ~1.4826
            rasp.observe("rate", v)
        return rasp

    def test_cold_start_is_ok(self):
        rasp = RuntimeSelfProtection(min_samples=8)
        verdict = rasp.observe("rate", 10.0)
        assert verdict.cold_start is True
        assert verdict.severity == "ok"

    def test_ok_band(self):
        verdict = self._primed().observe("rate", 11.0)  # z ~ 0
        assert verdict.severity == "ok"
        assert verdict.anomalous is False

    def test_watch_band(self):
        # z ~ 5 / 1.4826 = 3.37 -> above 0.7*3.5=2.45, below 3.5.
        verdict = self._primed().observe("rate", 16.0)
        assert verdict.severity == "watch"
        assert verdict.anomalous is False

    def test_alert_band(self):
        verdict = self._primed().observe("rate", 100.0)
        assert verdict.severity == "alert"
        assert verdict.anomalous is True


class TestMonitorState:
    def test_per_metric_detectors(self):
        rasp = RuntimeSelfProtection()
        rasp.observe("rate", 1.0)
        rasp.observe("payload_size", 1.0)
        rasp.observe("rate", 2.0)
        assert rasp.tracked_metrics == ("payload_size", "rate")

    def test_recent_anomaly_count_and_under_attack(self):
        rasp = RuntimeSelfProtection(min_samples=4, z_threshold=3.5)
        for v in (10.0, 10.0, 10.0, 10.0):
            rasp.observe("rate", v)
        # Escalating spikes each stay an outlier against the adapting baseline
        # (a same-valued sustained flood would be absorbed into the window).
        for spike in (1e3, 1e5, 1e7):
            rasp.observe("rate", spike)
        assert rasp.recent_anomaly_count >= 3
        assert rasp.under_attack(min_anomalies=3) is True

    def test_not_under_attack_when_calm(self):
        rasp = RuntimeSelfProtection(min_samples=4)
        for v in (10.0, 10.1, 9.9, 10.0, 10.05):
            rasp.observe("rate", v)
        assert rasp.under_attack() is False

    def test_verdict_to_dict_tenant_safe(self):
        verdict = RuntimeSelfProtection().observe("rate", 5.0)
        assert set(verdict.to_dict()) == {
            "metric",
            "value",
            "robust_z",
            "severity",
            "anomalous",
            "cold_start",
        }


class TestGuardWiring:
    def test_production_guard_exposes_rasp(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        rasp = guard.rasp
        assert isinstance(rasp, RuntimeSelfProtection)
        assert guard.rasp is rasp  # cached
        assert rasp.observe("rate", 1.0).cold_start is True
