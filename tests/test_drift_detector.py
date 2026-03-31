# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for statistical drift detection pipeline (STRONG).

Covers: z-score computation, severity classification, baseline tracking,
window analysis, pipeline integration with compliance, and performance.
"""

from __future__ import annotations

from director_ai.compliance.audit_log import AuditEntry, AuditLog
from director_ai.compliance.drift_detector import DriftDetector, _norm_cdf

_BASE = 1_700_000_000.0  # fixed epoch for deterministic tests


def _entry(approved=True, score=0.8, timestamp=None) -> AuditEntry:
    return AuditEntry(
        prompt="q",
        response="a",
        model="gpt-4o",
        provider="openai",
        score=score,
        approved=approved,
        verdict_confidence=0.9,
        task_type="qa",
        domain="",
        latency_ms=15.0,
        timestamp=timestamp if timestamp is not None else _BASE,
    )


class TestNormCdf:
    def test_symmetry(self):
        assert abs(_norm_cdf(0.0) - 0.5) < 1e-6

    def test_tails(self):
        assert _norm_cdf(-10.0) == 0.0
        assert _norm_cdf(10.0) == 1.0

    def test_known_value(self):
        # P(Z < 1.96) ≈ 0.975
        assert abs(_norm_cdf(1.96) - 0.975) < 0.001


class TestDriftDetectorEmpty:
    def test_empty_log(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        d = DriftDetector(log, window_days=7)
        result = d.analyze()
        assert result.detected is False
        assert result.severity == "none"
        assert result.p_value == 1.0
        log.close()


class TestDriftDetectorSingleWindow:
    def test_all_in_one_window(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        since = _BASE
        until = _BASE + 7 * 86400
        # All 10 entries within day 1-2 of a single 7-day window
        for i in range(10):
            log.log(_entry(timestamp=since + 86400 + i * 3600))
        d = DriftDetector(log, window_days=7)
        result = d.analyze(since=since, until=until)
        assert result.detected is False
        assert len(result.windows) == 1
        log.close()


class TestDriftDetectorNoDrift:
    def test_stable_rate(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        since = _BASE
        until = _BASE + 28 * 86400
        # Week 1 (day 0-6): 10% rejection in 100 entries
        for i in range(100):
            approved = i % 10 != 0
            log.log(_entry(approved=approved, timestamp=since + i * 6000))
        # Week 4 (day 21-27): 10% rejection in 100 entries
        for i in range(100):
            approved = i % 10 != 0
            log.log(_entry(approved=approved, timestamp=since + 21 * 86400 + i * 6000))
        d = DriftDetector(log, window_days=7)
        result = d.analyze(since=since, until=until)
        assert result.detected is False
        assert result.severity == "none"
        log.close()


class TestDriftDetectorDrift:
    def test_severe_drift(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        since = _BASE
        until = _BASE + 28 * 86400
        # Week 1 (day 0-6): 5% rejection (1 out of 20)
        for i in range(20):
            approved = i != 0
            log.log(_entry(approved=approved, timestamp=since + i * 3600))
        # Week 4 (day 21-27): 80% rejection (16 out of 20)
        for i in range(20):
            approved = i < 4
            log.log(
                _entry(
                    approved=approved,
                    score=0.3 if not approved else 0.8,
                    timestamp=since + 21 * 86400 + i * 3600,
                )
            )
        d = DriftDetector(log, window_days=7, alpha=0.05)
        result = d.analyze(since=since, until=until)
        assert result.detected is True
        assert result.severity == "severe"
        assert result.rate_change > 0.5
        assert result.z_score > 0
        assert result.p_value < 0.05
        log.close()

    def test_moderate_drift(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        since = _BASE
        until = _BASE + 28 * 86400
        # Week 1 (day 0-6): 5% rejection (5 out of 100)
        for i in range(100):
            approved = i >= 5
            log.log(_entry(approved=approved, timestamp=since + i * 600))
        # Week 4 (day 21-27): 15% rejection (15 out of 100)
        for i in range(100):
            approved = i >= 15
            log.log(_entry(approved=approved, timestamp=since + 21 * 86400 + i * 600))
        d = DriftDetector(log, window_days=7, alpha=0.05)
        result = d.analyze(since=since, until=until)
        assert result.detected is True
        assert result.severity == "moderate"
        assert 0.05 <= result.rate_change <= 0.15
        log.close()


class TestDriftDetectorZTest:
    def test_z_test_equal_rates(self):
        z, p = DriftDetector._two_proportion_z(10, 100, 10, 100)
        assert abs(z) < 0.001
        assert p > 0.4

    def test_z_test_different_rates(self):
        z, p = DriftDetector._two_proportion_z(5, 100, 50, 100)
        assert z > 2.0
        assert p < 0.01

    def test_z_test_zero_samples(self):
        z, p = DriftDetector._two_proportion_z(0, 0, 5, 100)
        assert z == 0.0
        assert p == 1.0

    def test_z_test_all_zero_rejections(self):
        z, p = DriftDetector._two_proportion_z(0, 100, 0, 100)
        assert z == 0.0
        assert p == 1.0


class TestDriftDetectorClassify:
    def test_not_detected(self):
        assert DriftDetector._classify(0.2, False, 0.5) == "none"

    def test_mild(self):
        assert DriftDetector._classify(0.03, True, 0.01) == "mild"

    def test_moderate(self):
        assert DriftDetector._classify(0.10, True, 0.01) == "moderate"

    def test_severe(self):
        assert DriftDetector._classify(0.20, True, 0.001) == "severe"
