# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for online calibrator."""

from __future__ import annotations

from director_ai.core.calibration.feedback_store import FeedbackStore
from director_ai.core.calibration.online_calibrator import OnlineCalibrator


class TestCalibrationReport:
    def test_empty_store(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        cal = OnlineCalibrator(store)
        report = cal.calibrate()
        assert report.correction_count == 0
        assert report.optimal_threshold is None
        store.close()

    def test_perfect_guardrail(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(30):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.correction_count == 30
        assert report.current_accuracy == 1.0
        assert report.fpr == 0.0
        assert report.fnr == 0.0
        assert report.tpr == 1.0
        store.close()

    def test_all_false_positives(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(30):
            store.report(f"q{i}", f"a{i}", True, False, guardrail_score=0.6)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.fpr == 1.0
        assert report.tpr == 0.0
        store.close()

    def test_mixed_errors(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        # 10 true positives
        for i in range(10):
            store.report(f"tp{i}", f"atp{i}", True, True, guardrail_score=0.8)
        # 5 false positives
        for i in range(5):
            store.report(f"fp{i}", f"afp{i}", True, False, guardrail_score=0.55)
        # 10 true negatives
        for i in range(10):
            store.report(f"tn{i}", f"atn{i}", False, False, guardrail_score=0.3)
        # 5 false negatives
        for i in range(5):
            store.report(f"fn{i}", f"afn{i}", False, True, guardrail_score=0.45)

        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.correction_count == 30
        assert 0.6 < report.current_accuracy < 0.7
        assert report.fpr > 0
        assert report.fnr > 0
        assert report.fpr_ci > 0
        assert report.fnr_ci > 0
        store.close()


class TestThresholdSweep:
    def test_optimal_threshold_separable(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        # Good responses score high, bad responses score low
        for i in range(25):
            store.report(
                f"good{i}", f"a{i}", True, True, guardrail_score=0.8 + i * 0.005
            )
        for i in range(25):
            store.report(
                f"bad{i}", f"b{i}", False, False, guardrail_score=0.2 + i * 0.005
            )

        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.optimal_threshold is not None
        assert 0.3 < report.optimal_threshold < 0.8
        store.close()

    def test_insufficient_corrections(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(5):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.optimal_threshold is None
        store.close()


class TestDomainFilter:
    def test_calibrate_by_domain(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(25):
            store.report(
                f"med{i}", f"a{i}", True, True, guardrail_score=0.8, domain="medical"
            )
        for i in range(25):
            store.report(
                f"fin{i}", f"b{i}", True, False, guardrail_score=0.6, domain="finance"
            )

        cal = OnlineCalibrator(store, min_corrections=20)
        med_report = cal.calibrate(domain="medical")
        fin_report = cal.calibrate(domain="finance")
        assert med_report.current_accuracy == 1.0
        assert fin_report.fpr == 1.0
        store.close()
