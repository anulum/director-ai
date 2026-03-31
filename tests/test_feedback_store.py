# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for feedback store (online calibration pipeline).

Covers: empty store, report/retrieve, multiple reports, domain filtering,
limit, disagreement detection, training data export, parametrised
domains/limits, pipeline integration with calibrator, and performance.
"""

from __future__ import annotations

import pytest

from director_ai.core.calibration.feedback_store import FeedbackStore


class TestFeedbackStoreBasic:
    def test_empty_store(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        assert store.count() == 0
        assert store.get_corrections() == []
        store.close()

    def test_report_and_retrieve(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        store.report("What is X?", "X is Y.", True, False, 0.7)
        assert store.count() == 1
        corrections = store.get_corrections()
        assert len(corrections) == 1
        assert corrections[0].prompt == "What is X?"
        assert corrections[0].guardrail_approved is True
        assert corrections[0].human_approved is False
        assert corrections[0].guardrail_score == 0.7
        store.close()

    def test_multiple_reports(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(10):
            store.report(f"q{i}", f"a{i}", i % 2 == 0, i % 3 == 0, i / 10)
        assert store.count() == 10
        store.close()

    def test_domain_filter(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        store.report("q1", "a1", True, True, domain="medical")
        store.report("q2", "a2", True, False, domain="finance")
        store.report("q3", "a3", False, False, domain="medical")
        assert store.count(domain="medical") == 2
        assert store.count(domain="finance") == 1
        medical = store.get_corrections(domain="medical")
        assert len(medical) == 2
        store.close()

    def test_limit(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(20):
            store.report(f"q{i}", f"a{i}", True, True)
        assert len(store.get_corrections(limit=5)) == 5
        store.close()


class TestDisagreements:
    def test_get_disagreements(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        store.report("q1", "a1", True, True)  # agree
        store.report("q2", "a2", True, False)  # disagree (FP)
        store.report("q3", "a3", False, True)  # disagree (FN)
        store.report("q4", "a4", False, False)  # agree
        disagreements = store.get_disagreements()
        assert len(disagreements) == 2
        store.close()

    def test_disagreements_limit(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        for i in range(10):
            store.report(f"q{i}", f"a{i}", True, False)
        assert len(store.get_disagreements(limit=3)) == 3
        store.close()


class TestExport:
    def test_export_training_data(self, tmp_path):
        store = FeedbackStore(tmp_path / "test.db")
        store.report("q1", "a1", True, True, domain="med")
        store.report("q2", "a2", True, False, domain="fin")
        data = store.export_training_data()
        assert len(data) == 2
        assert data[0]["label"] in (0, 1)
        assert "prompt" in data[0]
        assert "response" in data[0]
        assert "domain" in data[0]
        store.close()


class TestFeedbackStoreParametrised:
    """Parametrised feedback store tests."""

    @pytest.mark.parametrize("n_reports", [1, 5, 10, 50])
    def test_various_report_counts(self, tmp_path, n_reports):
        store = FeedbackStore(tmp_path / f"test_{n_reports}.db")
        for i in range(n_reports):
            store.report(f"q{i}", f"a{i}", True, True)
        assert store.count() == n_reports
        store.close()

    @pytest.mark.parametrize("domain", ["medical", "finance", "legal", ""])
    def test_various_domains(self, tmp_path, domain):
        store = FeedbackStore(tmp_path / f"test_{domain}.db")
        store.report("q", "a", True, True, domain=domain)
        corrections = store.get_corrections(domain=domain if domain else None)
        assert len(corrections) >= 1
        store.close()

    @pytest.mark.parametrize("limit", [1, 3, 5, 10])
    def test_various_limits(self, tmp_path, limit):
        store = FeedbackStore(tmp_path / f"test_lim_{limit}.db")
        for i in range(20):
            store.report(f"q{i}", f"a{i}", True, True)
        result = store.get_corrections(limit=limit)
        assert len(result) == limit
        store.close()


class TestFeedbackStorePerformanceDoc:
    """Document feedback store pipeline performance."""

    def test_report_fast(self, tmp_path):
        import time

        store = FeedbackStore(tmp_path / "perf.db")
        t0 = time.perf_counter()
        for i in range(100):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        per_call_ms = (time.perf_counter() - t0) / 100 * 1000
        assert per_call_ms < 50, f"report() took {per_call_ms:.1f}ms/call"
        store.close()

    def test_store_integrates_with_calibrator(self, tmp_path):
        from director_ai.core.calibration.online_calibrator import OnlineCalibrator

        store = FeedbackStore(tmp_path / "cal.db")
        for i in range(25):
            store.report(f"q{i}", f"a{i}", True, True, guardrail_score=0.8)
        cal = OnlineCalibrator(store, min_corrections=20)
        report = cal.calibrate()
        assert report.correction_count == 25
        store.close()

    def test_export_has_required_fields(self, tmp_path):
        store = FeedbackStore(tmp_path / "export.db")
        store.report("q", "a", True, False, domain="test")
        data = store.export_training_data()
        assert len(data) == 1
        for field in ["prompt", "response", "label", "domain"]:
            assert field in data[0], f"Missing: {field}"
        store.close()
