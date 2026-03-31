# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for feedback store."""

from __future__ import annotations

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
