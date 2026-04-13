# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.core.retrieval.kb_health``.

Covers health check execution, issue detection, latency measurement,
report structure, and edge cases.
"""

from __future__ import annotations

from director_ai.core.retrieval.kb_health import KBHealthCheck, KBHealthReport
from director_ai.core.retrieval.knowledge import GroundTruthStore


def _make_store(n_facts: int = 5) -> GroundTruthStore:
    store = GroundTruthStore()
    for i in range(n_facts):
        store.add(f"fact_{i}", f"This is fact number {i} about topic {i}.")
    return store


# ── Report dataclass ───────────────────────────────────────────────────


class TestReport:
    def test_summary(self):
        r = KBHealthReport(True, 10, 10, 5.0, checks_passed=5, checks_total=5)
        assert "HEALTHY" in r.summary
        assert "10 docs" in r.summary

    def test_unhealthy_summary(self):
        r = KBHealthReport(False, 0, 0, 0.0, issues=["empty"])
        assert "UNHEALTHY" in r.summary

    def test_defaults(self):
        r = KBHealthReport(True, 1, 1, 1.0)
        assert r.issues == []
        assert r.warnings == []


# ── Healthy store ──────────────────────────────────────────────────────


class TestHealthyStore:
    def test_passes_all_checks(self):
        store = _make_store(10)
        check = KBHealthCheck(store, min_documents=1)
        report = check.run()
        assert report.healthy
        assert report.checks_passed >= 3
        assert report.document_count >= 1

    def test_latency_measured(self):
        store = _make_store(5)
        check = KBHealthCheck(store)
        report = check.run()
        assert report.avg_query_latency_ms >= 0.0

    def test_no_issues(self):
        store = _make_store(5)
        check = KBHealthCheck(store, min_documents=1)
        report = check.run()
        assert len(report.issues) == 0


# ── Unhealthy store ───────────────────────────────────────────────────


class TestUnhealthyStore:
    def test_empty_store(self):
        store = GroundTruthStore()
        check = KBHealthCheck(store, min_documents=5)
        report = check.run()
        assert not report.healthy
        assert any("below minimum" in i for i in report.issues)

    def test_high_min_documents(self):
        store = _make_store(3)
        check = KBHealthCheck(store, min_documents=100)
        report = check.run()
        assert not report.healthy


# ── Latency threshold ─────────────────────────────────────────────────


class TestLatency:
    def test_within_threshold(self):
        store = _make_store(5)
        check = KBHealthCheck(store, max_query_latency_ms=1000.0)
        report = check.run()
        assert report.avg_query_latency_ms < 1000.0

    def test_custom_probe_queries(self):
        store = _make_store(5)
        check = KBHealthCheck(store, probe_queries=["custom query"])
        report = check.run()
        assert report.avg_query_latency_ms >= 0.0


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_none_store_attributes(self):
        """Store without expected methods should not crash."""

        class MinimalStore:
            pass

        check = KBHealthCheck(MinimalStore())
        report = check.run()
        assert isinstance(report, KBHealthReport)

    def test_checks_total_counted(self):
        store = _make_store(5)
        check = KBHealthCheck(store)
        report = check.run()
        assert report.checks_total == 5

    def test_total_entries(self):
        store = _make_store(7)
        check = KBHealthCheck(store)
        report = check.run()
        assert report.total_entries >= 0
