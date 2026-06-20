# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``director_ai.core.retrieval.kb_health``.

Covers health check execution, issue detection, latency measurement,
report structure, and edge cases.
"""

from __future__ import annotations

from director_ai.core.retrieval import kb_health
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

    def test_direct_count_store_is_supported(self):
        class CountStore:
            def count(self):
                return 4

            def retrieve_context(self, _query):
                return ["indexed policy"]

        report = KBHealthCheck(CountStore(), min_documents=4).run()

        assert report.healthy is True
        assert report.document_count == 4
        assert report.total_entries == 4

    def test_backend_count_store_is_supported(self):
        class Backend:
            def count(self):
                return 6

        class BackendStore:
            backend = Backend()

            def retrieve_context(self, _query):
                return ["indexed policy"]

        report = KBHealthCheck(BackendStore(), min_documents=6).run()

        assert report.healthy is True
        assert report.document_count == 6
        assert report.total_entries == 6


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

    def test_count_probe_failure_degrades_to_empty_issue(self):
        class BrokenCountStore:
            def count(self):
                raise RuntimeError("backend count unavailable")

            def retrieve_context(self, _query):
                return ["still queryable"]

        report = KBHealthCheck(BrokenCountStore(), min_documents=1).run()

        assert report.healthy is False
        assert report.document_count == 0
        assert any("below minimum" in issue for issue in report.issues)

    def test_query_exceptions_are_reported_without_crashing(self):
        class BrokenQueryStore:
            facts = {"policy": "queryable metadata exists"}

            def retrieve_context(self, _query):
                raise RuntimeError("query backend unavailable")

        report = KBHealthCheck(BrokenQueryStore(), min_documents=1).run()

        assert report.healthy is False
        assert any("Store is not queryable" in issue for issue in report.issues)
        assert report.avg_query_latency_ms == float("inf")

    def test_empty_entries_emit_warning_but_do_not_make_store_unhealthy(self):
        class EmptyEntryStore:
            facts = {"blank": "  ", "short": "ok", "valid": "indexed fact"}

            def retrieve_context(self, _query):
                return ["indexed fact"]

        report = KBHealthCheck(EmptyEntryStore(), min_documents=1).run()

        assert report.healthy is True
        assert any(
            "empty or very short entries" in warning for warning in report.warnings
        )


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

    def test_empty_probe_query_list_reports_zero_latency(self):
        store = _make_store(5)
        check = KBHealthCheck(store)
        check._probe_queries = []

        assert check._measure_query_latency() == 0.0

    def test_mean_latency_falls_back_when_accelerator_fails(self, monkeypatch):
        monkeypatch.setattr(
            kb_health,
            "rust_mean",
            lambda _values: (_ for _ in ()).throw(RuntimeError("accelerator offline")),
        )

        assert kb_health._mean_float([1.0, 2.0, 3.0]) == 2.0


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
