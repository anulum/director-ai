# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for EU AI Act compliance reporter."""

from __future__ import annotations

import time

from director_ai.compliance.audit_log import AuditEntry, AuditLog
from director_ai.compliance.reporter import ComplianceReporter


def _entry(
    model="gpt-4o",
    score=0.8,
    approved=True,
    confidence=0.9,
    latency=15.0,
    domain="",
    ts_offset=0,
    human_override=None,
) -> AuditEntry:
    return AuditEntry(
        prompt="q",
        response="a",
        model=model,
        provider="openai",
        score=score,
        approved=approved,
        verdict_confidence=confidence,
        task_type="qa",
        domain=domain,
        latency_ms=latency,
        timestamp=time.time() + ts_offset,
        human_override=human_override,
    )


class TestReporterEmpty:
    def test_empty_log(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        reporter = ComplianceReporter(log)
        report = reporter.generate_report()
        assert report.total_interactions == 0
        assert report.overall_hallucination_rate == 0.0
        log.close()


class TestReporterMetrics:
    def test_all_approved(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        for _ in range(20):
            log.log(_entry(approved=True, score=0.85))
        reporter = ComplianceReporter(log)
        report = reporter.generate_report()
        assert report.total_interactions == 20
        assert report.overall_hallucination_rate == 0.0
        assert report.incident_count == 0
        log.close()

    def test_mixed_results(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        for _ in range(15):
            log.log(_entry(approved=True, score=0.8))
        for _ in range(5):
            log.log(_entry(approved=False, score=0.3))
        reporter = ComplianceReporter(log)
        report = reporter.generate_report()
        assert report.total_interactions == 20
        assert report.overall_hallucination_rate == 0.25
        assert report.incident_count == 5
        assert report.overall_hallucination_rate_ci > 0
        log.close()

    def test_human_overrides(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        for _ in range(10):
            log.log(_entry())
        for _ in range(3):
            log.log(_entry(human_override=True))
        reporter = ComplianceReporter(log)
        report = reporter.generate_report()
        assert report.human_override_count == 3
        assert abs(report.human_override_rate - 3 / 13) < 0.01
        log.close()


class TestReporterPerModel:
    def test_multiple_models(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        for _ in range(10):
            log.log(_entry(model="gpt-4o", approved=True))
        for _ in range(10):
            log.log(_entry(model="claude-4", approved=False, score=0.2))
        reporter = ComplianceReporter(log)
        report = reporter.generate_report()
        assert len(report.model_metrics) == 2
        gpt = next(m for m in report.model_metrics if m.model == "gpt-4o")
        claude = next(m for m in report.model_metrics if m.model == "claude-4")
        assert gpt.hallucination_rate == 0.0
        assert claude.hallucination_rate == 1.0
        log.close()


class TestReporterDrift:
    def test_no_drift(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        now = time.time()
        for i in range(30):
            log.log(_entry(approved=True, ts_offset=-i * 86400))
        reporter = ComplianceReporter(log, drift_window_days=7)
        report = reporter.generate_report(since=now - 30 * 86400)
        assert report.drift_detected is False
        assert report.drift_severity == 0.0
        log.close()

    def test_drift_detected(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        now = time.time()
        # Week 1: all approved
        for i in range(7):
            log.log(_entry(approved=True, ts_offset=-(28 - i) * 86400))
        # Week 4: mostly rejected
        for i in range(7):
            log.log(_entry(approved=False, score=0.2, ts_offset=-i * 86400))
        reporter = ComplianceReporter(log, drift_window_days=7, drift_threshold=0.05)
        report = reporter.generate_report(since=now - 30 * 86400)
        assert report.drift_detected is True
        assert report.drift_severity > 0.05
        log.close()


class TestReporterMarkdown:
    def test_markdown_output(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        for _ in range(10):
            log.log(_entry(approved=True))
        for _ in range(2):
            log.log(_entry(approved=False, score=0.3))
        reporter = ComplianceReporter(log)
        report = reporter.generate_report()
        md = report.to_markdown()
        assert "Article 15" in md
        assert "Accuracy Metrics" in md
        assert "Human Oversight" in md
        assert "Drift Detection" in md
        assert "Incident Summary" in md
        log.close()
