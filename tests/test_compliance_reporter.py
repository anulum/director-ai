# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for EU AI Act compliance reporter pipeline."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from director_ai.compliance import annex_iv as annex_iv_module
from director_ai.compliance.audit_log import AuditEntry, AuditLog
from director_ai.compliance.reporter import (
    AnnexIVTechnicalDocumentationContext,
    Article15TemplateContext,
    ComplianceReporter,
    _wilson_ci,
)


def _annex_iv_context(**overrides) -> AnnexIVTechnicalDocumentationContext:
    values = {
        "provider_name": "Example Provider GmbH",
        "system_version": "2026.07",
        "previous_version_relationship": "Supersedes 2026.06; policy-only update.",
        "external_dependencies": "Director-AI and the approved model endpoint.",
        "software_firmware_requirements": "CPython 3.12; no firmware dependency.",
        "distribution_forms": "Container image and authenticated API.",
        "intended_hardware": "x86-64 server with operator-qualified resources.",
        "user_interface": "Authenticated review API and operator dashboard.",
        "instructions_for_use": "See the deployment runbook.",
        "development_methods": "Reviewed source changes and pinned dependencies.",
        "design_specifications": "Thresholded evidence-grounding guardrail.",
        "architecture_and_resources": "Gateway, scorer, audit store, review queue.",
        "data_requirements": "Versioned grounding corpus and evaluation partitions.",
        "predetermined_changes": "Threshold changes require release review.",
        "validation_and_testing": "Focused tests, preflight, and release evidence.",
        "monitoring_functioning_control": "Metrics, drift, incidents, and overrides.",
        "performance_metric_rationale": "Rates and Wilson intervals match the risk.",
        "lifecycle_changes": "Changes are recorded in the release ledger.",
        "standards_and_specifications": "Operator-maintained standards register.",
        "eu_declaration_of_conformity_ref": "Pending applicability determination.",
    }
    values.update(overrides)
    return AnnexIVTechnicalDocumentationContext(**values)


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

    def test_empty_drift_periods_and_wilson_zero_total(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        reporter = ComplianceReporter(log)

        assert reporter._compute_drift_periods([], 100.0, 200.0) == []
        assert _wilson_ci(successes=0, total=0) == 1.0

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
        assert "**System:** Director-AI Hallucination Guardrail" in md
        assert "generated automatically by Director-AI v3.21.0" in md
        log.close()


class TestArticle15Template:
    @staticmethod
    def _context() -> Article15TemplateContext:
        return Article15TemplateContext(
            system_name="Director-AI hospital triage guard",
            intended_purpose=(
                "Score generated triage advice against hospital knowledge-base facts."
            ),
            deployment_context="EU clinical decision-support assistant gateway.",
            risk_management_summary=(
                "Low-score responses are blocked and routed to clinical review."
            ),
            data_governance_summary=(
                "Audit events are stored per tenant with PII redaction enabled."
            ),
            robustness_summary=(
                "Streaming halt, NLI scoring, drift checks, and adversarial tests run."
            ),
            cybersecurity_summary=(
                "API-key tenant binding, rate limits, and signed KB entries are enabled."
            ),
            human_oversight_summary=(
                "Human reviewers can override, reject, or request regeneration."
            ),
            post_market_monitoring_summary=(
                "Operations reviews drift, incidents, and override rates every week."
            ),
            known_limitations=("Does not diagnose patients.",),
            residual_risks=("Clinical context can be incomplete.",),
            evidence_refs=(
                "docs/PRODUCTION_CHECKLIST.md#compliance",
                "SECURITY.md#residual-risks",
            ),
        )

    def test_article15_template_contains_required_sections_without_raw_entries(
        self, tmp_path
    ):
        log = AuditLog(tmp_path / "test.db")
        log.log(
            _entry(
                score=0.92,
                approved=True,
                domain="medical",
                human_override=False,
            )
        )
        log.log(_entry(score=0.21, approved=False, domain="medical"))
        report = ComplianceReporter(log).generate_report()
        context = self._context()

        payload = report.to_article15_template(context)
        markdown = report.to_article15_markdown(context)

        assert payload["system"]["name"] == "Director-AI hospital triage guard"
        assert payload["article_15_sections"]["accuracy"]["total_interactions"] == 2
        assert payload["article_15_sections"]["robustness"]["summary"].startswith(
            "Streaming halt"
        )
        assert payload["article_15_sections"]["cybersecurity"]["summary"].startswith(
            "API-key tenant binding"
        )
        assert payload["article_15_sections"]["residual_risk"] == {
            "known_limitations": ["Does not diagnose patients."],
            "residual_risks": ["Clinical context can be incomplete."],
        }
        assert payload["privacy"] == {
            "payload_classification": "tenant_safe",
            "raw_interaction_text_included": False,
        }
        assert "'q'" not in repr(payload)
        assert "'a'" not in repr(payload)
        assert "EU AI Act Article 15 Technical Documentation" in markdown
        assert "Accuracy, Robustness, and Cybersecurity" in markdown
        assert "docs/PRODUCTION_CHECKLIST.md#compliance" in markdown
        log.close()

    def test_annex_iv_template_preserves_nine_sections_and_claim_boundary(
        self, tmp_path
    ):
        log = AuditLog(tmp_path / "test.db")
        log.log(_entry(score=0.92, approved=True))
        report = ComplianceReporter(log).generate_report()
        context = replace(self._context(), annex_iv=_annex_iv_context())

        payload = report.to_annex_iv_template(context)
        markdown = report.to_annex_iv_markdown(context)
        combined = report.to_article15_template(context)

        sections = payload["sections"]
        assert list(sections) == [
            "1_general_description",
            "2_development_and_system_elements",
            "3_monitoring_functioning_and_control",
            "4_performance_metrics",
            "5_risk_management_system",
            "6_lifecycle_changes",
            "7_standards_and_specifications",
            "8_eu_declaration_of_conformity",
            "9_post_market_monitoring",
        ]
        metrics = sections["4_performance_metrics"]["measured_performance"]
        assert metrics["total_interactions"] == 1
        assert payload["claim_boundary"] == {
            "operator_authored_context_required": True,
            "conformity_assessment_claimed": False,
            "legal_advice": False,
        }
        assert "annex_iv_technical_documentation" in combined
        assert "## 1. General Description" in markdown
        assert "## 9. Post-Market Monitoring" in markdown
        assert "Conformity assessment claimed: false" in markdown
        assert "'q'" not in repr(payload)
        assert "'a'" not in repr(payload)
        log.close()

    def test_annex_iv_template_requires_explicit_nested_context(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        report = ComplianceReporter(log).generate_report()

        with pytest.raises(ValueError, match="annex_iv context is required"):
            report.to_annex_iv_template(self._context())

        with pytest.raises(
            ValueError,
            match=r"annex_iv\.provider_name is required",
        ):
            _annex_iv_context(provider_name=" ")
        log.close()

    def test_article15_markdown_appends_annex_iv_when_supplied(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        report = ComplianceReporter(log).generate_report()
        context = replace(self._context(), annex_iv=_annex_iv_context())

        markdown = report.to_article15_markdown(context)

        assert "# EU AI Act Article 15 Technical Documentation" in markdown
        assert "# EU AI Act Annex IV Technical Documentation" in markdown
        assert "Pending applicability determination." in markdown
        log.close()

    @pytest.mark.parametrize(
        ("field_name", "invalid_value", "error"),
        (
            ("sections", "not-a-map", "sections must be a dict"),
            ("evidence_refs", "not-a-list", "evidence_refs must be a list"),
        ),
    )
    def test_annex_iv_markdown_rejects_invalid_payload_schema(
        self,
        tmp_path,
        monkeypatch,
        field_name,
        invalid_value,
        error,
    ):
        log = AuditLog(tmp_path / "test.db")
        report = ComplianceReporter(log).generate_report()
        context = replace(self._context(), annex_iv=_annex_iv_context())
        original_builder = annex_iv_module.build_annex_iv_template

        def invalid_builder(report_value, context_value):
            payload = original_builder(report_value, context_value)
            payload[field_name] = invalid_value
            return payload

        monkeypatch.setattr(
            annex_iv_module,
            "build_annex_iv_template",
            invalid_builder,
        )
        with pytest.raises(TypeError, match=error):
            report.to_annex_iv_markdown(context)
        log.close()

    def test_article15_markdown_reports_empty_residuals_and_evidence(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        report = ComplianceReporter(log).generate_report()
        context = Article15TemplateContext(
            system_name="Director-AI hospital triage guard",
            intended_purpose="Score generated triage advice against facts.",
            deployment_context="EU clinical decision-support assistant gateway.",
            risk_management_summary="Low-score responses are blocked.",
            data_governance_summary="Audit events are stored per tenant.",
            robustness_summary="Drift checks run weekly.",
            cybersecurity_summary="Signed KB entries are enabled.",
            human_oversight_summary="Reviewers can override decisions.",
            post_market_monitoring_summary="Operations review incidents weekly.",
            known_limitations=(" ",),
            residual_risks=(),
            evidence_refs=(" ",),
            annex_iv=_annex_iv_context(),
        )

        markdown = report.to_article15_markdown(context)

        assert "- No residual risks supplied in this template context." in markdown
        assert "- No evidence references supplied." in markdown
        assert "  - None supplied." in markdown

        log.close()

    def test_article15_markdown_rejects_invalid_template_payload_schema(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        report = ComplianceReporter(log).generate_report()
        original_template_payload = report.to_article15_template

        def invalid_template_payload(context):
            payload = original_template_payload(context)
            payload["article_15_sections"] = "not-a-section-map"
            return payload

        report.to_article15_template = invalid_template_payload

        with pytest.raises(TypeError, match="article_15_sections must be a dict"):
            report.to_article15_markdown(self._context())

        log.close()

    def test_article15_markdown_rejects_invalid_list_payload_schema(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        report = ComplianceReporter(log).generate_report()
        original_template_payload = report.to_article15_template

        def invalid_template_payload(context):
            payload = original_template_payload(context)
            payload["evidence_refs"] = "not-a-list"
            return payload

        report.to_article15_template = invalid_template_payload

        with pytest.raises(TypeError, match="evidence_refs must be a list"):
            report.to_article15_markdown(self._context())

        log.close()

    def test_article15_template_rejects_missing_required_context(self, tmp_path):
        log = AuditLog(tmp_path / "test.db")
        report = ComplianceReporter(log).generate_report()

        try:
            Article15TemplateContext(
                system_name="",
                intended_purpose="purpose",
                deployment_context="context",
                risk_management_summary="risk",
                data_governance_summary="data",
                robustness_summary="robust",
                cybersecurity_summary="security",
                human_oversight_summary="oversight",
                post_market_monitoring_summary="monitoring",
            )
        except ValueError as exc:
            assert "system_name is required" in str(exc)
        else:  # pragma: no cover - asserts failure message clarity
            raise AssertionError("missing system_name should be rejected")

        assert report.total_interactions == 0
        log.close()


def test_generate_report_returns_the_article15_report_contract(tmp_path):
    from director_ai.compliance.reporter import Article15Report

    log = AuditLog(tmp_path / "contract.db")
    try:
        report = ComplianceReporter(log).generate_report()
    finally:
        log.close()

    assert isinstance(report, Article15Report)
    assert report.total_interactions == 0
    assert report.period_start <= report.period_end
    assert report.report_timestamp > 0
