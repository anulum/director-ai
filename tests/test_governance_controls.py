# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Computed Governance Controls Tests

"""Multi-angle tests for the computed NIST/ISO 42001/EU AI Act controls.

Everything goes through the public ``director_ai.compliance`` surface;
audit-chain signals are exercised against a real SQLite ``AuditLog``,
including a deliberate tamper to prove the chain verification is live,
not a stored flag.
"""

import sqlite3
import time

import pytest

from director_ai.compliance import (
    AuditEntry,
    AuditLog,
    ControlSignal,
    GovernanceControl,
    GovernanceControlsReport,
    ReadinessStatus,
    compute_governance_controls,
)
from director_ai.core.config import DirectorConfig


def _entry(**overrides):
    base = {
        "prompt": "What colour is the sky?",
        "response": "Blue.",
        "model": "mock-model",
        "provider": "test",
        "score": 0.9,
        "approved": True,
        "verdict_confidence": 0.8,
        "task_type": "chat",
        "domain": "",
        "latency_ms": 12.0,
        "timestamp": time.time(),
    }
    base.update(overrides)
    return AuditEntry(**base)


@pytest.fixture
def audit_log(tmp_path):
    log = AuditLog(str(tmp_path / "audit.sqlite"))
    log.log(_entry())
    log.log(_entry(response="Azure.", score=0.7))
    yield log
    log.close()


def _control(report: GovernanceControlsReport, control_id: str) -> GovernanceControl:
    matches = [c for c in report.controls if c.control_id == control_id]
    assert len(matches) == 1, f"{control_id} missing from report"
    return matches[0]


def _signal(control: GovernanceControl, name: str) -> ControlSignal:
    matches = [s for s in control.signals if s.name == name]
    assert len(matches) == 1, f"signal {name} missing from {control.control_id}"
    return matches[0]


class TestControlSignalValidation:
    def test_blank_name_rejected(self):
        with pytest.raises(ValueError, match="name"):
            ControlSignal(name="  ", observed="x", satisfied=True)

    def test_blank_observation_rejected(self):
        with pytest.raises(ValueError, match="observation"):
            ControlSignal(name="x", observed="  ", satisfied=True)

    def test_to_dict_shape(self):
        signal = ControlSignal(name="s", observed="seen", satisfied=False)
        assert signal.to_dict() == {
            "name": "s",
            "observed": "seen",
            "satisfied": False,
        }


class TestGovernanceControlValidation:
    def _kwargs(self, **overrides):
        base = {
            "control_id": "gov-x-01",
            "title": "Test control",
            "nist_ai_rmf_refs": ("GOVERN 1",),
            "iso42001_refs": ("Clause 6.1",),
            "eu_ai_act_refs": ("Article 9(1)",),
            "signals": (ControlSignal(name="s", observed="ok", satisfied=True),),
        }
        base.update(overrides)
        return base

    def test_id_is_normalised_uppercase(self):
        control = GovernanceControl(**self._kwargs())
        assert control.control_id == "GOV-X-01"

    @pytest.mark.parametrize("bad_id", ["", "  ", "id with spaces", "x/y"])
    def test_invalid_id_rejected(self, bad_id):
        with pytest.raises(ValueError, match="control_id"):
            GovernanceControl(**self._kwargs(control_id=bad_id))

    def test_blank_title_rejected(self):
        with pytest.raises(ValueError, match="title"):
            GovernanceControl(**self._kwargs(title="  "))

    @pytest.mark.parametrize("bad", [(), ("RMF 1",), ("govern 1",)])
    def test_nist_refs_must_use_rmf_functions(self, bad):
        with pytest.raises(ValueError, match="nist_ai_rmf_refs"):
            GovernanceControl(**self._kwargs(nist_ai_rmf_refs=bad))

    @pytest.mark.parametrize("bad", [(), ("6.1",), ("ISO 42001 6.1",)])
    def test_iso_refs_must_use_clause_or_annex(self, bad):
        with pytest.raises(ValueError, match="iso42001_refs"):
            GovernanceControl(**self._kwargs(iso42001_refs=bad))

    @pytest.mark.parametrize("bad", [(), ("Art 9",), ("9(1)",)])
    def test_eu_refs_must_use_article_prefix(self, bad):
        with pytest.raises(ValueError, match="eu_ai_act_refs"):
            GovernanceControl(**self._kwargs(eu_ai_act_refs=bad))

    def test_annex_a_reference_accepted(self):
        control = GovernanceControl(**self._kwargs(iso42001_refs=("A.7",)))
        assert control.iso42001_refs == ("A.7",)


class TestStatusDerivation:
    def _control_with(self, satisfied_flags):
        signals = tuple(
            ControlSignal(name=f"s{i}", observed="seen", satisfied=flag)
            for i, flag in enumerate(satisfied_flags)
        )
        return GovernanceControl(
            control_id="GOV-D-01",
            title="Derivation",
            nist_ai_rmf_refs=("MEASURE 2",),
            iso42001_refs=("Clause 9.1",),
            eu_ai_act_refs=("Article 12(1)",),
            signals=signals,
        )

    def test_all_satisfied_is_pass(self):
        assert self._control_with([True, True]).status is ReadinessStatus.PASS

    def test_some_satisfied_is_warning(self):
        assert self._control_with([True, False]).status is ReadinessStatus.WARNING

    def test_none_satisfied_is_fail(self):
        assert self._control_with([False, False]).status is ReadinessStatus.FAIL

    def test_no_signals_is_not_applicable(self):
        assert self._control_with([]).status is ReadinessStatus.NOT_APPLICABLE


class TestComputedReportWithoutInputs:
    def test_degrades_honestly_with_no_config_and_no_audit_log(self, tmp_path):
        report = compute_governance_controls(evidence_root=tmp_path)
        assert report.inputs == {
            "config_attached": False,
            "audit_log_attached": False,
            "evidence_root": str(tmp_path),
        }
        for control_id in (
            "GOV-RISK-01",
            "GOV-DATA-01",
            "GOV-DOC-01",
            "GOV-LOG-01",
            "GOV-ACC-01",
            "GOV-OVR-01",
        ):
            assert _control(report, control_id).status is ReadinessStatus.FAIL
        summary = report.summary()
        assert summary["failures"] == 6
        assert summary["risk_level"] == "critical"
        assert summary["readiness_score"] == 0.0

    def test_generated_at_override_and_default(self, tmp_path):
        stamped = compute_governance_controls(
            evidence_root=tmp_path, generated_at="2026-07-12T00:00:00Z"
        )
        assert stamped.generated_at == "2026-07-12T00:00:00Z"
        defaulted = compute_governance_controls(evidence_root=tmp_path)
        assert defaulted.generated_at.endswith("Z")


class TestComputedReportWithLiveState:
    def _config(self, **overrides):
        base = {
            "use_nli": False,
            "scorer_backend": "lite",
            "vector_backend": "memory",
            "redact_pii": True,
            "tenant_routing": True,
        }
        base.update(overrides)
        return DirectorConfig(**base)

    def test_full_state_passes_risk_data_log_and_oversight(self, audit_log, tmp_path):
        report = compute_governance_controls(
            config=self._config(),
            audit_log=audit_log,
            evidence_root=tmp_path,
        )
        assert _control(report, "GOV-RISK-01").status is ReadinessStatus.PASS
        assert _control(report, "GOV-DATA-01").status is ReadinessStatus.PASS
        assert _control(report, "GOV-LOG-01").status is ReadinessStatus.PASS
        assert _control(report, "GOV-ACC-01").status is ReadinessStatus.PASS
        assert _control(report, "GOV-OVR-01").status is ReadinessStatus.PASS

    def test_documentation_signals_reflect_evidence_root(self, tmp_path):
        (tmp_path / "docs" / "_generated").mkdir(parents=True)
        (tmp_path / "docs" / "_generated" / "capability_manifest.json").write_text(
            "{}", encoding="utf-8"
        )
        report = compute_governance_controls(evidence_root=tmp_path)
        doc = _control(report, "GOV-DOC-01")
        assert _signal(doc, "capability_inventory_present").satisfied is True
        assert _signal(doc, "public_benchmark_evidence_present").satisfied is False
        assert doc.status is ReadinessStatus.WARNING

    def test_repo_checkout_satisfies_documentation_artefacts(self, audit_log):
        import director_ai

        repo_root = __import__("pathlib").Path(director_ai.__file__).parents[2]
        report = compute_governance_controls(
            audit_log=audit_log, evidence_root=repo_root
        )
        assert _control(report, "GOV-DOC-01").status is ReadinessStatus.PASS

    def test_disabled_knobs_downgrade_data_governance(self, tmp_path):
        config = self._config(redact_pii=False, tenant_routing=False)
        report = compute_governance_controls(config=config, evidence_root=tmp_path)
        data = _control(report, "GOV-DATA-01")
        assert data.status is ReadinessStatus.WARNING
        assert _signal(data, "pii_redaction_enabled").satisfied is False
        assert _signal(data, "tenant_isolation_enabled").satisfied is False
        assert _signal(data, "grounding_store_configured").satisfied is True

    def test_inverted_thresholds_fail_guard_signal(self, tmp_path):
        config = self._config(coherence_threshold=0.3, hard_limit=0.5)
        report = compute_governance_controls(config=config, evidence_root=tmp_path)
        risk = _control(report, "GOV-RISK-01")
        assert _signal(risk, "guard_thresholds_configured").satisfied is False
        oversight = _control(report, "GOV-OVR-01")
        assert _signal(oversight, "review_band_configured").satisfied is False

    def test_empty_audit_log_is_warning_not_pass(self, tmp_path):
        log = AuditLog(str(tmp_path / "empty.sqlite"))
        try:
            report = compute_governance_controls(audit_log=log, evidence_root=tmp_path)
            record = _control(report, "GOV-LOG-01")
            assert _signal(record, "tamper_evident_chain_verified").satisfied is True
            assert _signal(record, "audit_entries_recorded").satisfied is False
            assert record.status is ReadinessStatus.WARNING
            accuracy = _control(report, "GOV-ACC-01")
            assert (
                _signal(accuracy, "interactions_recorded_for_metrics").satisfied
                is False
            )
        finally:
            log.close()

    def test_tampered_chain_is_detected_live(self, tmp_path):
        db_path = tmp_path / "tampered.sqlite"
        log = AuditLog(str(db_path))
        log.log(_entry())
        log.log(_entry(response="Azure."))
        log.close()
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE audit_log SET score = 0.01 WHERE id = 1")
            conn.commit()
        reopened = AuditLog(str(db_path))
        try:
            report = compute_governance_controls(
                audit_log=reopened, evidence_root=tmp_path
            )
            record = _control(report, "GOV-LOG-01")
            chain = _signal(record, "tamper_evident_chain_verified")
            assert chain.satisfied is False
            assert "FAILED at row" in chain.observed
            assert record.status is ReadinessStatus.WARNING
        finally:
            reopened.close()


class TestReportSerialisation:
    def test_to_dict_shape_and_privacy_block(self, tmp_path):
        report = compute_governance_controls(evidence_root=tmp_path)
        payload = report.to_dict()
        assert payload["frameworks"] == [
            "NIST AI RMF 1.0",
            "ISO/IEC 42001:2023",
            "EU AI Act (Regulation (EU) 2024/1689)",
        ]
        assert payload["computed"] is True
        assert payload["privacy"]["certification_claimed"] is False
        assert payload["privacy"]["raw_interaction_text_included"] is False
        assert "not" in payload["disclaimer"]
        ids = [control["control_id"] for control in payload["controls"]]
        assert ids == [
            "GOV-RISK-01",
            "GOV-DATA-01",
            "GOV-DOC-01",
            "GOV-LOG-01",
            "GOV-ACC-01",
            "GOV-OVR-01",
        ]
        first = payload["controls"][0]
        assert first["nist_ai_rmf_refs"] == ["GOVERN 1", "MANAGE 1"]
        assert first["iso42001_refs"] == ["Clause 6.1", "Clause 8.2"]
        assert first["eu_ai_act_refs"] == ["Article 9(1)", "Article 9(2)"]
        assert all("observed" in s for s in first["signals"])

    def test_markdown_contains_crosswalk_and_disclaimer(self, audit_log, tmp_path):
        report = compute_governance_controls(
            audit_log=audit_log, evidence_root=tmp_path
        )
        rendered = report.to_markdown()
        assert "# Computed AI-Governance Controls" in rendered
        assert "| GOV-LOG-01 |" in rendered
        assert "MANAGE 4" in rendered
        assert "Clause 9.1" in rendered
        assert "Article 12(1)" in rendered
        assert "✓ tamper_evident_chain_verified" in rendered
        assert _DISCLAIMER_SNIPPET in rendered

    def test_summary_counts_mixed_statuses(self, audit_log, tmp_path):
        report = compute_governance_controls(
            audit_log=audit_log, evidence_root=tmp_path
        )
        summary = report.summary()
        assert summary["total_controls"] == 6
        assert (
            summary["passed"]
            + summary["warnings"]
            + summary["failures"]
            + summary["not_applicable"]
            == 6
        )
        assert summary["risk_level"] in ("critical", "attention_required", "ready")


_DISCLAIMER_SNIPPET = "not an EU AI Act conformity assessment"
