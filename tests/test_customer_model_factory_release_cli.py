# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory release CLI tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.customer_model_factory.evidence_pack import (
    CustomerEvidencePackManifest,
)
from director_ai.core.customer_model_factory.monitoring_manifest import (
    CustomerMonitoringManifest,
    MonitoringMetrics,
    MonitoringThresholds,
)
from director_ai.core.customer_model_factory.risk_register import CustomerRiskRegister
from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)
from tools.assemble_customer_model_factory_release import main as assemble_main


def _write_inputs(tmp_path: Path, *, enterprise_ready: bool = True) -> dict[str, Path]:
    runtime = CustomerRuntimePackage(
        schema_version="1.0.0",
        runtime_id="runtime-bank-alpha-20260518",
        ready=True,
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        deployment_id="bank-alpha-prod-20260518",
        evidence_hash="a" * 64,
        runtime_mode="offline_private",
        runtime_config={
            "customer_id": "bank-alpha",
            "workspace_id": "bank-alpha-prod",
            "tenant_id": "bank-alpha-tenant",
            "deployment_id": "bank-alpha-prod-20260518",
            "deployment_hash": "b" * 64,
            "evidence_hash": "a" * 64,
            "selected_model_artifact_uri": "gs://customer-artifacts/bank-alpha/models/cmf-bank-alpha",
            "audit_log_uri": "gs://customer-artifacts/bank-alpha/audit/decision-log.jsonl",
            "evidence_pack_uri": "gs://customer-artifacts/bank-alpha/evidence/pack",
            "telemetry_mode": "customer_controlled",
            "external_callbacks_allowed": False,
        },
        findings=(),
        runtime_hash="c" * 64,
    )
    evidence = CustomerEvidencePackManifest(
        schema_version="1.0.0",
        package_id="evidence-bank-alpha-20260518",
        ready=True,
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        deployment_id="bank-alpha-prod-20260518",
        environment="production",
        classification="restricted",
        export_uri="gs://customer-artifacts/bank-alpha/evidence/pack",
        external_callbacks_allowed=False,
        callback_endpoints=(),
        artefacts={"deployment_hash": "b" * 64},
        control_evidence={"human_escalation": ("requires_escalation",)},
        findings=(),
        evidence_hash="a" * 64,
    )
    monitoring = CustomerMonitoringManifest(
        schema_version="1.0.0",
        monitoring_id="monitor-bank-alpha-20260518",
        ready=True,
        health_status="within_control",
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        runtime_id="runtime-bank-alpha-20260518",
        deployment_id="bank-alpha-prod-20260518",
        evidence_hash="a" * 64,
        decision_log_uri="gs://customer-artifacts/bank-alpha/audit/decision-log.jsonl",
        review_queue_uri="gs://customer-artifacts/bank-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/bank-alpha/incidents/fn.jsonl",
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        metrics=MonitoringMetrics(
            total_decisions=1200,
            input_drift_score=0.04,
            source_corpus_drift_score=0.03,
            false_positive_review_count=2,
            false_negative_incident_count=0,
            abstention_rate=0.05,
            escalation_rate=0.09,
            latency_p95_ms=48.0,
            cost_per_1k_decisions=0.12,
        ),
        thresholds=MonitoringThresholds(
            max_input_drift_score=0.20,
            max_source_corpus_drift_score=0.15,
            max_false_negative_incidents=0,
            max_abstention_rate=0.20,
            max_escalation_rate=0.30,
            max_latency_p95_ms=250.0,
            max_cost_per_1k_decisions=1.50,
        ),
        package_version={"runtime_id": "runtime-bank-alpha-20260518"},
        retraining_recommended=False,
        recommendations=(),
        findings=(),
        monitoring_hash="d" * 64,
    )
    risk = CustomerRiskRegister(
        schema_version="1.0.0",
        register_id="risk_register_bank_alpha-20260518",
        ready=True,
        customer_id="bank-alpha",
        workspace_id="bank-alpha-prod",
        tenant_id="bank-alpha-tenant",
        deployment_id="bank-alpha-prod-20260518",
        evidence_hash="a" * 64,
        monitoring_hash="d" * 64,
        generated_at="2026-05-18",
        risks=(),
        findings=(),
        register_hash="e" * 64,
    )
    paths = {
        "runtime": tmp_path / "runtime.json",
        "evidence": tmp_path / "evidence.json",
        "monitoring": tmp_path / "monitoring.json",
        "risk": tmp_path / "risk.json",
        "enterprise": tmp_path / "enterprise.json",
        "output": tmp_path / "release_gate.json",
    }
    runtime.write_json(paths["runtime"])
    evidence.write_json(paths["evidence"])
    monitoring.write_json(paths["monitoring"])
    risk.write_json(paths["risk"])
    paths["enterprise"].write_text(
        json.dumps(
            {
                "ready": enterprise_ready,
                "blocking_debt_ids": [] if enterprise_ready else ["TRUST-DEBT-0002"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return paths


def test_release_cli_writes_ready_release_gate(tmp_path: Path):
    paths = _write_inputs(tmp_path)

    exit_code = assemble_main(
        [
            "--release-id",
            "release-bank-alpha-20260518",
            "--generated-at",
            "2026-05-18T18:45:00Z",
            "--enterprise-readiness",
            str(paths["enterprise"]),
            "--runtime-package",
            str(paths["runtime"]),
            "--evidence-pack",
            str(paths["evidence"]),
            "--monitoring-manifest",
            str(paths["monitoring"]),
            "--risk-register",
            str(paths["risk"]),
            "--output",
            str(paths["output"]),
        ]
    )
    payload = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["promotion_allowed"] is True
    assert payload["blockers"] == []
    assert payload["artifact_hashes"]["risk_register_hash"] == "e" * 64


def test_release_cli_fails_closed_when_enterprise_readiness_blocks(tmp_path: Path):
    paths = _write_inputs(tmp_path, enterprise_ready=False)

    exit_code = assemble_main(
        [
            "--release-id",
            "release-bank-alpha-blocked",
            "--generated-at",
            "2026-05-18T18:45:00Z",
            "--enterprise-readiness",
            str(paths["enterprise"]),
            "--runtime-package",
            str(paths["runtime"]),
            "--evidence-pack",
            str(paths["evidence"]),
            "--monitoring-manifest",
            str(paths["monitoring"]),
            "--risk-register",
            str(paths["risk"]),
            "--output",
            str(paths["output"]),
        ]
    )
    payload = json.loads(paths["output"].read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["promotion_allowed"] is False
    assert any(
        blocker["code"] == "enterprise_trust_not_ready"
        for blocker in payload["blockers"]
    )
