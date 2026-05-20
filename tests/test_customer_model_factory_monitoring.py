# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory monitoring tests

from __future__ import annotations

import json
from pathlib import Path

from director_ai.core.customer_model_factory.monitoring_manifest import (
    MonitoringMetrics,
    MonitoringThresholds,
    build_monitoring_manifest,
)
from director_ai.core.customer_model_factory.runtime_package import (
    CustomerRuntimePackage,
)

ROOT = Path(__file__).resolve().parents[1]


def _runtime_package(
    *,
    audit_log_uri: str = "gs://customer-artifacts/customer-alpha/audit/decision-log.jsonl",
):
    return CustomerRuntimePackage(
        schema_version="1.0.0",
        runtime_id="runtime-customer-alpha-20260518",
        ready=True,
        customer_id="customer-alpha",
        workspace_id="customer-alpha-prod",
        tenant_id="customer-alpha-tenant",
        deployment_id="customer-alpha-prod-20260518",
        evidence_hash="a" * 64,
        runtime_mode="offline_private",
        runtime_config={
            "customer_id": "customer-alpha",
            "workspace_id": "customer-alpha-prod",
            "tenant_id": "customer-alpha-tenant",
            "deployment_id": "customer-alpha-prod-20260518",
            "deployment_hash": "b" * 64,
            "evidence_hash": "a" * 64,
            "selected_benchmark_id": "customer-alpha-private-v1",
            "selected_model_artifact_uri": "gs://customer-artifacts/customer-alpha/models/cmf-customer-alpha",
            "threshold": 0.72,
            "abstention_threshold": 0.58,
            "escalation_threshold": 0.40,
            "require_citations": True,
            "audit_log_uri": audit_log_uri,
            "evidence_pack_uri": "gs://customer-artifacts/customer-alpha/evidence/pack",
            "rollback_package_uri": "gs://customer-artifacts/customer-alpha/deployments/previous.json",
            "retention_days": 365,
            "telemetry_mode": "customer_controlled",
            "external_callbacks_allowed": False,
            "callback_endpoints": [],
        },
        findings=(),
        runtime_hash="c" * 64,
    )


def _metrics(**overrides: object) -> MonitoringMetrics:
    metrics = MonitoringMetrics(
        total_decisions=1200,
        input_drift_score=0.04,
        source_corpus_drift_score=0.03,
        false_positive_review_count=2,
        false_negative_incident_count=0,
        abstention_rate=0.05,
        escalation_rate=0.09,
        latency_p95_ms=48.0,
        cost_per_1k_decisions=0.12,
    )
    payload = metrics.to_dict()
    payload.update(overrides)
    return MonitoringMetrics.from_dict(payload)


def _thresholds() -> MonitoringThresholds:
    return MonitoringThresholds(
        max_input_drift_score=0.20,
        max_source_corpus_drift_score=0.15,
        max_false_negative_incidents=0,
        max_abstention_rate=0.20,
        max_escalation_rate=0.30,
        max_latency_p95_ms=250.0,
        max_cost_per_1k_decisions=1.50,
    )


def test_monitoring_manifest_reports_within_control_when_evidence_and_metrics_are_ready():
    manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-20260518",
        runtime_package=_runtime_package(),
        metrics=_metrics(),
        thresholds=_thresholds(),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )

    assert manifest.ready is True
    assert manifest.health_status == "within_control"
    assert manifest.retraining_recommended is False
    assert manifest.recommendations == ()
    assert manifest.package_version["runtime_id"] == "runtime-customer-alpha-20260518"
    assert manifest.decision_log_uri.endswith("decision-log.jsonl")
    assert len(manifest.monitoring_hash) == 64


def test_monitoring_manifest_recommends_retraining_on_threshold_breaches():
    manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-incident",
        runtime_package=_runtime_package(),
        metrics=_metrics(
            input_drift_score=0.31,
            false_negative_incident_count=1,
            latency_p95_ms=350.0,
        ),
        thresholds=_thresholds(),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )

    assert manifest.ready is True
    assert manifest.health_status == "review_required"
    assert manifest.retraining_recommended is True
    assert {item["code"] for item in manifest.recommendations} >= {
        "input_drift_threshold_breached",
        "false_negative_incident_threshold_breached",
        "latency_threshold_breached",
    }


def test_monitoring_manifest_recommends_retraining_on_all_operational_breaches():
    manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-all-breaches",
        runtime_package=_runtime_package(),
        metrics=_metrics(
            source_corpus_drift_score=0.26,
            abstention_rate=0.31,
            escalation_rate=0.41,
            cost_per_1k_decisions=2.75,
        ),
        thresholds=_thresholds(),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )

    assert manifest.ready is True
    assert manifest.health_status == "review_required"
    assert {item["code"] for item in manifest.recommendations} >= {
        "source_corpus_drift_threshold_breached",
        "abstention_threshold_breached",
        "escalation_threshold_breached",
        "cost_threshold_breached",
    }


def test_monitoring_manifest_blocks_health_claim_without_decision_log_evidence():
    manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-no-log",
        runtime_package=_runtime_package(audit_log_uri=""),
        metrics=_metrics(),
        thresholds=_thresholds(),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )

    assert manifest.ready is False
    assert manifest.health_status == "evidence_blocked"
    assert any(
        finding["code"] == "decision_log_missing" for finding in manifest.findings
    )


def test_monitoring_manifest_blocks_missing_identity_and_queue_evidence():
    runtime = _runtime_package()
    runtime = CustomerRuntimePackage(
        **{
            **runtime.to_dict(),
            "ready": False,
            "findings": [{"code": "deployment_not_ready"}],
        }
    )

    manifest = build_monitoring_manifest(
        monitoring_id=" ",
        runtime_package=runtime,
        metrics=_metrics(),
        thresholds=_thresholds(),
        observation_window=" ",
        monitored_at=" ",
        review_queue_uri=" ",
        incident_queue_uri=" ",
    )

    assert manifest.ready is False
    assert manifest.health_status == "evidence_blocked"
    assert {finding["code"] for finding in manifest.findings} >= {
        "monitoring_id_missing",
        "runtime_package_not_ready",
        "observation_window_missing",
        "monitored_at_missing",
        "review_queue_missing",
        "incident_queue_missing",
    }


def test_monitoring_manifest_blocks_tenant_boundary_mismatch():
    runtime = _runtime_package()
    runtime.runtime_config["tenant_id"] = "wrong-tenant"

    manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-mismatch",
        runtime_package=runtime,
        metrics=_metrics(),
        thresholds=_thresholds(),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )

    assert manifest.ready is False
    assert any(
        finding["code"] == "tenant_boundary_mismatch" for finding in manifest.findings
    )


def test_monitoring_manifest_blocks_all_runtime_boundary_mismatches():
    runtime = _runtime_package()
    runtime.runtime_config.update(
        {
            "customer_id": "wrong-customer",
            "workspace_id": "wrong-workspace",
            "deployment_id": "wrong-deployment",
        }
    )

    manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-mismatches",
        runtime_package=runtime,
        metrics=_metrics(),
        thresholds=_thresholds(),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )

    assert manifest.ready is False
    assert {finding["code"] for finding in manifest.findings} >= {
        "customer_boundary_mismatch",
        "workspace_boundary_mismatch",
        "deployment_boundary_mismatch",
    }


def test_monitoring_manifest_serialises_and_round_trips(tmp_path: Path):
    manifest = build_monitoring_manifest(
        monitoring_id="monitor-customer-alpha-20260518",
        runtime_package=_runtime_package(),
        metrics=_metrics(),
        thresholds=_thresholds(),
        observation_window="2026-05-18T00:00:00Z/2026-05-18T12:00:00Z",
        monitored_at="2026-05-18T12:05:00Z",
        review_queue_uri="gs://customer-artifacts/customer-alpha/review/fp.jsonl",
        incident_queue_uri="gs://customer-artifacts/customer-alpha/incidents/fn.jsonl",
    )

    output = manifest.write_json(tmp_path / "monitoring_manifest.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload == manifest.to_dict()
    assert payload["monitoring_hash"] == manifest.monitoring_hash


def test_monitoring_schema_is_machine_readable():
    schema_path = ROOT / "schemas" / "customer-model-factory-monitoring.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "DIRECTOR-AI Customer Model Factory Monitoring Manifest"
    assert set(schema["required"]) >= {
        "monitoring_id",
        "runtime_id",
        "health_status",
        "metrics",
        "thresholds",
        "package_version",
        "monitoring_hash",
    }
