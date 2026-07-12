# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for server compliance endpoints.

Covers: report JSON/markdown, drift detection, dashboard windows,
503 when not configured, parametrised formats, endpoint field
verification, and pipeline performance documentation.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from director_ai.compliance.audit_log import AuditEntry, AuditLog
from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


def _populate(db_path: str, n_approved=8, n_rejected=2):
    log = AuditLog(db_path)
    for _ in range(n_approved):
        log.log(
            AuditEntry(
                prompt="q",
                response="a",
                model="gpt-4o",
                provider="server",
                score=0.85,
                approved=True,
                verdict_confidence=0.9,
                task_type="review",
                domain="",
                latency_ms=12.0,
                timestamp=time.time(),
            )
        )
    for _ in range(n_rejected):
        log.log(
            AuditEntry(
                prompt="q",
                response="bad",
                model="gpt-4o",
                provider="server",
                score=0.2,
                approved=False,
                verdict_confidence=0.7,
                task_type="review",
                domain="",
                latency_ms=18.0,
                timestamp=time.time(),
            )
        )
    log.close()


@pytest.fixture
def compliance_client(tmp_path):
    db_path = str(tmp_path / "compliance.db")
    _populate(db_path)
    cfg = DirectorConfig(compliance_db_path=db_path)
    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def no_compliance_client():
    cfg = DirectorConfig()
    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


class TestComplianceEndpoints:
    def test_report_json(self, compliance_client):
        resp = compliance_client.get("/v1/compliance/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_interactions"] == 10
        assert data["overall_hallucination_rate"] == 0.2
        assert "model_metrics" in data

    def test_report_markdown(self, compliance_client):
        resp = compliance_client.get("/v1/compliance/report", params={"fmt": "md"})
        assert resp.status_code == 200
        assert "Article 15" in resp.text

    def test_drift_endpoint(self, compliance_client):
        resp = compliance_client.get("/v1/compliance/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert "detected" in data
        assert "severity" in data
        assert "z_score" in data

    def test_dashboard_endpoint(self, compliance_client):
        resp = compliance_client.get("/v1/compliance/dashboard")
        assert resp.status_code == 200
        data = resp.json()
        assert "24h" in data
        assert "7d" in data
        assert "30d" in data
        assert "total" in data["30d"]


class TestComplianceNotConfigured:
    def test_report_503(self, no_compliance_client):
        resp = no_compliance_client.get("/v1/compliance/report")
        assert resp.status_code == 503

    def test_drift_503(self, no_compliance_client):
        resp = no_compliance_client.get("/v1/compliance/drift")
        assert resp.status_code == 503

    def test_dashboard_503(self, no_compliance_client):
        resp = no_compliance_client.get("/v1/compliance/dashboard")
        assert resp.status_code == 503

    @pytest.mark.parametrize(
        "endpoint",
        ["/v1/compliance/report", "/v1/compliance/drift", "/v1/compliance/dashboard"],
    )
    def test_all_endpoints_503_when_disabled(self, no_compliance_client, endpoint):
        resp = no_compliance_client.get(endpoint)
        assert resp.status_code == 503


class TestCompliancePerformanceDoc:
    """Document compliance server pipeline performance."""

    def test_drift_response_has_all_fields(self, compliance_client):
        resp = compliance_client.get("/v1/compliance/drift")
        data = resp.json()
        for field in ["detected", "severity", "z_score"]:
            assert field in data, f"Missing: {field}"

    def test_dashboard_has_all_windows(self, compliance_client):
        resp = compliance_client.get("/v1/compliance/dashboard")
        data = resp.json()
        for window in ["24h", "7d", "30d"]:
            assert window in data, f"Missing window: {window}"


class TestGovernanceControlsEndpoint:
    def test_configured_deployment_passes_record_keeping(self, tmp_path, monkeypatch):
        # A durable seal secret is required for the chain to verify across
        # AuditLog instances (the fixture writer vs the server's own handle).
        monkeypatch.setenv("DIRECTOR_AUDIT_HMAC_SECRET", "governance-endpoint-test")
        db_path = str(tmp_path / "compliance.db")
        _populate(db_path)
        app = create_app(config=DirectorConfig(compliance_db_path=db_path))
        with TestClient(app) as client:
            resp = client.get("/v1/compliance/governance-controls")
        assert resp.status_code == 200
        data = resp.json()
        assert data["computed"] is True
        assert data["inputs"]["audit_log_attached"] is True
        assert data["inputs"]["config_attached"] is True
        by_id = {c["control_id"]: c for c in data["controls"]}
        assert by_id["GOV-LOG-01"]["status"] == "passed"
        signals = {s["name"]: s for s in by_id["GOV-LOG-01"]["signals"]}
        assert signals["tamper_evident_chain_verified"]["satisfied"] is True
        assert signals["audit_entries_recorded"]["satisfied"] is True

    def test_unconfigured_deployment_degrades_instead_of_503(
        self, no_compliance_client
    ):
        resp = no_compliance_client.get("/v1/compliance/governance-controls")
        assert resp.status_code == 200
        data = resp.json()
        assert data["inputs"]["audit_log_attached"] is False
        by_id = {c["control_id"]: c for c in data["controls"]}
        assert by_id["GOV-LOG-01"]["status"] == "failing"

    def test_markdown_format(self, compliance_client):
        resp = compliance_client.get(
            "/v1/compliance/governance-controls", params={"fmt": "md"}
        )
        assert resp.status_code == 200
        assert "Computed AI-Governance Controls" in resp.text
        assert "GOV-LOG-01" in resp.text
