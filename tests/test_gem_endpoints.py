# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Phase 5 gem REST endpoints (numeric, reasoning, temporal)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


@pytest.fixture()
def client():
    from starlette.testclient import TestClient

    cfg = DirectorConfig(use_nli=False)
    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


# -- /v1/verify/numeric -------------------------------------------------------


class TestVerifyNumeric:
    def test_clean_text_returns_valid(self, client):
        resp = client.post("/v1/verify/numeric", json={"text": "The sky is blue."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["error_count"] == 0

    def test_bad_arithmetic_detected(self, client):
        text = "Revenue grew 50% from $100 to $120."
        resp = client.post("/v1/verify/numeric", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data["error_count"] >= 1
        assert any(i["issue_type"] == "arithmetic" for i in data["issues"])

    def test_probability_over_100(self, client):
        text = "There is a 150% probability chance of rain."
        resp = client.post("/v1/verify/numeric", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert any(i["issue_type"] == "probability" for i in data["issues"])

    def test_date_logic_error(self, client):
        text = "She was born in 1990 and died in 1980."
        resp = client.post("/v1/verify/numeric", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert any(i["issue_type"] == "date_logic" for i in data["issues"])

    def test_empty_text_rejected(self, client):
        resp = client.post("/v1/verify/numeric", json={"text": ""})
        assert resp.status_code == 422

    def test_response_model_fields(self, client):
        resp = client.post("/v1/verify/numeric", json={"text": "Test 42."})
        assert resp.status_code == 200
        data = resp.json()
        assert "claims_found" in data
        assert "issues" in data
        assert "valid" in data
        assert "error_count" in data
        assert "warning_count" in data


# -- /v1/verify/reasoning -----------------------------------------------------


class TestVerifyReasoning:
    def test_valid_chain(self, client):
        text = (
            "Step 1: All mammals are warm-blooded. "
            "Step 2: Dogs are mammals. "
            "Step 3: Therefore, dogs are warm-blooded."
        )
        resp = client.post("/v1/verify/reasoning", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_found"] >= 2
        assert "verdicts" in data
        assert "chain_valid" in data

    def test_circular_reasoning(self, client):
        text = (
            "Step 1: The earth is round because it is spherical. "
            "Step 2: The earth is spherical because it is round."
        )
        resp = client.post("/v1/verify/reasoning", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        if data["steps_found"] >= 2:
            assert any(
                v["verdict"] in ("circular", "non_sequitur")
                for v in data["verdicts"]
            )

    def test_single_step_returns_empty(self, client):
        resp = client.post("/v1/verify/reasoning", json={"text": "Just one sentence."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_found"] <= 1
        assert data["issues_found"] == 0

    def test_response_model_fields(self, client):
        text = "Step 1: A is true. Step 2: Therefore B."
        resp = client.post("/v1/verify/reasoning", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert "steps_found" in data
        assert "verdicts" in data
        assert "chain_valid" in data
        assert "issues_found" in data


# -- /v1/temporal-freshness ---------------------------------------------------


class TestTemporalFreshness:
    def test_position_claim_flagged(self, client):
        text = "The CEO of Apple is Tim Cook."
        resp = client.post("/v1/temporal-freshness", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_temporal_claims"] is True
        assert len(data["claims"]) >= 1
        assert any(c["claim_type"] == "position" for c in data["claims"])

    def test_no_temporal_claims(self, client):
        text = "Water is composed of hydrogen and oxygen."
        resp = client.post("/v1/temporal-freshness", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_staleness_risk"] == 0.0

    def test_response_model_fields(self, client):
        text = "The president of France is someone."
        resp = client.post("/v1/temporal-freshness", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert "claims" in data
        assert "overall_staleness_risk" in data
        assert "has_temporal_claims" in data
        assert "stale_claim_count" in data
        for claim in data["claims"]:
            assert "text" in claim
            assert "claim_type" in claim
            assert "staleness_risk" in claim
            assert "reason" in claim

    def test_empty_text_rejected(self, client):
        resp = client.post("/v1/temporal-freshness", json={"text": ""})
        assert resp.status_code == 422
