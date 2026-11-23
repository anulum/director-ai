# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for the verification-gem REST endpoints.

Covers: explainability, claim provenance, conformal prediction,
feedback loop, adversarial, consensus, cost endpoints, pipeline
integration with FastAPI server, and performance documentation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from starlette.testclient import TestClient

pytest.importorskip("fastapi", reason="fastapi not installed")

from director_ai.core.config import DirectorConfig
from director_ai.server import create_app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Serve one app, built with NLI off, to every test in this module."""
    from starlette.testclient import TestClient

    cfg = DirectorConfig(use_nli=False)
    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


# -- /v1/verify/numeric -------------------------------------------------------


class TestVerifyNumeric:
    """Tests for the numeric verification endpoint ``/v1/verify/numeric``."""

    def test_clean_text_returns_valid(self, client: TestClient) -> None:
        """Text without numeric claims verifies as valid, with no errors."""
        resp = client.post("/v1/verify/numeric", json={"text": "The sky is blue."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is True
        assert data["error_count"] == 0

    def test_bad_arithmetic_detected(self, client: TestClient) -> None:
        """A growth percentage that contradicts its own figures is an arithmetic error."""
        text = "Revenue grew 50% from $100 to $120."
        resp = client.post("/v1/verify/numeric", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["valid"] is False
        assert data["error_count"] >= 1
        assert any(i["issue_type"] == "arithmetic" for i in data["issues"])

    def test_probability_over_100(self, client: TestClient) -> None:
        """A probability above 100 % is reported as a probability issue."""
        text = "There is a 150% probability chance of rain."
        resp = client.post("/v1/verify/numeric", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert any(i["issue_type"] == "probability" for i in data["issues"])

    def test_date_logic_error(self, client: TestClient) -> None:
        """A death year before the birth year is reported as a date-logic issue."""
        text = "She was born in 1990 and died in 1980."
        resp = client.post("/v1/verify/numeric", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert any(i["issue_type"] == "date_logic" for i in data["issues"])

    def test_empty_text_rejected(self, client: TestClient) -> None:
        """Empty text is refused with HTTP 422."""
        resp = client.post("/v1/verify/numeric", json={"text": ""})
        assert resp.status_code == 422

    def test_response_model_fields(self, client: TestClient) -> None:
        """The response carries the claim, issue, validity and count fields."""
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
    """Tests for the reasoning-chain endpoint ``/v1/verify/reasoning``."""

    def test_valid_chain(self, client: TestClient) -> None:
        """A three-step syllogism yields at least two steps and a chain verdict."""
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

    def test_circular_reasoning(self, client: TestClient) -> None:
        """Two steps that justify each other get a circular or non-sequitur verdict."""
        text = (
            "Step 1: The earth is round because it is spherical. "
            "Step 2: The earth is spherical because it is round."
        )
        resp = client.post("/v1/verify/reasoning", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        if data["steps_found"] >= 2:
            assert any(
                v["verdict"] in ("circular", "non_sequitur") for v in data["verdicts"]
            )

    def test_single_step_returns_empty(self, client: TestClient) -> None:
        """A single sentence yields at most one step and no issues."""
        resp = client.post("/v1/verify/reasoning", json={"text": "Just one sentence."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["steps_found"] <= 1
        assert data["issues_found"] == 0

    def test_response_model_fields(self, client: TestClient) -> None:
        """The response carries the steps, verdicts and chain-validity fields."""
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
    """Tests for the temporal-freshness endpoint ``/v1/temporal-freshness``."""

    def test_position_claim_flagged(self, client: TestClient) -> None:
        """A claim about who holds a position is flagged as a position claim."""
        text = "The CEO of Apple is Tim Cook."
        resp = client.post("/v1/temporal-freshness", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_temporal_claims"] is True
        assert len(data["claims"]) >= 1
        assert any(c["claim_type"] == "position" for c in data["claims"])

    def test_no_temporal_claims(self, client: TestClient) -> None:
        """A timeless scientific fact carries no staleness risk."""
        text = "Water is composed of hydrogen and oxygen."
        resp = client.post("/v1/temporal-freshness", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall_staleness_risk"] == 0.0

    def test_response_model_fields(self, client: TestClient) -> None:
        """The response carries the claims, staleness-risk and temporal-claim fields."""
        text = "The president of France is someone."
        resp = client.post("/v1/temporal-freshness", json={"text": text})
        assert resp.status_code == 200
        data = resp.json()
        assert "claims" in data
        assert "overall_staleness_risk" in data
        assert "has_temporal_claims" in data
        assert "stale_claim_count" in data
        assert data["citation_statuses"] == []
        assert data["external_status_risk"] == 0.0
        assert data["risky_status_count"] == 0
        for claim in data["claims"]:
            assert "text" in claim
            assert "claim_type" in claim
            assert "staleness_risk" in claim
            assert "reason" in claim
            assert "source_id" in claim
            assert "external_status" in claim

    def test_empty_text_rejected(self, client: TestClient) -> None:
        """Empty text is refused with HTTP 422."""
        resp = client.post("/v1/temporal-freshness", json={"text": ""})
        assert resp.status_code == 422


# -- /v1/consensus ----------------------------------------------------------


class TestConsensus:
    """Tests for the cross-model consensus endpoint ``/v1/consensus``."""

    def test_high_agreement(self, client: TestClient) -> None:
        """Two identical answers reach full agreement over one pair."""
        resp = client.post(
            "/v1/consensus",
            json={
                "responses": [
                    {"model": "gpt-4o", "response": "Paris is the capital of France"},
                    {"model": "claude", "response": "Paris is the capital of France"},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_consensus"] is True
        assert data["agreement_score"] == 1.0
        assert data["num_models"] == 2
        assert len(data["pairs"]) == 1

    def test_disagreement(self, client: TestClient) -> None:
        """Two unrelated answers score below half agreement."""
        resp = client.post(
            "/v1/consensus",
            json={
                "responses": [
                    {"model": "a", "response": "The answer is forty two"},
                    {
                        "model": "b",
                        "response": "Bananas grow on trees in tropical regions",
                    },
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agreement_score"] < 0.5

    def test_three_models(self, client: TestClient) -> None:
        """Three answers are compared over all three pairs."""
        resp = client.post(
            "/v1/consensus",
            json={
                "responses": [
                    {"model": "a", "response": "Water boils at 100 degrees Celsius"},
                    {"model": "b", "response": "Water boils at 100 degrees Celsius"},
                    {"model": "c", "response": "Water boils at 100 degrees Celsius"},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_models"] == 3
        assert len(data["pairs"]) == 3

    def test_too_few_responses_rejected(self, client: TestClient) -> None:
        """A single response is refused with HTTP 422."""
        resp = client.post(
            "/v1/consensus",
            json={"responses": [{"model": "a", "response": "only one"}]},
        )
        assert resp.status_code == 422


# -- /v1/conformal/predict ---------------------------------------------------


class TestConformal:
    """Tests for the conformal prediction endpoint ``/v1/conformal/predict``."""

    def test_uncalibrated(self, client: TestClient) -> None:
        """Without calibration data the interval spans 0 to 1 around the inverted score."""
        resp = client.post("/v1/conformal/predict", json={"score": 0.8})
        assert resp.status_code == 200
        data = resp.json()
        assert data["point_estimate"] == pytest.approx(0.2, abs=0.01)
        assert data["lower"] == 0.0
        assert data["upper"] == 1.0
        assert data["is_reliable"] is False

    def test_calibrated(self, client: TestClient) -> None:
        """Thirty-six labelled samples give a reliable interval inside 0 to 1."""
        scores = [0.9, 0.85, 0.1, 0.15, 0.88, 0.12] * 6  # 36 samples
        labels = [False, False, True, True, False, True] * 6
        resp = client.post(
            "/v1/conformal/predict",
            json={
                "score": 0.7,
                "calibration_scores": scores,
                "calibration_labels": labels,
                "coverage": 0.9,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["calibration_size"] == 36
        assert data["is_reliable"] is True
        assert 0.0 <= data["lower"] <= data["upper"] <= 1.0

    def test_mismatched_lengths_rejected(self, client: TestClient) -> None:
        """Calibration scores and labels of different lengths are refused with HTTP 422."""
        resp = client.post(
            "/v1/conformal/predict",
            json={
                "score": 0.5,
                "calibration_scores": [0.9, 0.1],
                "calibration_labels": [False],
            },
        )
        assert resp.status_code == 422

    def test_score_out_of_range_rejected(self, client: TestClient) -> None:
        """A score above 1 is refused with HTTP 422."""
        resp = client.post("/v1/conformal/predict", json={"score": 1.5})
        assert resp.status_code == 422


# -- /v1/compliance/feedback-loops -------------------------------------------


class TestFeedbackLoops:
    """Tests for the feedback-loop endpoint ``/v1/compliance/feedback-loops``."""

    def test_no_loop(self, client: TestClient) -> None:
        """An input unrelated to the earlier outputs is not a loop."""
        resp = client.post(
            "/v1/compliance/feedback-loops",
            json={
                "input_text": "What is machine learning?",
                "previous_outputs": ["The weather is sunny today."],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["loop_detected"] is False

    def test_loop_detected(self, client: TestClient) -> None:
        """An input that repeats an earlier output is a loop with a severity."""
        output = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."
        resp = client.post(
            "/v1/compliance/feedback-loops",
            json={
                "input_text": output,
                "previous_outputs": [output],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["loop_detected"] is True
        assert data["severity"] in ("low", "medium", "high")
        assert data["similarity"] > 0.5

    def test_no_previous_outputs(self, client: TestClient) -> None:
        """Without earlier outputs no loop is reported."""
        resp = client.post(
            "/v1/compliance/feedback-loops",
            json={
                "input_text": "Some question to the AI system.",
                "previous_outputs": [],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["loop_detected"] is False

    def test_short_input_no_match(self, client: TestClient) -> None:
        """A very short repeated input is not reported as a loop."""
        resp = client.post(
            "/v1/compliance/feedback-loops",
            json={
                "input_text": "Hi",
                "previous_outputs": ["Hi"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["loop_detected"] is False


# -- /v1/agentic/check-step -------------------------------------------------


class TestAgenticCheckStep:
    """Tests for the agent step-check endpoint ``/v1/agentic/check-step``."""

    def test_normal_step(self, client: TestClient) -> None:
        """A first step towards the goal is numbered 1 and does not halt."""
        resp = client.post(
            "/v1/agentic/check-step",
            json={
                "goal": "Find quarterly revenue for Q3 2025",
                "action": "search_documents",
                "args": "revenue Q3 2025",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["step_number"] == 1
        assert data["should_halt"] is False

    def test_circular_detection(self, client: TestClient) -> None:
        """Six identical earlier steps make a repeat halt or warn, with a reason."""
        history = [{"action": "search", "args": "test"}] * 6
        resp = client.post(
            "/v1/agentic/check-step",
            json={
                "goal": "Find data",
                "action": "search",
                "args": "test",
                "step_history": history,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["should_halt"] is True or data["should_warn"] is True
        assert len(data["reasons"]) >= 1

    def test_step_limit(self, client: TestClient) -> None:
        """A step beyond the step limit halts."""
        history = [{"action": f"tool_{i}", "args": str(i)} for i in range(50)]
        resp = client.post(
            "/v1/agentic/check-step",
            json={
                "goal": "Find data",
                "action": "tool_50",
                "args": "50",
                "step_history": history,
                "max_steps": 50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["should_halt"] is True

    def test_response_fields(self, client: TestClient) -> None:
        """The response carries the step, halt, warning, reason, drift and budget fields."""
        resp = client.post(
            "/v1/agentic/check-step",
            json={
                "goal": "Test goal",
                "action": "test_action",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        for field in (
            "step_number",
            "should_halt",
            "should_warn",
            "reasons",
            "goal_drift_score",
            "budget_remaining_pct",
        ):
            assert field in data
