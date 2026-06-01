# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Predictive pre-halt steering tests."""

from __future__ import annotations

import dataclasses
from typing import Literal, TypedDict

import pytest

from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.trajectory import (
    PredictivePreHaltSteering,
    PreflightVerdict,
    TrajectoryResult,
)


class SteeringConstructorKwargs(TypedDict, total=False):
    min_simulations: int
    uncertainty_margin: float


class VerdictProbabilityKwargs(TypedDict):
    halt_rate: float
    ci_low: float
    ci_high: float


def _envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="inference_steering",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.9,
    )


def _verdict(
    *,
    n: int = 8,
    halt_rate: float,
    ci_low: float,
    ci_high: float,
    recommended: Literal["proceed", "warn", "halt"] = "proceed",
) -> PreflightVerdict:
    trajectories = tuple(
        TrajectoryResult(
            trajectory_id=index,
            seed=17 + index,
            tokens=("candidate", str(index)),
            final_coherence=1.0 - halt_rate,
            approved=index >= round(halt_rate * n),
        )
        for index in range(n)
    )
    return PreflightVerdict(
        n_simulations=n,
        halt_rate=halt_rate,
        mean_coherence=1.0 - halt_rate,
        std_coherence=0.05,
        ci_low=ci_low,
        ci_high=ci_high,
        recommended=recommended,
        reason="synthetic trajectory verdict",
        trajectories=trajectories,
    )


def test_predictive_steering_halts_when_halt_probability_crosses_threshold():
    steering = PredictivePreHaltSteering(min_simulations=4)

    decision = steering.evaluate(
        _verdict(
            halt_rate=0.75,
            ci_low=0.61,
            ci_high=0.91,
            recommended="halt",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    assert decision.action == "halt"
    assert decision.guard_decision.decision == "halt"
    assert decision.guard_decision.risk_score == pytest.approx(0.75)
    assert decision.reason == "predictive_halt_threshold"
    assert decision.to_safety_event(hook_id="prehalt.steering").hook_scope == (
        "trajectory"
    )


def test_predictive_steering_escalates_when_uncertainty_crosses_threshold():
    steering = PredictivePreHaltSteering(min_simulations=4)

    decision = steering.evaluate(
        _verdict(
            halt_rate=0.22,
            ci_low=0.08,
            ci_high=0.68,
            recommended="proceed",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    assert decision.action == "escalate"
    assert decision.guard_decision.decision == "warn"
    assert decision.reason == "predictive_uncertainty"
    assert decision.recommended_backend == "strong_verifier"


def test_predictive_steering_escalates_insufficient_simulation_evidence():
    steering = PredictivePreHaltSteering(min_simulations=8)

    decision = steering.evaluate(
        _verdict(n=3, halt_rate=0.0, ci_low=0.0, ci_high=0.1),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    assert decision.action == "escalate"
    assert decision.guard_decision.reason == "predictive_insufficient_simulations"
    assert decision.guard_decision.decision == "warn"


def test_predictive_steering_proceeds_when_low_risk_and_stable():
    steering = PredictivePreHaltSteering(min_simulations=4)

    decision = steering.evaluate(
        _verdict(halt_rate=0.04, ci_low=0.01, ci_high=0.18),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    assert decision.action == "proceed"
    assert decision.guard_decision.decision == "allow"
    assert decision.recommended_backend == "current"


def test_predictive_steering_audit_payload_excludes_trajectory_text():
    steering = PredictivePreHaltSteering(min_simulations=4)

    decision = steering.evaluate(
        _verdict(halt_rate=0.04, ci_low=0.01, ci_high=0.18),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    payload = decision.to_dict()

    assert "candidate0" not in str(payload)
    assert "tokens" not in str(payload)
    assert payload["guard_decision"]["risk_envelope"]["action_category"] == (
        "inference_steering"
    )


def test_predictive_steering_escalates_when_preflight_warns():
    steering = PredictivePreHaltSteering(min_simulations=4)

    decision = steering.evaluate(
        _verdict(
            halt_rate=0.18,
            ci_low=0.05,
            ci_high=0.30,
            recommended="warn",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    assert decision.action == "escalate"
    assert decision.reason == "predictive_preflight_warning"
    assert decision.guard_decision.risk_score == pytest.approx(0.30)
    assert decision.guard_decision.attributes["recommended_backend"] == (
        "strong_verifier"
    )


def test_predictive_steering_evidence_refs_only_failed_trajectories():
    steering = PredictivePreHaltSteering(min_simulations=4)

    decision = steering.evaluate(
        _verdict(
            n=4,
            halt_rate=0.5,
            ci_low=0.2,
            ci_high=0.7,
            recommended="halt",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    assert decision.evidence_refs == ("trajectory:0", "trajectory:1")
    assert decision.to_dict()["evidence_refs"] == ["trajectory:0", "trajectory:1"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_simulations": 0},
        {"uncertainty_margin": -0.1},
        {"uncertainty_margin": float("inf")},
    ],
)
def test_predictive_steering_rejects_invalid_constructor_inputs(
    kwargs: SteeringConstructorKwargs,
):
    with pytest.raises(ValueError):
        PredictivePreHaltSteering(**kwargs)


@pytest.mark.parametrize(
    "verdict_kwargs",
    [
        {"halt_rate": -0.01, "ci_low": 0.0, "ci_high": 0.1},
        {"halt_rate": 0.1, "ci_low": -0.01, "ci_high": 0.1},
        {"halt_rate": 0.1, "ci_low": 0.0, "ci_high": 1.01},
    ],
)
def test_predictive_steering_rejects_out_of_range_probabilities(
    verdict_kwargs: VerdictProbabilityKwargs,
):
    steering = PredictivePreHaltSteering(min_simulations=4)

    with pytest.raises(ValueError):
        steering.evaluate(
            _verdict(**verdict_kwargs),
            risk_envelope=_envelope(),
            policy_id="policy.prehalt.regulated",
        )


def test_predictive_steering_rejects_blank_policy_id():
    steering = PredictivePreHaltSteering(min_simulations=4)

    with pytest.raises(ValueError, match="policy_id"):
        steering.evaluate(
            _verdict(halt_rate=0.1, ci_low=0.0, ci_high=0.2),
            risk_envelope=_envelope(),
            policy_id=" ",
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("action", "retry", "unsupported steering action"),
        ("reason", " ", "reason is required"),
        ("ci_low", 0.8, "ci_low must be <= ci_high"),
        ("recommended_backend", " ", "recommended_backend is required"),
    ],
)
def test_pre_halt_decision_rejects_invalid_audit_fields(
    field_name: str,
    value: object,
    message: str,
):
    steering = PredictivePreHaltSteering(min_simulations=4)
    decision = steering.evaluate(
        _verdict(halt_rate=0.04, ci_low=0.01, ci_high=0.18),
        risk_envelope=_envelope(),
        policy_id="policy.prehalt.regulated",
    )

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(decision, **{field_name: value})
