# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — guard control tests

from __future__ import annotations

import pytest

from director_ai.core.guard_control import (
    GuardDecision,
    NoGoPolicy,
    RiskEnvelope,
    VerifierSignal,
)
from director_ai.core.safety_event import SafetyEvent


def test_guard_decision_serializes_to_tenant_safe_event_without_raw_payloads() -> None:
    signal = VerifierSignal(
        verifier="nli",
        modality="text",
        score=0.92,
        verdict="contradiction",
        confidence_low=0.87,
        confidence_high=0.96,
        evidence_refs=("kb://fact-7",),
        latency_ms=12.5,
        failure_mode="",
    )
    envelope = RiskEnvelope(
        action_category="text",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.7,
        no_go_threshold=0.9,
    )
    decision = GuardDecision(
        decision="halt",
        risk_score=0.92,
        confidence_low=0.87,
        confidence_high=0.96,
        policy_id="policy.fact",
        reason="factual_contradiction",
        tenant_safe_explanation="The answer conflicts with verified evidence.",
        evidence_refs=("kb://fact-7",),
        verifier_signals=(signal,),
        risk_envelope=envelope,
        attributes={"raw_prompt": "must-not-appear", "safe_key": "safe-value"},
    )

    payload = decision.to_dict()
    rendered = repr(payload)

    assert payload["decision"] == "halt"
    assert payload["risk_score"] == pytest.approx(0.92)
    assert payload["risk_envelope"]["domain"] == "regulated"
    assert payload["verifier_signals"][0]["confidence_high"] == pytest.approx(0.96)
    assert "must-not-appear" not in rendered
    assert "raw_prompt" not in rendered

    event = decision.to_safety_event(hook_id="guard.control", hook_scope="agent")

    assert isinstance(event, SafetyEvent)
    assert event.policy_decision == "halt"
    assert event.observed_score == pytest.approx(0.92)
    assert event.evidence_refs == ("kb://fact-7",)
    assert event.attributes["policy_id"] == "policy.fact"


def test_no_go_policy_blocks_irreversible_high_risk_actions() -> None:
    policy = NoGoPolicy(
        default_threshold=0.8,
        irreversible_threshold=0.5,
        require_human_review_for_irreversible=True,
    )
    decision = GuardDecision(
        decision="warn",
        risk_score=0.62,
        confidence_low=0.55,
        confidence_high=0.78,
        policy_id="policy.physical",
        reason="trajectory_uncertain",
        tenant_safe_explanation="The physical action has unresolved risk.",
        evidence_refs=("physical:trajectory",),
        verifier_signals=(),
        risk_envelope=RiskEnvelope(
            action_category="physical",
            reversibility="irreversible",
            domain="physical",
            calibrated_threshold=0.7,
            no_go_threshold=0.6,
        ),
    )

    verdict = policy.evaluate(decision)

    assert verdict.decision == "block"
    assert verdict.reason == "no_go_irreversible_risk"
    assert verdict.requires_human_review is True
    assert verdict.original_decision == decision


def test_guard_control_rejects_invalid_scores_and_impossible_intervals() -> None:
    with pytest.raises(ValueError, match="risk_score"):
        GuardDecision(
            decision="allow",
            risk_score=1.2,
            confidence_low=0.0,
            confidence_high=1.0,
            policy_id="policy.invalid",
            reason="invalid",
            tenant_safe_explanation="invalid",
            evidence_refs=(),
            verifier_signals=(),
            risk_envelope=RiskEnvelope(
                action_category="text",
                reversibility="reversible",
                domain="general",
                calibrated_threshold=0.5,
                no_go_threshold=0.9,
            ),
        )

    with pytest.raises(ValueError, match="confidence_low"):
        VerifierSignal(
            verifier="numeric",
            modality="text",
            score=0.4,
            verdict="unsupported",
            confidence_low=0.8,
            confidence_high=0.7,
            evidence_refs=(),
            latency_ms=0.0,
            failure_mode="",
        )
