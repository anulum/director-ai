# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — guard decision tests

from __future__ import annotations

import math

import pytest

from director_ai.core.guard_control.decision import (
    GuardDecision,
    RiskEnvelope,
    VerifierSignal,
)


def _risk_envelope(**overrides) -> RiskEnvelope:
    values = {
        "action_category": "tool",
        "reversibility": "costly",
        "domain": "security",
        "calibrated_threshold": 0.6,
        "no_go_threshold": 0.8,
    }
    values.update(overrides)
    return RiskEnvelope(**values)


def _verifier_signal(**overrides) -> VerifierSignal:
    values = {
        "verifier": "nli",
        "modality": "text",
        "score": 0.7,
        "verdict": "supported",
        "confidence_low": 0.6,
        "confidence_high": 0.8,
        "evidence_refs": ("kb://fact",),
        "latency_ms": 3.5,
        "failure_mode": "",
    }
    values.update(overrides)
    return VerifierSignal(**values)


def _guard_decision(**overrides) -> GuardDecision:
    values = {
        "decision": "warn",
        "risk_score": 0.7,
        "confidence_low": 0.6,
        "confidence_high": 0.8,
        "policy_id": "policy.guard",
        "reason": "operator_review",
        "tenant_safe_explanation": "The operation needs review.",
        "evidence_refs": ("kb://fact",),
        "verifier_signals": (_verifier_signal(),),
        "risk_envelope": _risk_envelope(),
        "attributes": {"safe_key": "safe-value", "raw_prompt": "discarded"},
    }
    values.update(overrides)
    return GuardDecision(**values)


def test_risk_envelope_rejects_unsupported_classification_values() -> None:
    with pytest.raises(ValueError, match="action_category"):
        _risk_envelope(action_category="unsupported")
    with pytest.raises(ValueError, match="reversibility"):
        _risk_envelope(reversibility="unknown")
    with pytest.raises(ValueError, match="domain"):
        _risk_envelope(domain="unlisted")


def test_risk_envelope_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="calibrated_threshold"):
        _risk_envelope(calibrated_threshold=math.inf)
    with pytest.raises(ValueError, match="no_go_threshold"):
        _risk_envelope(no_go_threshold=-0.1)


def test_verifier_signal_validates_required_fields_and_latency() -> None:
    with pytest.raises(ValueError, match="verifier is required"):
        _verifier_signal(verifier=" ")
    with pytest.raises(ValueError, match="modality"):
        _verifier_signal(modality="unsupported")
    with pytest.raises(ValueError, match="verdict is required"):
        _verifier_signal(verdict=" ")
    with pytest.raises(ValueError, match="latency_ms"):
        _verifier_signal(latency_ms=-1.0)


def test_verifier_signal_serialises_tuple_refs_and_failure_mode() -> None:
    signal = _verifier_signal(evidence_refs=[7, "kb://fact"], failure_mode="timeout")

    payload = signal.to_dict()

    assert signal.evidence_refs == ("7", "kb://fact")
    assert payload["evidence_refs"] == ["7", "kb://fact"]
    assert payload["failure_mode"] == "timeout"


def test_guard_decision_rejects_invalid_decision_and_required_text() -> None:
    with pytest.raises(ValueError, match="decision"):
        _guard_decision(decision="defer")
    with pytest.raises(ValueError, match="policy_id"):
        _guard_decision(policy_id=" ")
    with pytest.raises(ValueError, match="reason"):
        _guard_decision(reason=" ")
    with pytest.raises(ValueError, match="tenant_safe_explanation"):
        _guard_decision(tenant_safe_explanation=" ")


def test_guard_decision_filters_sensitive_attribute_keys() -> None:
    decision = _guard_decision(
        attributes={
            "safe": "kept",
            "raw_prompt": "discard",
            "credential_hint": "discard",
            "TOKEN_ID": "discard",
        }
    )

    assert decision.attributes == {"safe": "kept"}
    payload = decision.to_dict()
    assert payload["attributes"] == {"safe": "kept"}
    assert payload["verifier_signals"][0]["verifier"] == "nli"


def test_guard_decision_to_safety_event_preserves_tenant_safe_context() -> None:
    decision = _guard_decision(attributes={"workflow": "release-gate"})

    event = decision.to_safety_event(
        hook_id="guard.control",
        hook_scope="agent",
        request_id="request-1",
        tenant_id="tenant-a",
        latency_ms=12.5,
    )

    assert event.policy_decision == "warn"
    assert event.halt_reason == "operator_review"
    assert event.threshold == pytest.approx(0.6)
    assert event.observed_score == pytest.approx(0.7)
    assert event.request_id == "request-1"
    assert event.tenant_id == "tenant-a"
    assert event.latency_ms == pytest.approx(12.5)
    assert event.attributes["policy_id"] == "policy.guard"
    assert event.attributes["risk_domain"] == "security"
    assert event.attributes["action_category"] == "tool"
    assert event.attributes["reversibility"] == "costly"
    assert event.attributes["workflow"] == "release-gate"
