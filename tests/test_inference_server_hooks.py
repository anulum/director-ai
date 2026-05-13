# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - inference-server hook tests

from __future__ import annotations

import math
from typing import Literal, cast

import pytest

from director_ai import (
    InferenceHookRequest,
    InferenceServerHook,
    InferenceServerHookPolicy,
    build_inference_server_hook,
)
from director_ai.core.guard_control import GuardDecision, RiskEnvelope
from director_ai.core.trajectory import PreHaltSteeringDecision
from director_ai.integrations.inference_server_hooks import InferenceServerName


def test_allow_decision_preserves_logits() -> None:
    hook = InferenceServerHook("vllm", lambda text: 0.91)
    request = InferenceHookRequest(
        server="vllm",
        accumulated_text="known fact: ",
        candidate_token="grounded",
        token_id=2,
    )

    decision = hook.check(request, logits=[0.1, 0.2, 0.3])

    assert decision.allow is True
    assert decision.score == pytest.approx(0.91)
    assert decision.adjusted_logits == (0.1, 0.2, 0.3)
    assert decision.blocked_token_ids == ()
    assert decision.safety_event is None
    assert decision.server_payload == {
        "server": "vllm",
        "action": "allow",
        "allow": True,
        "score": pytest.approx(0.91),
    }


def test_halt_masks_candidate_token_and_emits_event() -> None:
    hook = InferenceServerHook(
        "tgi",
        lambda text: 0.19,
        InferenceServerHookPolicy(hard_limit=0.4, block_logit=-999.0),
    )
    request = InferenceHookRequest(
        server="tgi",
        accumulated_text="mass of electron is ",
        candidate_token="ten kilograms",
        token_id=1,
        request_id="req-1",
        tenant_id="tenant-a",
        metadata={"profile": "stem"},
    )

    decision = hook.check(request, logits=[4.0, 5.0, 6.0])

    assert decision.allow is False
    assert decision.reason == "coherence_below_threshold"
    assert decision.adjusted_logits == (4.0, -999.0, 6.0)
    assert decision.blocked_token_ids == (1,)
    assert decision.server_payload == {
        "server": "tgi",
        "allow": False,
        "score": pytest.approx(0.19),
        "action": "filter_next_token",
        "token_ids": [1],
    }
    assert decision.safety_event is not None
    assert decision.safety_event.hook_scope == "inference_server"
    assert decision.safety_event.policy_decision == "halt"
    assert decision.safety_event.request_id == "req-1"
    assert decision.safety_event.tenant_id == "tenant-a"
    assert decision.safety_event.attributes == {
        "server": "tgi",
        "token_id": "1",
        "profile": "stem",
    }


def test_predictive_pre_halt_escalation_biases_candidate_before_sampling() -> None:
    hook = InferenceServerHook(
        "vllm",
        lambda text: 0.99,
        InferenceServerHookPolicy(steering_bias_logit=-6.5),
    )
    request = InferenceHookRequest(
        server="vllm",
        accumulated_text="measurement result: ",
        candidate_token="unsupported",
        token_id=2,
        request_id="req-prehalt",
        tenant_id="tenant-a",
        metadata={"risk_profile": "regulated"},
    )

    decision = hook.steer(
        request,
        _steering_decision(action="escalate", guard_decision="warn"),
        logits=[0.4, 0.8, 1.2, 1.6],
    )

    assert decision.allow is True
    assert decision.score == pytest.approx(0.62)
    assert decision.reason == "predictive_uncertainty"
    assert decision.adjusted_logits == (0.4, 0.8, -6.5, 1.6)
    assert decision.blocked_token_ids == ()
    assert decision.server_payload == {
        "server": "vllm",
        "allow": True,
        "score": pytest.approx(0.62),
        "action": "bias_token",
        "token_biases": {2: -6.5},
    }
    assert decision.safety_event is not None
    assert decision.safety_event.hook_id == "inference_server.vllm.prehalt"
    assert decision.safety_event.hook_scope == "inference_server"
    assert decision.safety_event.policy_decision == "warn"
    assert decision.safety_event.evidence_refs == ("trajectory:7",)
    assert decision.safety_event.attributes["server"] == "vllm"
    assert decision.safety_event.attributes["token_id"] == "2"
    assert decision.safety_event.attributes["steering_action"] == "escalate"
    assert decision.safety_event.attributes["risk_profile"] == "regulated"


def test_predictive_pre_halt_halt_uses_existing_block_path() -> None:
    hook = InferenceServerHook(
        "llama_cpp",
        lambda text: 0.99,
        InferenceServerHookPolicy(block_logit=-1234.0),
    )
    request = InferenceHookRequest(
        server="llama_cpp",
        accumulated_text="",
        candidate_token="unsupported",
        token_id=1,
    )

    decision = hook.steer(
        request,
        _steering_decision(action="halt", guard_decision="halt", risk_score=0.91),
        logits=[2.0, 4.0, 6.0],
    )

    assert decision.allow is False
    assert decision.score == pytest.approx(0.91)
    assert decision.adjusted_logits == (2.0, -1234.0, 6.0)
    assert decision.blocked_token_ids == (1,)
    assert decision.server_payload["action"] == "logit_bias"
    assert decision.server_payload["bias"] == -1234.0
    assert decision.safety_event is not None
    assert decision.safety_event.policy_decision == "halt"


@pytest.mark.parametrize(
    ("server", "action"),
    [
        ("vllm", "mask_token"),
        ("tgi", "filter_next_token"),
        ("llama_cpp", "logit_bias"),
    ],
)
def test_server_payloads_use_target_action_names(server: str, action: str) -> None:
    hook = build_inference_server_hook(
        cast(InferenceServerName, server),
        lambda text: 0.0,
        hard_limit=0.5,
        block_token_id=3,
        block_logit=-77.0,
    )
    request = InferenceHookRequest(
        server=cast(InferenceServerName, server),
        accumulated_text="",
        candidate_token="x",
    )

    decision = hook.check(request, logits=[1.0, 2.0, 3.0, 4.0])

    assert decision.server_payload["action"] == action
    assert decision.adjusted_logits == (1.0, 2.0, 3.0, -77.0)
    assert decision.blocked_token_ids == (3,)
    if server == "llama_cpp":
        assert decision.server_payload["token_id"] == 3
        assert decision.server_payload["bias"] == -77.0
    else:
        assert decision.server_payload["token_ids"] == [3]


def test_halt_without_token_id_still_returns_event() -> None:
    hook = InferenceServerHook("llama_cpp", lambda text: -math.inf)
    request = InferenceHookRequest(
        server="llama_cpp",
        accumulated_text="",
        candidate_token="ungrounded",
    )

    decision = hook.check(request)

    assert decision.allow is False
    assert decision.score == 0.0
    assert decision.adjusted_logits is None
    assert decision.blocked_token_ids == ()
    assert decision.safety_event is not None
    assert decision.server_payload == {
        "server": "llama_cpp",
        "allow": False,
        "score": 0.0,
        "action": "logit_bias",
    }


def test_invalid_server_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported inference server"):
        InferenceHookRequest(
            server=cast(InferenceServerName, "unknown"),
            accumulated_text="",
            candidate_token="x",
        )


def test_request_server_must_match_hook_server() -> None:
    hook = InferenceServerHook("vllm", lambda text: 1.0)
    request = InferenceHookRequest(
        server="tgi",
        accumulated_text="",
        candidate_token="x",
    )

    with pytest.raises(ValueError, match="does not match hook"):
        hook.check(request)


def test_score_is_clamped_to_unit_interval() -> None:
    hook = InferenceServerHook("vllm", lambda text: math.inf)
    request = InferenceHookRequest(
        server="vllm",
        accumulated_text="",
        candidate_token="x",
    )

    decision = hook.check(request)

    assert decision.allow is True
    assert decision.score == 1.0


def _steering_decision(
    *,
    action: Literal["proceed", "escalate", "halt"],
    guard_decision: Literal["allow", "warn", "halt"],
    risk_score: float = 0.62,
) -> PreHaltSteeringDecision:
    envelope = RiskEnvelope(
        action_category="inference_steering",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.9,
    )
    guard = GuardDecision(
        decision=guard_decision,
        risk_score=risk_score,
        confidence_low=0.33,
        confidence_high=0.74,
        policy_id="policy.prehalt.regulated",
        reason="predictive_uncertainty"
        if action == "escalate"
        else "predictive_halt_threshold",
        tenant_safe_explanation="Trajectory risk crosses the policy threshold.",
        evidence_refs=("trajectory:7",),
        verifier_signals=(),
        risk_envelope=envelope,
        attributes={
            "steering_action": action,
            "recommended_backend": "review_lane",
            "halt_probability": f"{risk_score:.6f}",
            "ci_low": "0.330000",
            "ci_high": "0.740000",
        },
    )
    return PreHaltSteeringDecision(
        action=action,
        reason=guard.reason,
        halt_probability=risk_score,
        ci_low=0.33,
        ci_high=0.74,
        recommended_backend="review_lane",
        evidence_refs=("trajectory:7",),
        guard_decision=guard,
    )
