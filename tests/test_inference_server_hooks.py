# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - inference-server hook tests

from __future__ import annotations

import math
from typing import cast

import pytest

from director_ai import (
    InferenceHookRequest,
    InferenceServerHook,
    InferenceServerHookPolicy,
    build_inference_server_hook,
)
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
