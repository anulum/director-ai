# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Halt Recovery Tests

from __future__ import annotations

import pytest

from director_ai.core.runtime.structured_recovery import (
    StructuredRecoveryConfig,
    StructuredRecoveryResult,
    StructuredRecoveryState,
)
from director_ai.core.streaming import StreamingKernel


def test_structured_recovery_config_accepts_supported_kinds_and_policies() -> None:
    config = StructuredRecoveryConfig(kind="json", policy="last_valid")

    assert config.kind == "json"
    assert config.policy == "last_valid"


@pytest.mark.parametrize("kind", ["xml", "", "JSON"])
def test_structured_recovery_config_rejects_unknown_kind(kind: str) -> None:
    with pytest.raises(ValueError, match="structured_recovery kind"):
        StructuredRecoveryConfig(kind=kind, policy="last_valid")


@pytest.mark.parametrize("policy", ["repair", "", "debug"])
def test_structured_recovery_config_rejects_unknown_policy(policy: str) -> None:
    with pytest.raises(ValueError, match="structured_recovery policy"):
        StructuredRecoveryConfig(kind="json", policy=policy)


def test_structured_recovery_result_defaults_are_parser_safe() -> None:
    result = StructuredRecoveryResult(
        kind="json",
        policy="redacted",
        halted_at=3,
    )

    assert result.last_valid_output == ""
    assert result.raw_partial == ""
    assert result.valid is False
    assert result.errors == []
    assert result.metadata == {}


def test_json_recovery_keeps_last_valid_object_before_malformed_suffix() -> None:
    state = StructuredRecoveryState(StructuredRecoveryConfig(kind="json"))
    state.update('{"status": "ok"}')
    state.update('{"status": "ok"}{"status": ')

    result = state.finalise(halted_at=2)

    assert result.halted_at == 2
    assert result.valid is True
    assert result.last_valid_output == '{"status":"ok"}'
    assert result.raw_partial == ""
    assert result.metadata["json_root"] == "object"


def test_json_recovery_keeps_last_valid_array_before_mid_element_halt() -> None:
    state = StructuredRecoveryState(StructuredRecoveryConfig(kind="json"))
    state.update('[{"id": 1}]')
    state.update('[{"id": 1}, {"id":')

    result = state.finalise(halted_at=4)

    assert result.valid is True
    assert result.last_valid_output == '[{"id":1}]'
    assert result.metadata["json_root"] == "array"


def test_json_schema_rejection_does_not_replace_previous_checkpoint() -> None:
    config = StructuredRecoveryConfig(
        kind="json",
        json_schema={
            "type": "object",
            "required": ["status"],
            "properties": {"status": {"type": "string"}},
        },
    )
    state = StructuredRecoveryState(config)
    state.update('{"status": "ok"}')
    state.update('{"status": 42}')

    result = state.finalise(halted_at=3)

    assert result.valid is True
    assert result.last_valid_output == '{"status":"ok"}'
    assert any("schema" in error for error in result.errors)


TOOL_MANIFEST = {
    "book_flight": {
        "parameters": {
            "route": {"type": "object"},
            "passengers": {"type": "integer"},
        },
        "returns": "booking confirmation",
    },
}


def test_tool_call_recovery_preserves_nested_arguments() -> None:
    state = StructuredRecoveryState(
        StructuredRecoveryConfig(
            kind="tool_call",
            tool_manifest=TOOL_MANIFEST,
        )
    )
    state.update(
        '{"function_name":"book_flight","arguments":{"route":{"from":"ZRH",'
        '"to":"PRG"},"passengers":2}}'
    )
    state.update('{"function_name":"book_flight","arguments":{"route":')

    result = state.finalise(halted_at=5)

    assert result.valid is True
    assert result.last_valid_output == (
        '{"arguments":{"passengers":2,"route":{"from":"ZRH","to":"PRG"}},'
        '"function_name":"book_flight"}'
    )
    assert result.metadata["function_name"] == "book_flight"


def test_tool_call_unknown_function_keeps_previous_valid_checkpoint() -> None:
    state = StructuredRecoveryState(
        StructuredRecoveryConfig(
            kind="tool_call",
            tool_manifest=TOOL_MANIFEST,
        )
    )
    state.update(
        '{"function_name":"book_flight","arguments":{"route":{"from":"ZRH",'
        '"to":"PRG"},"passengers":2}}'
    )
    state.update('{"function_name":"wire_money","arguments":{"amount":1000}}')

    result = state.finalise(halted_at=6)

    assert result.valid is True
    assert "book_flight" in result.last_valid_output
    assert any("tool_call" in error for error in result.errors)


def test_reasoning_chain_recovery_keeps_last_valid_envelope() -> None:
    state = StructuredRecoveryState(
        StructuredRecoveryConfig(
            kind="reasoning_chain",
            reasoning_support_threshold=0.8,
        )
    )
    state.update(
        '{"steps":["The sky is blue","Therefore the sky has a visible colour"]}'
    )
    state.update('{"steps":["The sky is blue","Therefore')

    result = state.finalise(halted_at=7)

    assert result.valid is True
    assert result.last_valid_output == (
        '{"steps":["The sky is blue","Therefore the sky has a visible colour"]}'
    )
    assert result.metadata["steps_found"] >= 2


def test_reasoning_chain_invalid_update_keeps_previous_checkpoint() -> None:
    state = StructuredRecoveryState(
        StructuredRecoveryConfig(
            kind="reasoning_chain",
            reasoning_support_threshold=0.8,
        )
    )
    state.update('{"steps":["A implies B","Therefore B follows from A"]}')
    state.update('{"steps":["A implies B"]}')

    result = state.finalise(halted_at=4)

    assert result.valid is True
    assert "Therefore B follows" in result.last_valid_output
    assert any("reasoning" in error for error in result.errors)


def test_streaming_json_recovery_attaches_last_valid_output_on_halt() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    config = StructuredRecoveryConfig(kind="json")
    scores = iter([0.8, 0.8, 0.2])

    session = kernel.stream_tokens(
        ['{"status":"ok"}', '{"status":', '"unsafe"'],
        lambda _text: next(scores),
        structured_recovery=config,
    )

    assert session.halted is True
    assert session.structured_recovery is not None
    assert session.structured_recovery.halted_at == session.halt_index
    assert session.structured_recovery.last_valid_output == '{"status":"ok"}'


def test_streaming_redacted_policy_suppresses_partial_payload_on_halt() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    config = StructuredRecoveryConfig(kind="json", policy="redacted")

    session = kernel.stream_tokens(
        ['{"status":"ok"}', '{"status":'],
        lambda text: 0.2 if text.endswith('{"status":') else 0.8,
        structured_recovery=config,
    )

    assert session.structured_recovery is not None
    assert session.structured_recovery.last_valid_output == ""
    assert session.structured_recovery.raw_partial == ""
    assert session.structured_recovery.valid is False


def test_streaming_raw_partial_policy_marks_payload_invalid() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    config = StructuredRecoveryConfig(kind="json", policy="raw_partial")

    session = kernel.stream_tokens(
        ['{"status":"ok"}', '{"status":'],
        lambda text: 0.2 if text.endswith('{"status":') else 0.8,
        structured_recovery=config,
    )

    assert session.structured_recovery is not None
    assert session.structured_recovery.last_valid_output == ""
    assert session.structured_recovery.raw_partial.endswith('{"status":')
    assert session.structured_recovery.valid is False


def test_unconfigured_streaming_recovery_preserves_plain_text_behaviour() -> None:
    kernel = StreamingKernel(hard_limit=0.5)
    session = kernel.stream_tokens(
        ["Good ", "Bad "],
        lambda text: 0.2 if "Bad" in text else 0.8,
    )

    assert session.halted is True
    assert session.structured_recovery is None
    assert session.output == "Good "
    assert kernel.stream_output(["Bad "], lambda _text: 0.2).startswith(
        "[KERNEL INTERRUPT:"
    )
