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
