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
