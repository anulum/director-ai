# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Halt Recovery

"""Structured recovery checkpoints for halted token streams."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "StructuredRecoveryConfig",
    "StructuredRecoveryResult",
    "StructuredRecoveryState",
]

_KINDS = frozenset({"json", "tool_call", "reasoning_chain"})
_POLICIES = frozenset({"last_valid", "redacted", "raw_partial"})


@dataclass(frozen=True)
class StructuredRecoveryConfig:
    """Configuration for opt-in parser-safe recovery after stream halt."""

    kind: str
    policy: str = "last_valid"
    json_schema: dict[str, Any] | None = None
    tool_manifest: dict[str, Any] | None = None
    execution_log: Sequence[dict[str, Any]] | None = None
    score_fn: Callable[..., float] | None = None
    reasoning_support_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported structured_recovery kind {self.kind!r}")
        if self.policy not in _POLICIES:
            raise ValueError(f"unsupported structured_recovery policy {self.policy!r}")
        if self.reasoning_support_threshold < 0.0:
            raise ValueError("reasoning_support_threshold must be non-negative")


@dataclass(frozen=True)
class StructuredRecoveryResult:
    """Parser-safe structured recovery result attached to StreamSession."""

    kind: str
    policy: str
    halted_at: int
    last_valid_output: str = ""
    raw_partial: str = ""
    valid: bool = False
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StructuredRecoveryState:
    """Mutable structured checkpoint state for a single stream."""

    def __init__(self, config: StructuredRecoveryConfig) -> None:
        self.config = config
        self.raw_partial = ""
        self.last_valid_output = ""
        self.last_valid_metadata: dict[str, Any] = {}
        self.errors: list[str] = []

    def update(self, text: str) -> None:
        self.raw_partial = text

    def finalise(self, halted_at: int) -> StructuredRecoveryResult:
        return StructuredRecoveryResult(
            kind=self.config.kind,
            policy=self.config.policy,
            halted_at=halted_at,
            last_valid_output=(
                self.last_valid_output if self.config.policy == "last_valid" else ""
            ),
            raw_partial=self.raw_partial if self.config.policy == "raw_partial" else "",
            valid=bool(self.last_valid_output and self.config.policy == "last_valid"),
            errors=list(self.errors),
            metadata=dict(self.last_valid_metadata),
        )
