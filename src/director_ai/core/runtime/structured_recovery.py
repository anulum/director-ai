# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Halt Recovery

"""Structured recovery checkpoints for halted token streams."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..verification.json_verifier import verify_json
from ..verification.reasoning_verifier import verify_reasoning_chain
from ..verification.tool_call_verifier import verify_tool_call

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
        if self.config.kind == "json":
            self._update_json(text)
        if self.config.kind == "tool_call":
            self._update_tool_call(text)
        if self.config.kind == "reasoning_chain":
            self._update_reasoning_chain(text)

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

    def _update_json(self, text: str) -> None:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            self._remember_error(f"json_parse:{detail}")
            return
        verdict = verify_json(text, schema=self.config.json_schema)
        if not verdict.valid_json:
            self._remember_error(f"json_parse:{verdict.parse_error}")
            return
        if verdict.schema_valid is False or verdict.error_count:
            self._remember_error("json_schema:invalid")
            return
        self.last_valid_output = json.dumps(
            data,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.last_valid_metadata = {
            "json_root": self._json_root_name(data),
            "field_errors": 0,
        }

    def _remember_error(self, message: str) -> None:
        if not self.errors or self.errors[-1] != message:
            self.errors.append(message)

    def _update_tool_call(self, text: str) -> None:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as exc:
            detail = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            self._remember_error(f"tool_call_parse:{detail}")
            return
        if not isinstance(data, dict):
            self._remember_error("tool_call:envelope_not_object")
            return
        function_name = data.get("function_name")
        arguments = data.get("arguments")
        if not isinstance(function_name, str) or not isinstance(arguments, dict):
            self._remember_error("tool_call:missing_function_name_or_arguments")
            return
        verdict = verify_tool_call(
            function_name=function_name,
            arguments=arguments,
            claimed_result=str(data.get("claimed_result", "")),
            manifest=self.config.tool_manifest,
            execution_log=(
                list(self.config.execution_log)
                if self.config.execution_log is not None
                else None
            ),
            score_fn=self.config.score_fn,
        )
        if (
            not verdict.function_exists
            or not verdict.arguments_valid
            or not verdict.result_plausible
            or verdict.fabrication_suspected
        ):
            self._remember_error(f"tool_call:invalid:{verdict.reason}")
            return
        self.last_valid_output = json.dumps(
            data,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.last_valid_metadata = {
            "function_name": function_name,
            "argument_count": len(arguments),
        }

    def _update_reasoning_chain(self, text: str) -> None:
        payload = text
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            steps = data["steps"]
            if not all(isinstance(step, str) and step.strip() for step in steps):
                self._remember_error("reasoning_chain:invalid_steps")
                return
            payload = "\n".join(
                f"{index + 1}. {step}" for index, step in enumerate(steps)
            )
        result = verify_reasoning_chain(
            payload,
            score_fn=self.config.score_fn,
            support_threshold=self.config.reasoning_support_threshold,
        )
        if result.steps_found < 2 or not result.chain_valid:
            self._remember_error("reasoning_chain:invalid")
            return
        self.last_valid_output = (
            json.dumps(data, separators=(",", ":"), sort_keys=True)
            if data is not None
            else text
        )
        self.last_valid_metadata = {
            "steps_found": result.steps_found,
            "issues_found": result.issues_found,
        }

    @staticmethod
    def _json_root_name(data: object) -> str:
        if isinstance(data, dict):
            return "object"
        if isinstance(data, list):
            return "array"
        return type(data).__name__
