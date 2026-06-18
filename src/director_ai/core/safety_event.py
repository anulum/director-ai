# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SafetyEvent schema

"""Tenant-safe safety event records shared by runtime hooks."""

from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .types import HaltEvidence, HaltTraceAttribution

__all__ = [
    "SAFETY_EVENT_JSON_SCHEMA",
    "SAFETY_EVENT_SCHEMA_VERSION",
    "SafetyEvent",
    "new_safety_event_id",
    "utc_timestamp",
    "validate_safety_event_payload",
]

SAFETY_EVENT_SCHEMA_VERSION = "director.safety_event.v1"
_DECISIONS = frozenset({"allow", "warn", "halt", "block"})
_SCOPES = frozenset(
    {
        "streaming",
        "containment",
        "attestation",
        "ontology",
        "trajectory",
        "cyber_physical",
        "inference_server",
        "swarm",
        "agent",
    },
)
_BLOCKED_PARTS = (
    "credential",
    "image",
    "password",
    "private-key",
    "prompt",
    "raw",
    "secret",
    "sensor",
    "token",
)
_TENANT_SAFE_TEXT_ALLOWLIST = frozenset({"token-id", "token_id"})

SAFETY_EVENT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://anulum.github.io/director-ai/schemas/safety-event.schema.json",
    "title": "Director SafetyEvent",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "event_id",
        "timestamp",
        "request_id",
        "tenant_id",
        "hook_id",
        "hook_scope",
        "policy_decision",
        "halt_reason",
        "threshold",
        "observed_score",
        "latency_ms",
        "evidence_refs",
        "tenant_safe_explanation",
        "trace_attribution",
        "attributes",
    ],
    "properties": {
        "schema_version": {"const": SAFETY_EVENT_SCHEMA_VERSION},
        "event_id": {"type": "string", "pattern": "^sevt_[0-9a-f]{32}$|^sevt_.+"},
        "timestamp": {"type": "string", "minLength": 1},
        "request_id": {"type": "string"},
        "tenant_id": {"type": "string"},
        "hook_id": {"type": "string", "minLength": 1},
        "hook_scope": {"type": "string", "enum": sorted(_SCOPES)},
        "policy_decision": {"type": "string", "enum": sorted(_DECISIONS)},
        "halt_reason": {"type": "string", "minLength": 1},
        "threshold": {
            "anyOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {"type": "null"},
            ],
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "observed_score": {
            "anyOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {"type": "null"},
            ],
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "latency_ms": {
            "anyOf": [{"type": "number", "minimum": 0.0}, {"type": "null"}],
        },
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "tenant_safe_explanation": {"type": "string", "minLength": 1},
        "trace_attribution": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "fact_source",
                        "retrieval_path",
                        "scorer_path",
                        "token_offset",
                        "threshold",
                        "causal_contribution",
                    ],
                    "properties": {
                        "fact_source": {"type": "string"},
                        "retrieval_path": {"type": "string"},
                        "scorer_path": {"type": "string"},
                        "token_offset": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}]
                        },
                        "threshold": {
                            "anyOf": [
                                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                {"type": "null"},
                            ]
                        },
                        "causal_contribution": {"type": "number"},
                    },
                },
                {"type": "null"},
            ]
        },
        "attributes": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
    },
}


def utc_timestamp() -> str:
    """Return an RFC-3339 UTC timestamp for event records."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def new_safety_event_id() -> str:
    """Return an opaque ID for a safety event."""
    return f"sevt_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class SafetyEvent:
    """Single halt or policy decision emitted by a runtime hook."""

    event_id: str
    timestamp: str
    hook_id: str
    hook_scope: str
    policy_decision: str
    halt_reason: str
    tenant_safe_explanation: str
    schema_version: str = SAFETY_EVENT_SCHEMA_VERSION
    request_id: str = ""
    tenant_id: str = ""
    threshold: float | None = None
    observed_score: float | None = None
    latency_ms: float | None = None
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    trace_attribution: HaltTraceAttribution | None = None
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate event identity, scores, scope, decision, and safe metadata."""
        if self.schema_version != SAFETY_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported SafetyEvent schema_version")
        if not self.event_id.strip():
            raise ValueError("event_id is required")
        if not self.timestamp.strip():
            raise ValueError("timestamp is required")
        if not self.hook_id.strip():
            raise ValueError("hook_id is required")
        if self.hook_scope not in _SCOPES:
            raise ValueError(f"unsupported hook_scope {self.hook_scope!r}")
        if self.policy_decision not in _DECISIONS:
            raise ValueError(f"unsupported policy_decision {self.policy_decision!r}")
        if not self.halt_reason.strip():
            raise ValueError("halt_reason is required")
        if not self.tenant_safe_explanation.strip():
            raise ValueError("tenant_safe_explanation is required")
        _validate_unit_interval("threshold", self.threshold)
        _validate_unit_interval("observed_score", self.observed_score)
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        object.__setattr__(self, "evidence_refs", tuple(self.evidence_refs))
        object.__setattr__(
            self,
            "attributes",
            {str(key): str(value) for key, value in self.attributes.items()},
        )

    @classmethod
    def from_halt_evidence(
        cls,
        evidence: HaltEvidence,
        *,
        hook_id: str,
        hook_scope: str = "streaming",
        policy_decision: str = "halt",
        event_id: str | None = None,
        timestamp: str | None = None,
        request_id: str = "",
        tenant_id: str = "",
        latency_ms: float | None = None,
        attributes: dict[str, str] | None = None,
    ) -> SafetyEvent:
        """Build a tenant-safe event from structured halt evidence."""
        trace = evidence.trace_attribution
        return cls(
            event_id=event_id or new_safety_event_id(),
            timestamp=timestamp or utc_timestamp(),
            request_id=request_id,
            tenant_id=tenant_id,
            hook_id=hook_id,
            hook_scope=hook_scope,
            policy_decision=policy_decision,
            halt_reason=evidence.reason,
            threshold=trace.threshold if trace is not None else None,
            observed_score=evidence.last_score,
            latency_ms=latency_ms,
            evidence_refs=_evidence_refs(evidence),
            tenant_safe_explanation=(
                evidence.suggested_action or "Review the safety decision."
            ),
            trace_attribution=trace,
            attributes=attributes or {},
        )

    @classmethod
    def from_policy_decision(
        cls,
        *,
        hook_id: str,
        hook_scope: str,
        policy_decision: str,
        halt_reason: str,
        tenant_safe_explanation: str,
        event_id: str | None = None,
        timestamp: str | None = None,
        request_id: str = "",
        tenant_id: str = "",
        threshold: float | None = None,
        observed_score: float | None = None,
        latency_ms: float | None = None,
        evidence_refs: Sequence[str] = (),
        attributes: dict[str, str] | None = None,
    ) -> SafetyEvent:
        """Build a tenant-safe event from a hook policy decision."""
        return cls(
            event_id=event_id or new_safety_event_id(),
            timestamp=timestamp or utc_timestamp(),
            request_id=request_id,
            tenant_id=tenant_id,
            hook_id=hook_id,
            hook_scope=hook_scope,
            policy_decision=policy_decision,
            halt_reason=halt_reason,
            threshold=threshold,
            observed_score=observed_score,
            latency_ms=latency_ms,
            evidence_refs=tuple(evidence_refs),
            tenant_safe_explanation=tenant_safe_explanation,
            attributes=attributes or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict without raw prompt or fact text."""
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "hook_id": self.hook_id,
            "hook_scope": self.hook_scope,
            "policy_decision": self.policy_decision,
            "halt_reason": self.halt_reason,
            "threshold": self.threshold,
            "observed_score": self.observed_score,
            "latency_ms": self.latency_ms,
            "evidence_refs": list(self.evidence_refs),
            "tenant_safe_explanation": self.tenant_safe_explanation,
            "trace_attribution": _trace_to_dict(self.trace_attribution),
            "attributes": dict(self.attributes),
        }


def validate_safety_event_payload(payload: Mapping[str, Any]) -> SafetyEvent:
    """Validate and reconstruct a tenant-safe ``SafetyEvent`` payload.

    This is a lightweight stdlib validator for the exported JSON schema shape.
    It intentionally checks the deployment-critical constraints locally instead
    of requiring the optional ``jsonschema`` package at runtime.
    """
    allowed = set(SAFETY_EVENT_JSON_SCHEMA["properties"])
    required = set(SAFETY_EVENT_JSON_SCHEMA["required"])
    keys = set(payload)
    unknown = keys - allowed
    if unknown:
        raise ValueError(f"unknown field(s): {sorted(unknown)}")
    missing = required - keys
    if missing:
        raise ValueError(f"missing required field(s): {sorted(missing)}")
    _assert_type("schema_version", payload["schema_version"], str)
    _assert_type("event_id", payload["event_id"], str)
    _assert_type("timestamp", payload["timestamp"], str)
    _assert_type("request_id", payload["request_id"], str)
    _assert_type("tenant_id", payload["tenant_id"], str)
    _assert_type("hook_id", payload["hook_id"], str)
    _assert_type("hook_scope", payload["hook_scope"], str)
    _assert_type("policy_decision", payload["policy_decision"], str)
    _assert_type("halt_reason", payload["halt_reason"], str)
    _assert_type("tenant_safe_explanation", payload["tenant_safe_explanation"], str)
    threshold = _optional_float("threshold", payload["threshold"], unit=True)
    observed_score = _optional_float(
        "observed_score", payload["observed_score"], unit=True
    )
    latency_ms = _optional_float("latency_ms", payload["latency_ms"], unit=False)
    if latency_ms is not None and latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")
    evidence_refs = payload["evidence_refs"]
    if not isinstance(evidence_refs, list):
        raise ValueError("evidence_refs must be an array")
    for ref in evidence_refs:
        if not isinstance(ref, str):
            raise ValueError("evidence_refs entries must be strings")
        _assert_tenant_safe_text(ref, field_name="evidence_refs")
    attributes = payload["attributes"]
    if not isinstance(attributes, Mapping):
        raise ValueError("attributes must be an object")
    safe_attributes = _tenant_safe_mapping(attributes, field_name="attributes")
    trace_attribution = _trace_from_payload(payload["trace_attribution"])
    _assert_tenant_safe_text(
        payload["tenant_safe_explanation"],
        field_name="tenant_safe_explanation",
    )
    return SafetyEvent(
        schema_version=payload["schema_version"],
        event_id=payload["event_id"],
        timestamp=payload["timestamp"],
        request_id=payload["request_id"],
        tenant_id=payload["tenant_id"],
        hook_id=payload["hook_id"],
        hook_scope=payload["hook_scope"],
        policy_decision=payload["policy_decision"],
        halt_reason=payload["halt_reason"],
        threshold=threshold,
        observed_score=observed_score,
        latency_ms=latency_ms,
        evidence_refs=tuple(evidence_refs),
        trace_attribution=trace_attribution,
        tenant_safe_explanation=payload["tenant_safe_explanation"],
        attributes=safe_attributes,
    )


def _validate_unit_interval(name: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def _assert_type(name: str, value: Any, expected: type) -> None:
    if not isinstance(value, expected):
        raise ValueError(f"{name} must be {expected.__name__}")


def _optional_float(name: str, value: Any, *, unit: bool) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if unit:
        _validate_unit_interval(name, number)
    return number


def _tenant_safe_mapping(
    values: Mapping[str, Any],
    *,
    field_name: str,
) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in values.items():
        key_s = str(key)
        value_s = str(value)
        _assert_tenant_safe_text(key_s, field_name=field_name)
        _assert_tenant_safe_text(value_s, field_name=field_name)
        safe[key_s] = value_s
    return safe


def _trace_from_payload(value: Any) -> HaltTraceAttribution | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("trace_attribution must be an object or null")
    allowed = {
        "fact_source",
        "retrieval_path",
        "scorer_path",
        "token_offset",
        "threshold",
        "causal_contribution",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"trace_attribution unknown field(s): {sorted(unknown)}")
    missing = allowed - set(value)
    if missing:
        raise ValueError(f"trace_attribution missing field(s): {sorted(missing)}")
    _assert_type("trace_attribution.fact_source", value["fact_source"], str)
    _assert_type("trace_attribution.retrieval_path", value["retrieval_path"], str)
    _assert_type("trace_attribution.scorer_path", value["scorer_path"], str)
    token_offset_value = value["token_offset"]
    if token_offset_value is None:
        token_offset = -1
    elif isinstance(token_offset_value, bool) or not isinstance(
        token_offset_value, int
    ):
        raise ValueError("trace_attribution.token_offset must be integer or null")
    else:
        token_offset = token_offset_value
    threshold = _optional_float(
        "trace_attribution.threshold", value["threshold"], unit=True
    )
    causal = _optional_float(
        "trace_attribution.causal_contribution",
        value["causal_contribution"],
        unit=False,
    )
    if causal is None:
        raise ValueError("trace_attribution.causal_contribution must be a number")
    return HaltTraceAttribution(
        fact_source=value["fact_source"],
        retrieval_path=value["retrieval_path"],
        scorer_path=value["scorer_path"],
        token_offset=token_offset,
        threshold=threshold,
        causal_contribution=causal,
    )


def _assert_tenant_safe_text(value: str, *, field_name: str) -> None:
    lowered = value.lower().replace("_", "-")
    if lowered in _TENANT_SAFE_TEXT_ALLOWLIST:
        return
    if any(part in lowered for part in _BLOCKED_PARTS):
        raise ValueError(f"{field_name} must be tenant-safe")


def _evidence_refs(evidence: HaltEvidence) -> tuple[str, ...]:
    refs: list[str] = []
    for index, chunk in enumerate(evidence.evidence_chunks):
        refs.append(chunk.source or f"chunk:{index}")
    return tuple(refs)


def _trace_to_dict(trace: HaltTraceAttribution | None) -> dict[str, Any] | None:
    if trace is None:
        return None
    return {
        "fact_source": trace.fact_source,
        "retrieval_path": trace.retrieval_path,
        "scorer_path": trace.scorer_path,
        "token_offset": trace.token_offset,
        "threshold": trace.threshold,
        "causal_contribution": trace.causal_contribution,
    }
