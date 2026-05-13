# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — safety protocol transport envelope

"""Interoperable, tenant-safe transport contract for guard signals."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from director_ai.core.safety_event import (
    SAFETY_EVENT_SCHEMA_VERSION,
    SafetyEvent,
    utc_timestamp,
    validate_safety_event_payload,
)

__all__ = [
    "DIRECTOR_SAFETY_PROTOCOL_VERSION",
    "DirectorSafetySignal",
    "director_safety_signal_from_event",
    "new_director_safety_signal_id",
    "validate_director_safety_signal",
]

DIRECTOR_SAFETY_PROTOCOL_VERSION = "director.safety_protocol.v1"
DIRECTOR_SAFETY_PROTOCOL_SCHEMA_REF = (
    "https://anulum.github.io/director-ai/api/director-safety-protocol/"
)
_SEVERITY_BY_DECISION = {
    "allow": "informational",
    "warn": "advisory",
    "halt": "terminal",
    "block": "terminal",
}
_ALLOWED_SEVERITIES = frozenset(_SEVERITY_BY_DECISION.values())
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


def new_director_safety_signal_id() -> str:
    """Return an opaque transport signal id."""
    return f"dsp_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class DirectorSafetySignal:
    """Protocol envelope for exchanging one tenant-safe guard signal."""

    signal_id: str
    emitted_at: str
    producer_id: str
    framework: str
    event: SafetyEvent
    protocol_version: str = DIRECTOR_SAFETY_PROTOCOL_VERSION
    schema_ref: str = DIRECTOR_SAFETY_PROTOCOL_SCHEMA_REF
    extensions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol_version != DIRECTOR_SAFETY_PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        if not self.signal_id.strip():
            raise ValueError("signal_id is required")
        if not self.emitted_at.strip():
            raise ValueError("emitted_at is required")
        if not self.producer_id.strip():
            raise ValueError("producer_id is required")
        if not self.framework.strip():
            raise ValueError("framework is required")
        _assert_tenant_safe_event(self.event)
        object.__setattr__(
            self,
            "extensions",
            _tenant_safe_mapping(self.extensions, field_name="extensions"),
        )

    def to_transport_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe transport payload."""
        event_payload = self.event.to_dict()
        return {
            "protocol_version": self.protocol_version,
            "schema_ref": self.schema_ref,
            "signal_id": self.signal_id,
            "emitted_at": self.emitted_at,
            "producer_id": self.producer_id,
            "framework": self.framework,
            "event_schema_version": SAFETY_EVENT_SCHEMA_VERSION,
            "event": event_payload,
            "interoperability": {
                "decision": self.event.policy_decision,
                "severity": _SEVERITY_BY_DECISION[self.event.policy_decision],
                "hook_scope": self.event.hook_scope,
                "halt_reason": self.event.halt_reason,
                "evidence_ref_count": len(self.event.evidence_refs),
            },
            "privacy": {
                "payload_classification": "tenant_safe",
                "raw_payload_included": False,
                "redaction_required": (
                    "raw_prompts",
                    "raw_completions",
                    "credentials",
                    "private_media",
                    "sensor_payloads",
                ),
            },
            "extensions": dict(self.extensions),
        }

    def to_json(self) -> str:
        """Serialize with deterministic key ordering for signatures and logs."""
        return json.dumps(
            self.to_transport_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )


def director_safety_signal_from_event(
    event: SafetyEvent,
    *,
    producer_id: str,
    framework: str = "generic",
    signal_id: str | None = None,
    emitted_at: str | None = None,
    extensions: Mapping[str, str] | None = None,
) -> DirectorSafetySignal:
    """Build a protocol envelope from an existing tenant-safe event."""
    return DirectorSafetySignal(
        signal_id=signal_id or new_director_safety_signal_id(),
        emitted_at=emitted_at or utc_timestamp(),
        producer_id=producer_id,
        framework=framework,
        event=event,
        extensions=extensions or {},
    )


def validate_director_safety_signal(
    payload: Mapping[str, Any],
) -> DirectorSafetySignal:
    """Validate and reconstruct a protocol signal from a transport payload."""
    protocol_version = str(payload.get("protocol_version", ""))
    if protocol_version != DIRECTOR_SAFETY_PROTOCOL_VERSION:
        raise ValueError("unsupported protocol_version")
    event_payload = payload.get("event")
    if not isinstance(event_payload, Mapping):
        raise ValueError("event payload is required")
    privacy = payload.get("privacy")
    if not isinstance(privacy, Mapping):
        raise ValueError("privacy payload is required")
    if privacy.get("payload_classification") != "tenant_safe":
        raise ValueError("privacy payload must be tenant-safe")
    if privacy.get("raw_payload_included") is not False:
        raise ValueError("raw payloads must not be included")
    interoperability = payload.get("interoperability")
    if not isinstance(interoperability, Mapping):
        raise ValueError("interoperability payload is required")
    severity = str(interoperability.get("severity", ""))
    if severity not in _ALLOWED_SEVERITIES:
        raise ValueError("unsupported severity")

    event = validate_safety_event_payload(event_payload)
    expected = _SEVERITY_BY_DECISION[event.policy_decision]
    if severity != expected:
        raise ValueError("severity does not match policy_decision")
    return DirectorSafetySignal(
        protocol_version=protocol_version,
        schema_ref=str(payload.get("schema_ref", "")),
        signal_id=str(payload.get("signal_id", "")),
        emitted_at=str(payload.get("emitted_at", "")),
        producer_id=str(payload.get("producer_id", "")),
        framework=str(payload.get("framework", "")),
        event=event,
        extensions=dict(payload.get("extensions", {})),
    )


def _assert_tenant_safe_event(event: SafetyEvent) -> None:
    _tenant_safe_mapping(event.attributes, field_name="event attributes")
    for ref in event.evidence_refs:
        _assert_safe_text(str(ref), field_name="evidence_refs")
    _assert_safe_text(event.tenant_safe_explanation, field_name="explanation")


def _tenant_safe_mapping(
    values: Mapping[str, str],
    *,
    field_name: str,
) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in values.items():
        key_s = str(key)
        value_s = str(value)
        _assert_safe_text(key_s, field_name=field_name)
        _assert_safe_text(value_s, field_name=field_name)
        safe[key_s] = value_s
    return safe


def _assert_safe_text(value: str, *, field_name: str) -> None:
    lowered = value.lower().replace("_", "-")
    if any(part in lowered for part in _BLOCKED_PARTS):
        raise ValueError(f"{field_name} must be tenant-safe")
