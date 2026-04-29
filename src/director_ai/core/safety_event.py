# SPDX-License-Identifier: AGPL-3.0-or-later
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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .types import HaltEvidence, HaltTraceAttribution

__all__ = [
    "SAFETY_EVENT_SCHEMA_VERSION",
    "SafetyEvent",
    "new_safety_event_id",
    "utc_timestamp",
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
        "swarm",
        "agent",
    },
)


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


def _validate_unit_interval(name: str, value: float | None) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


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
