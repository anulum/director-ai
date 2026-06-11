# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — trajectory rollback hooks

"""Tenant-safe rollback hooks for trajectory preflight decisions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .simulator import PreflightVerdict

RollbackStatus = Literal[
    "not_required",
    "armed",
    "executed",
    "already_executed",
    "failed",
]
RollbackHook = Callable[["RollbackHandle", str], Mapping[str, str] | None]

_BLOCKED_METADATA_PARTS = (
    "credential",
    "image",
    "password",
    "private_key",
    "prompt",
    "raw",
    "secret",
    "sensor",
    "token",
)


@dataclass(frozen=True)
class RollbackHandle:
    """Registered rollback hook for one proposed trajectory-controlled action."""

    rollback_id: str
    action_id: str
    tenant_id: str = ""
    evidence_refs: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rollback_id.strip():
            raise ValueError("rollback_id is required")
        if not self.action_id.strip():
            raise ValueError("action_id is required")
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))
        object.__setattr__(self, "metadata", _tenant_safe_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe handle without raw prompt or action payloads."""
        return {
            "rollback_id": self.rollback_id,
            "action_id": self.action_id,
            "tenant_id": self.tenant_id,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RollbackOutcome:
    """Result of arming or executing a rollback hook."""

    rollback_id: str
    action_id: str
    status: RollbackStatus
    reason: str
    executed: bool
    tenant_id: str = ""
    evidence_refs: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, str] = field(default_factory=dict)
    error_type: str = ""

    def __post_init__(self) -> None:
        if self.status not in {
            "not_required",
            "armed",
            "executed",
            "already_executed",
            "failed",
        }:
            raise ValueError(f"unsupported rollback status {self.status!r}")
        if not self.rollback_id.strip():
            raise ValueError("rollback_id is required")
        if not self.action_id.strip():
            raise ValueError("action_id is required")
        if not self.reason.strip():
            raise ValueError("reason is required")
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))
        object.__setattr__(self, "metadata", _tenant_safe_metadata(self.metadata))

    def to_dict(self) -> dict[str, object]:
        """Return a tenant-safe JSON representation."""
        payload: dict[str, object] = {
            "rollback_id": self.rollback_id,
            "action_id": self.action_id,
            "status": self.status,
            "reason": self.reason,
            "executed": self.executed,
            "tenant_id": self.tenant_id,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }
        if self.error_type:
            payload["error_type"] = self.error_type
        return payload


class TrajectoryRollbackManager:
    """Register and execute rollback hooks from trajectory decisions.

    Hooks are idempotent by rollback id: a successful rollback executes once,
    then later calls return ``already_executed``. The manager stores tenant-safe
    identifiers, evidence references, and metadata only; raw prompts, sampled
    tokens, action bodies, credentials, and sensor payloads should stay in the
    deployment's own protected system of record.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, tuple[RollbackHandle, RollbackHook]] = {}
        self._executed: set[str] = set()

    def register(
        self,
        *,
        rollback_id: str,
        action_id: str,
        hook: RollbackHook,
        tenant_id: str = "",
        evidence_refs: Sequence[str] = (),
        metadata: Mapping[str, str] | None = None,
    ) -> RollbackHandle:
        """Register a rollback hook and return its handle."""
        if rollback_id in self._hooks:
            raise ValueError(f"rollback_id {rollback_id!r} is already registered")
        handle = RollbackHandle(
            rollback_id=rollback_id,
            action_id=action_id,
            tenant_id=tenant_id,
            evidence_refs=evidence_refs,
            metadata=metadata or {},
        )
        self._hooks[rollback_id] = (handle, hook)
        return handle

    def evaluate_preflight(
        self,
        rollback_id: str,
        verdict: PreflightVerdict,
        *,
        steering_decision: object | None = None,
    ) -> RollbackOutcome:
        """Arm or execute rollback according to trajectory evidence."""
        action = str(getattr(steering_decision, "action", verdict.recommended))
        evidence_refs = _evidence_refs(verdict, steering_decision)
        if action == "halt" or verdict.recommended == "halt":
            return self.execute(
                rollback_id,
                reason="trajectory_preflight_halt",
                evidence_refs=evidence_refs,
            )
        if action == "escalate" or verdict.recommended == "warn":
            handle, _hook = self._lookup(rollback_id)
            return self._outcome(
                handle,
                status="armed",
                reason="trajectory_preflight_uncertain",
                executed=False,
                evidence_refs=evidence_refs,
            )
        handle, _hook = self._lookup(rollback_id)
        return self._outcome(
            handle,
            status="not_required",
            reason="trajectory_preflight_low_risk",
            executed=False,
            evidence_refs=evidence_refs,
        )

    def execute(
        self,
        rollback_id: str,
        *,
        reason: str,
        evidence_refs: Sequence[str] = (),
    ) -> RollbackOutcome:
        """Execute a registered rollback hook once."""
        handle, hook = self._lookup(rollback_id)
        combined_refs = _merge_refs(handle.evidence_refs, evidence_refs)
        if rollback_id in self._executed:
            return self._outcome(
                handle,
                status="already_executed",
                reason=reason,
                executed=False,
                evidence_refs=combined_refs,
            )
        try:
            metadata = hook(handle, reason) or {}
        except Exception as exc:  # pragma: no cover - exercised by tests
            return self._outcome(
                handle,
                status="failed",
                reason=reason,
                executed=False,
                evidence_refs=combined_refs,
                error_type=exc.__class__.__name__,
            )
        self._executed.add(rollback_id)
        return self._outcome(
            handle,
            status="executed",
            reason=reason,
            executed=True,
            evidence_refs=combined_refs,
            metadata=metadata,
        )

    def _lookup(self, rollback_id: str) -> tuple[RollbackHandle, RollbackHook]:
        try:
            return self._hooks[rollback_id]
        except KeyError as exc:
            raise KeyError(f"unknown rollback_id {rollback_id!r}") from exc

    @staticmethod
    def _outcome(
        handle: RollbackHandle,
        *,
        status: RollbackStatus,
        reason: str,
        executed: bool,
        evidence_refs: Sequence[str],
        metadata: Mapping[str, str] | None = None,
        error_type: str = "",
    ) -> RollbackOutcome:
        merged_metadata = dict(handle.metadata)
        merged_metadata.update(metadata or {})
        return RollbackOutcome(
            rollback_id=handle.rollback_id,
            action_id=handle.action_id,
            status=status,
            reason=reason,
            executed=executed,
            tenant_id=handle.tenant_id,
            evidence_refs=evidence_refs,
            metadata=merged_metadata,
            error_type=error_type,
        )


def _evidence_refs(
    verdict: PreflightVerdict,
    steering_decision: object | None,
) -> tuple[str, ...]:
    steering_refs = getattr(steering_decision, "evidence_refs", ())
    verdict_refs = tuple(
        f"trajectory:{trajectory.trajectory_id}"
        for trajectory in verdict.trajectories
        if not trajectory.approved
    )
    return _merge_refs(steering_refs, verdict_refs)


def _merge_refs(*groups: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for ref in group:
            ref_str = str(ref)
            if ref_str in seen:
                continue
            seen.add(ref_str)
            merged.append(ref_str)
    return tuple(merged)


def _tenant_safe_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in metadata.items():
        key_str = str(key)
        lowered = key_str.lower()
        if any(part in lowered for part in _BLOCKED_METADATA_PARTS):
            raise ValueError(f"metadata key {key_str!r} is not tenant-safe")
        safe[key_str] = str(value)
    return safe
