# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — agent passport registry

"""Registry for signed agent passports, capabilities, and coherence history."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Literal

try:
    from backfire_kernel import rust_mean, rust_sum_f64

    _RUST_AGENT_IDENTITY = True
except Exception:  # pragma: no cover - optional dependency
    _RUST_AGENT_IDENTITY = False

    def rust_mean(_values: list[float]) -> float:
        raise RuntimeError("backfire_kernel rust_mean is unavailable")

    def rust_sum_f64(_values: list[float]) -> float:
        raise RuntimeError("backfire_kernel rust_sum_f64 is unavailable")

from director_ai.core.guard_control import (
    GuardDecision,
    NoGoPolicy,
    RiskEnvelope,
    VerifierSignal,
)

from .passport import AgentPassport, PassportSigner, PassportVerificationError

PassportRegistryReason = Literal[
    "passport_allowed",
    "passport_invalid",
    "passport_revoked",
    "capability_mismatch",
    "no_go_irreversible_risk",
    "no_go_threshold_exceeded",
]


@dataclass(frozen=True)
class PassportActionVerdict:
    """Decision returned by :class:`AgentPassportRegistry`."""

    accepted: bool
    reason: PassportRegistryReason
    guard_decision: GuardDecision
    detail: str = ""


@dataclass(frozen=True)
class CoherenceHistoryEntry:
    """Event-linked coherence summary for one registered agent."""

    event_ref: str
    coherence_score: float
    decision: str

    def __post_init__(self) -> None:
        if not self.event_ref.strip():
            raise ValueError("event_ref must be non-empty")
        _validate_unit_interval("coherence_score", self.coherence_score)
        if not self.decision.strip():
            raise ValueError("decision must be non-empty")

    def to_dict(self) -> dict[str, str | float]:
        """Return a tenant-safe representation."""
        return {
            "event_ref": self.event_ref,
            "coherence_score": self.coherence_score,
            "decision": self.decision,
        }


class AgentPassportRegistry:
    """Registry abstraction over passport signing, verification, and policy."""

    def __init__(
        self,
        *,
        signer: PassportSigner,
        no_go_policy: NoGoPolicy | None = None,
        history_limit: int = 256,
    ) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self._signer = signer
        self._no_go_policy = no_go_policy or NoGoPolicy(
            irreversible_threshold=0.1,
            default_threshold=0.95,
        )
        self._history_limit = history_limit
        self._lock = threading.Lock()
        self._passports: dict[str, AgentPassport] = {}
        self._revoked_signatures: dict[str, str] = {}
        self._coherence_history: dict[str, list[CoherenceHistoryEntry]] = {}

    def issue_passport(
        self,
        *,
        agent_id: str,
        role: str,
        tenant_id: str = "",
        capabilities: tuple[str, ...] = (),
        ttl_seconds: float | None = None,
    ) -> AgentPassport:
        """Issue, verify, and register a new signed passport."""
        passport = self._signer.issue(
            agent_id=agent_id,
            role=role,
            tenant_id=tenant_id,
            capabilities=capabilities,
            ttl_seconds=ttl_seconds,
        )
        self.register(passport)
        return passport

    def register(self, passport: AgentPassport) -> None:
        """Register an externally issued passport after signature verification."""
        self._signer.verify(passport)
        with self._lock:
            self._passports[passport.agent_id] = passport

    def revoke(self, passport: AgentPassport, *, reason: str) -> None:
        """Revoke one exact passport signature."""
        if not reason.strip():
            raise ValueError("reason must be non-empty")
        if not passport.signature:
            raise ValueError("passport signature is required for revocation")
        with self._lock:
            self._revoked_signatures[passport.signature] = reason

    def rotate_signer(self, *, new_active_key: bytes, new_active_key_id: str) -> None:
        """Rotate the underlying signer while preserving old-key verification."""
        self._signer.rotate(
            new_active_key=new_active_key,
            new_active_key_id=new_active_key_id,
        )

    def evaluate_action(
        self,
        *,
        passport: AgentPassport,
        required_capability: str,
        risk_envelope: RiskEnvelope,
        event_ref: str,
    ) -> PassportActionVerdict:
        """Verify identity and capability claims for one proposed action."""
        if not required_capability.strip():
            raise ValueError("required_capability must be non-empty")
        if not event_ref.strip():
            raise ValueError("event_ref must be non-empty")
        invalid_detail = self._verification_failure(passport)
        if invalid_detail:
            decision = _decision(
                decision="block",
                reason="passport_invalid",
                risk_score=1.0,
                explanation="The agent passport could not be verified.",
                risk_envelope=risk_envelope,
                evidence_ref=event_ref,
                capability=required_capability,
                agent_id=passport.agent_id,
                detail=invalid_detail,
            )
            return PassportActionVerdict(
                accepted=False,
                reason="passport_invalid",
                guard_decision=decision,
                detail=invalid_detail,
            )
        with self._lock:
            revoked_reason = self._revoked_signatures.get(passport.signature)
        if revoked_reason is not None:
            decision = _decision(
                decision="block",
                reason="passport_revoked",
                risk_score=1.0,
                explanation="The agent passport has been revoked.",
                risk_envelope=risk_envelope,
                evidence_ref=event_ref,
                capability=required_capability,
                agent_id=passport.agent_id,
                detail=revoked_reason,
            )
            return PassportActionVerdict(
                accepted=False,
                reason="passport_revoked",
                guard_decision=decision,
                detail=revoked_reason,
            )
        if required_capability not in passport.capabilities:
            decision = _decision(
                decision="block",
                reason="capability_mismatch",
                risk_score=1.0,
                explanation="The agent passport does not grant this capability.",
                risk_envelope=risk_envelope,
                evidence_ref=event_ref,
                capability=required_capability,
                agent_id=passport.agent_id,
                detail="capability not present",
            )
            return PassportActionVerdict(
                accepted=False,
                reason="capability_mismatch",
                guard_decision=decision,
                detail="capability not present",
            )
        base_decision = _decision(
            decision="allow",
            reason="passport_allowed",
            risk_score=0.1,
            explanation="The agent passport is valid for this capability.",
            risk_envelope=risk_envelope,
            evidence_ref=event_ref,
            capability=required_capability,
            agent_id=passport.agent_id,
            detail="",
        )
        no_go = self._no_go_policy.evaluate(base_decision)
        if no_go.decision == "block":
            blocked = _decision(
                decision="block",
                reason=no_go.reason,
                risk_score=max(base_decision.risk_score, 0.1),
                explanation="No-go policy blocked the passport-authorised action.",
                risk_envelope=risk_envelope,
                evidence_ref=event_ref,
                capability=required_capability,
                agent_id=passport.agent_id,
                detail=no_go.reason,
            )
            return PassportActionVerdict(
                accepted=False,
                reason=no_go.reason,  # type: ignore[arg-type]
                guard_decision=blocked,
                detail=no_go.reason,
            )
        return PassportActionVerdict(
            accepted=True,
            reason="passport_allowed",
            guard_decision=base_decision,
        )

    def record_coherence(
        self,
        *,
        agent_id: str,
        event_ref: str,
        coherence_score: float,
        decision: str,
    ) -> None:
        """Attach a tenant-safe coherence result to one agent."""
        if not agent_id.strip():
            raise ValueError("agent_id must be non-empty")
        entry = CoherenceHistoryEntry(
            event_ref=event_ref,
            coherence_score=coherence_score,
            decision=decision,
        )
        with self._lock:
            history = self._coherence_history.setdefault(agent_id, [])
            history.append(entry)
            if len(history) > self._history_limit:
                del history[: len(history) - self._history_limit]

    def export_agent(self, agent_id: str) -> dict[str, Any]:
        """Export a privacy-preserving audit summary for one agent."""
        if not agent_id.strip():
            raise ValueError("agent_id must be non-empty")
        with self._lock:
            passport = self._passports.get(agent_id)
            history = tuple(self._coherence_history.get(agent_id, ()))
            revoked = (
                passport is not None and passport.signature in self._revoked_signatures
            )
        if passport is None:
            raise KeyError(f"agent {agent_id!r} is not registered")
        scores = [entry.coherence_score for entry in history]
        return {
            "agent_id": passport.agent_id,
            "role": passport.role,
            "tenant_id": passport.tenant_id,
            "capabilities": list(passport.capabilities),
            "key_id": passport.key_id,
            "issued_at": passport.issued_at,
            "expires_at": passport.expires_at,
            "revoked": revoked,
            "coherence_history": [entry.to_dict() for entry in history],
            "coherence_summary": _coherence_summary(scores),
        }

    def _verification_failure(self, passport: AgentPassport) -> str:
        try:
            self._signer.verify(passport)
        except PassportVerificationError as exc:
            return str(exc)
        return ""


def _decision(
    *,
    decision: str,
    reason: str,
    risk_score: float,
    explanation: str,
    risk_envelope: RiskEnvelope,
    evidence_ref: str,
    capability: str,
    agent_id: str,
    detail: str,
) -> GuardDecision:
    signal = VerifierSignal(
        verifier="agent_identity.registry",
        modality="identity",
        score=risk_score,
        verdict=reason,
        confidence_low=0.9,
        confidence_high=1.0,
        evidence_refs=(evidence_ref,),
        failure_mode="" if decision == "allow" else reason,
    )
    attrs = {
        "agent_id": agent_id,
        "capability": capability,
    }
    if detail:
        attrs["detail"] = detail
    return GuardDecision(
        decision=decision,
        risk_score=risk_score,
        confidence_low=0.9,
        confidence_high=1.0,
        policy_id="policy.agent_passport.registry",
        reason=reason,
        tenant_safe_explanation=explanation,
        evidence_refs=(evidence_ref,),
        verifier_signals=(signal,),
        risk_envelope=risk_envelope,
        attributes=attrs,
    )


def _coherence_summary(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        return {
            "count": 0,
            "minimum": 0.0,
            "mean": 0.0,
            "latest": 0.0,
        }
    return {
        "count": len(scores),
        "minimum": min(scores),
        "mean": _mean_float(scores),
        "latest": scores[-1],
    }


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _mean_float(values: list[float]) -> float:
    if not values:
        return 0.0
    if _RUST_AGENT_IDENTITY:
        try:
            return float(rust_mean(values))
        except Exception:
            pass
    return _sum_float(values) / len(values)


def _sum_float(values: list[float]) -> float:
    try:
        return float(rust_sum_f64(values))
    except Exception:
        return sum(values)
