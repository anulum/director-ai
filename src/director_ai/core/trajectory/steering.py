# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Predictive Pre-Halt Steering

"""CI-aware steering decisions before a runtime halt becomes likely."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from director_ai.core.guard_control import GuardDecision, RiskEnvelope
from director_ai.core.safety_event import SafetyEvent

from .simulator import PreflightVerdict, TrajectoryResult

__all__ = ["PreHaltSteeringDecision", "PredictivePreHaltSteering"]

SteeringAction = Literal["proceed", "escalate", "halt"]


@dataclass(frozen=True)
class PreHaltSteeringDecision:
    """Tenant-safe pre-halt steering decision derived from trajectory evidence."""

    action: SteeringAction
    reason: str
    halt_probability: float
    ci_low: float
    ci_high: float
    recommended_backend: str
    evidence_refs: Sequence[str]
    guard_decision: GuardDecision

    def __post_init__(self) -> None:
        if self.action not in {"proceed", "escalate", "halt"}:
            raise ValueError(f"unsupported steering action {self.action!r}")
        if not self.reason.strip():
            raise ValueError("reason is required")
        _validate_unit_interval("halt_probability", self.halt_probability)
        _validate_unit_interval("ci_low", self.ci_low)
        _validate_unit_interval("ci_high", self.ci_high)
        if self.ci_low > self.ci_high:
            raise ValueError("ci_low must be <= ci_high")
        if not self.recommended_backend.strip():
            raise ValueError("recommended_backend is required")
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))

    def to_dict(self) -> dict[str, Any]:
        """Serialise without trajectory token text or prompt payloads."""
        return {
            "action": self.action,
            "reason": self.reason,
            "halt_probability": self.halt_probability,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "recommended_backend": self.recommended_backend,
            "evidence_refs": list(self.evidence_refs),
            "guard_decision": self.guard_decision.to_dict(),
        }

    def to_safety_event(
        self,
        *,
        hook_id: str,
        hook_scope: str = "trajectory",
        request_id: str = "",
        tenant_id: str = "",
        latency_ms: float | None = None,
    ) -> SafetyEvent:
        """Convert the steering decision to the shared safety-event schema."""
        return self.guard_decision.to_safety_event(
            hook_id=hook_id,
            hook_scope=hook_scope,
            request_id=request_id,
            tenant_id=tenant_id,
            latency_ms=latency_ms,
        )


class PredictivePreHaltSteering:
    """Escalate or halt requests using calibrated trajectory risk evidence."""

    def __init__(
        self,
        *,
        min_simulations: int = 8,
        uncertainty_margin: float = 0.0,
    ) -> None:
        if min_simulations <= 0:
            raise ValueError("min_simulations must be positive")
        if uncertainty_margin < 0.0 or not math.isfinite(uncertainty_margin):
            raise ValueError("uncertainty_margin must be finite and non-negative")
        self._min_simulations = min_simulations
        self._uncertainty_margin = uncertainty_margin

    def evaluate(
        self,
        verdict: PreflightVerdict,
        *,
        risk_envelope: RiskEnvelope,
        policy_id: str,
    ) -> PreHaltSteeringDecision:
        """Return a pre-halt steering decision from a trajectory verdict."""
        if not policy_id.strip():
            raise ValueError("policy_id is required")
        halt_probability = _clamp_probability(verdict.halt_rate)
        ci_low = _clamp_probability(verdict.ci_low)
        ci_high = _clamp_probability(verdict.ci_high)
        threshold = risk_envelope.calibrated_threshold
        evidence_refs = _failed_trajectory_refs(verdict.trajectories)

        if verdict.n_simulations < self._min_simulations:
            return self._decision(
                action="escalate",
                reason="predictive_insufficient_simulations",
                backend="strong_verifier",
                risk_score=ci_high,
                halt_probability=halt_probability,
                ci_low=ci_low,
                ci_high=ci_high,
                evidence_refs=evidence_refs,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
            )
        if halt_probability >= threshold or verdict.recommended == "halt":
            return self._decision(
                action="halt",
                reason="predictive_halt_threshold",
                backend="manual_review",
                risk_score=halt_probability,
                halt_probability=halt_probability,
                ci_low=ci_low,
                ci_high=ci_high,
                evidence_refs=evidence_refs,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
            )
        if ci_high >= threshold + self._uncertainty_margin:
            return self._decision(
                action="escalate",
                reason="predictive_uncertainty",
                backend="strong_verifier",
                risk_score=ci_high,
                halt_probability=halt_probability,
                ci_low=ci_low,
                ci_high=ci_high,
                evidence_refs=evidence_refs,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
            )
        if verdict.recommended == "warn":
            return self._decision(
                action="escalate",
                reason="predictive_preflight_warning",
                backend="strong_verifier",
                risk_score=max(halt_probability, ci_high),
                halt_probability=halt_probability,
                ci_low=ci_low,
                ci_high=ci_high,
                evidence_refs=evidence_refs,
                risk_envelope=risk_envelope,
                policy_id=policy_id,
            )
        return self._decision(
            action="proceed",
            reason="predictive_low_risk",
            backend="current",
            risk_score=halt_probability,
            halt_probability=halt_probability,
            ci_low=ci_low,
            ci_high=ci_high,
            evidence_refs=evidence_refs,
            risk_envelope=risk_envelope,
            policy_id=policy_id,
        )

    def _decision(
        self,
        *,
        action: SteeringAction,
        reason: str,
        backend: str,
        risk_score: float,
        halt_probability: float,
        ci_low: float,
        ci_high: float,
        evidence_refs: tuple[str, ...],
        risk_envelope: RiskEnvelope,
        policy_id: str,
    ) -> PreHaltSteeringDecision:
        guard_decision = GuardDecision(
            decision=_policy_decision(action),
            risk_score=_clamp_probability(risk_score),
            confidence_low=ci_low,
            confidence_high=ci_high,
            policy_id=policy_id,
            reason=reason,
            tenant_safe_explanation=_explanation(action, reason),
            evidence_refs=evidence_refs,
            verifier_signals=(),
            risk_envelope=risk_envelope,
            attributes={
                "steering_action": action,
                "recommended_backend": backend,
                "halt_probability": f"{halt_probability:.6f}",
                "ci_low": f"{ci_low:.6f}",
                "ci_high": f"{ci_high:.6f}",
            },
        )
        return PreHaltSteeringDecision(
            action=action,
            reason=reason,
            halt_probability=halt_probability,
            ci_low=ci_low,
            ci_high=ci_high,
            recommended_backend=backend,
            evidence_refs=evidence_refs,
            guard_decision=guard_decision,
        )


def _policy_decision(action: SteeringAction) -> str:
    if action == "proceed":
        return "allow"
    if action == "escalate":
        return "warn"
    return "halt"


def _explanation(action: SteeringAction, reason: str) -> str:
    if action == "proceed":
        return "Trajectory evidence indicates low pre-halt risk."
    if reason == "predictive_insufficient_simulations":
        return "Trajectory evidence is insufficient; use a stronger verifier."
    if reason == "predictive_uncertainty":
        return "Trajectory uncertainty crosses the calibrated threshold."
    if action == "halt":
        return "Trajectory evidence predicts a likely runtime halt."
    return "Trajectory preflight recommends verifier escalation."


def _failed_trajectory_refs(
    trajectories: Sequence[TrajectoryResult],
) -> tuple[str, ...]:
    return tuple(
        f"trajectory:{trajectory.trajectory_id}"
        for trajectory in trajectories
        if not trajectory.approved
    )


def _clamp_probability(value: float) -> float:
    _validate_unit_interval("probability", value)
    return max(0.0, min(1.0, float(value)))


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
