# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — closed-loop physical grounding evaluator

"""Closed-loop pre-action and post-action physical state checks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from director_ai.core.guard_control import (
    GuardDecision,
    NoGoPolicy,
    RiskEnvelope,
    VerifierSignal,
)

from .budget import PhysicalBudgetExceededError, TenantPhysicalBudget
from .geometry import Vec3
from .hook import GroundingHook, GroundingVerdict, Violation
from .kinematics import PhysicalAction

PhysicalGroundingStatus = Literal["ok", "mismatch", "unsupported", "budget"]


@dataclass(frozen=True)
class SensorStateSnapshot:
    """Tenant-safe reference to one sensed or simulated physical state."""

    snapshot_ref: str
    sensor_id: str
    adapter_id: str
    timestamp: float
    end_effector_position: Vec3
    confidence: float = 1.0
    supported: bool = True
    status_detail: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_ref.strip():
            raise ValueError("snapshot_ref must be non-empty")
        if not self.sensor_id.strip():
            raise ValueError("sensor_id must be non-empty")
        if not self.adapter_id.strip():
            raise ValueError("adapter_id must be non-empty")
        if self.timestamp < 0.0:
            raise ValueError("timestamp must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, str | float | bool]:
        """Return metadata without raw sensor payloads."""
        return {
            "snapshot_ref": self.snapshot_ref,
            "sensor_id": self.sensor_id,
            "adapter_id": self.adapter_id,
            "timestamp": self.timestamp,
            "position_x": self.end_effector_position.x,
            "position_y": self.end_effector_position.y,
            "position_z": self.end_effector_position.z,
            "confidence": self.confidence,
            "supported": self.supported,
            "status_detail": self.status_detail,
        }


@dataclass(frozen=True)
class PhysicalGroundingViolation:
    """One closed-loop physical grounding violation."""

    stage: str
    status: PhysicalGroundingStatus
    constraint: str
    reason: str
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must be non-empty")
        if not self.constraint.strip():
            raise ValueError("constraint must be non-empty")
        if not self.reason.strip():
            raise ValueError("reason must be non-empty")
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))


@dataclass(frozen=True)
class PhysicalGroundingEvaluation:
    """Closed-loop evaluation result."""

    action: PhysicalAction
    decision: GuardDecision
    reason: str
    pre_action: GroundingVerdict
    post_action_verified: bool
    requires_human_review: bool = False
    violations: tuple[PhysicalGroundingViolation, ...] = field(default_factory=tuple)


class PhysicalGroundingEvaluator:
    """Compare perception, simulation, action, and post-action state."""

    def __init__(
        self,
        *,
        grounding_hook: GroundingHook,
        high_risk_physical_deployment: bool = False,
        state_tolerance_m: float = 0.05,
        budget: TenantPhysicalBudget | None = None,
        no_go_policy: NoGoPolicy | None = None,
    ) -> None:
        if state_tolerance_m < 0.0:
            raise ValueError("state_tolerance_m must be non-negative")
        self._hook = grounding_hook
        self._high_risk = high_risk_physical_deployment
        self._tolerance = state_tolerance_m
        self._budget = budget
        self._no_go = no_go_policy or NoGoPolicy(
            irreversible_threshold=0.1,
            default_threshold=0.95,
            require_human_review_for_irreversible=True,
        )

    def evaluate(
        self,
        *,
        action: PhysicalAction,
        risk_envelope: RiskEnvelope,
        pre_perception: SensorStateSnapshot,
        pre_simulation: SensorStateSnapshot,
        post_perception: SensorStateSnapshot | None = None,
        post_simulation: SensorStateSnapshot | None = None,
        tenant_id: str = "",
    ) -> PhysicalGroundingEvaluation:
        """Run pre-action grounding, snapshot consistency, and post-action checks."""
        budget_violation = self._consume_sensor_budget(action, tenant_id)
        pre_action = self._hook.evaluate(action, tenant_id=tenant_id)
        if budget_violation is not None:
            return self._evaluation(
                action=action,
                reason="physical_budget_exceeded",
                risk_score=1.0,
                risk_envelope=risk_envelope,
                pre_action=pre_action,
                violations=(budget_violation,),
                post_action_verified=False,
                decision_override="block",
            )
        violations: list[PhysicalGroundingViolation] = []
        violations.extend(
            _snapshot_pair_violations(
                stage="pre_action",
                observed=pre_perception,
                simulated=pre_simulation,
                tolerance_m=self._tolerance,
            )
        )
        if not pre_action.allowed:
            violations.extend(_grounding_violations(pre_action.violations))
        post_action_verified = False
        if post_perception is not None or post_simulation is not None:
            if post_perception is None or post_simulation is None:
                violations.append(
                    PhysicalGroundingViolation(
                        stage="post_action",
                        status="unsupported",
                        constraint="post_action_snapshot_pair",
                        reason="post-action verification requires both perception and simulation snapshots",
                        evidence_refs=(),
                    )
                )
            else:
                post_action_verified = True
                violations.extend(
                    _snapshot_pair_violations(
                        stage="post_action",
                        observed=post_perception,
                        simulated=post_simulation,
                        tolerance_m=self._tolerance,
                    )
                )
                violations.extend(
                    _target_violations(
                        action=action,
                        observed=post_perception,
                        simulated=post_simulation,
                        tolerance_m=self._tolerance,
                    )
                )
        reason = _reason_for(tuple(violations))
        risk_score = 0.0 if not violations else 0.9
        decision_override = self._decision_for(
            reason=reason,
            risk_envelope=risk_envelope,
            violations=tuple(violations),
        )
        no_go_candidate_risk = risk_score
        no_go_candidate_decision = decision_override
        if risk_envelope.reversibility == "irreversible":
            no_go_candidate_risk = max(no_go_candidate_risk, 0.1)
            if no_go_candidate_decision == "allow":
                no_go_candidate_decision = "warn"
        evaluation = self._evaluation(
            action=action,
            reason=reason,
            risk_score=no_go_candidate_risk,
            risk_envelope=risk_envelope,
            pre_action=pre_action,
            violations=tuple(violations),
            post_action_verified=post_action_verified,
            decision_override=no_go_candidate_decision,
        )
        no_go = (
            self._no_go.evaluate(evaluation.decision)
            if risk_envelope.reversibility == "irreversible"
            else None
        )
        if no_go is not None and no_go.decision == "block":
            return self._evaluation(
                action=action,
                reason=no_go.reason,
                risk_score=max(risk_score, 0.1),
                risk_envelope=risk_envelope,
                pre_action=pre_action,
                violations=tuple(violations),
                post_action_verified=post_action_verified,
                decision_override="block",
                requires_human_review=no_go.requires_human_review,
            )
        return evaluation

    def _consume_sensor_budget(
        self,
        action: PhysicalAction,
        tenant_id: str,
    ) -> PhysicalGroundingViolation | None:
        _ = action
        if self._budget is None:
            return None
        try:
            self._budget.consume(tenant_id, "sensor_fusion")
        except PhysicalBudgetExceededError as exc:
            return PhysicalGroundingViolation(
                stage="pre_action",
                status="budget",
                constraint=f"budget:{exc.counter}",
                reason=str(exc),
                evidence_refs=(f"physical_budget:{exc.counter}",),
            )
        return None

    def _decision_for(
        self,
        *,
        reason: str,
        risk_envelope: RiskEnvelope,
        violations: tuple[PhysicalGroundingViolation, ...],
    ) -> str:
        if not violations:
            return "allow"
        if reason == "physical_budget_exceeded":
            return "block"
        if risk_envelope.reversibility == "irreversible":
            return "block"
        return "block" if self._high_risk else "warn"

    def _evaluation(
        self,
        *,
        action: PhysicalAction,
        reason: str,
        risk_score: float,
        risk_envelope: RiskEnvelope,
        pre_action: GroundingVerdict,
        violations: tuple[PhysicalGroundingViolation, ...],
        post_action_verified: bool,
        decision_override: str,
        requires_human_review: bool = False,
    ) -> PhysicalGroundingEvaluation:
        decision = _guard_decision(
            decision=decision_override,
            reason=reason,
            risk_score=risk_score,
            risk_envelope=risk_envelope,
            violations=violations,
        )
        return PhysicalGroundingEvaluation(
            action=action,
            decision=decision,
            reason=reason,
            pre_action=pre_action,
            post_action_verified=post_action_verified,
            requires_human_review=requires_human_review,
            violations=violations,
        )


def _snapshot_pair_violations(
    *,
    stage: str,
    observed: SensorStateSnapshot,
    simulated: SensorStateSnapshot,
    tolerance_m: float,
) -> tuple[PhysicalGroundingViolation, ...]:
    violations: list[PhysicalGroundingViolation] = []
    for snapshot in (observed, simulated):
        if not snapshot.supported:
            violations.append(
                PhysicalGroundingViolation(
                    stage=stage,
                    status="unsupported",
                    constraint="sensor_adapter_supported",
                    reason=f"adapter {snapshot.adapter_id!r} reported unsupported state",
                    evidence_refs=(snapshot.snapshot_ref,),
                )
            )
    if violations:
        return tuple(violations)
    distance = observed.end_effector_position.distance(simulated.end_effector_position)
    if distance > tolerance_m:
        violations.append(
            PhysicalGroundingViolation(
                stage=stage,
                status="mismatch",
                constraint="perception_simulation_consistency",
                reason=(
                    f"perception/simulation state distance {distance:.6f} m "
                    f"exceeds tolerance {tolerance_m:.6f} m"
                ),
                evidence_refs=(observed.snapshot_ref, simulated.snapshot_ref),
            )
        )
    return tuple(violations)


def _target_violations(
    *,
    action: PhysicalAction,
    observed: SensorStateSnapshot,
    simulated: SensorStateSnapshot,
    tolerance_m: float,
) -> tuple[PhysicalGroundingViolation, ...]:
    violations: list[PhysicalGroundingViolation] = []
    for label, snapshot in (("perception", observed), ("simulation", simulated)):
        distance = snapshot.end_effector_position.distance(action.target_position)
        if distance > tolerance_m:
            violations.append(
                PhysicalGroundingViolation(
                    stage="post_action",
                    status="mismatch",
                    constraint=f"{label}_target_consistency",
                    reason=(
                        f"{label} post-action distance {distance:.6f} m "
                        f"exceeds tolerance {tolerance_m:.6f} m"
                    ),
                    evidence_refs=(snapshot.snapshot_ref,),
                )
            )
    return tuple(violations)


def _grounding_violations(
    violations: Sequence[Violation],
) -> tuple[PhysicalGroundingViolation, ...]:
    return tuple(
        PhysicalGroundingViolation(
            stage="pre_action",
            status="mismatch",
            constraint=violation.constraint,
            reason=violation.reason,
            evidence_refs=(f"physical:{violation.constraint}",),
        )
        for violation in violations
    )


def _reason_for(violations: tuple[PhysicalGroundingViolation, ...]) -> str:
    if not violations:
        return "physical_grounding_consistent"
    if any(v.status == "budget" for v in violations):
        return "physical_budget_exceeded"
    if any(v.status == "unsupported" for v in violations):
        return "physical_sensor_unsupported"
    return "physical_state_mismatch"


def _guard_decision(
    *,
    decision: str,
    reason: str,
    risk_score: float,
    risk_envelope: RiskEnvelope,
    violations: tuple[PhysicalGroundingViolation, ...],
) -> GuardDecision:
    evidence_refs = tuple(ref for v in violations for ref in v.evidence_refs)
    signal = VerifierSignal(
        verifier="cyber_physical.closed_loop",
        modality="physical",
        score=risk_score,
        verdict=reason,
        confidence_low=0.8,
        confidence_high=1.0,
        evidence_refs=evidence_refs,
        failure_mode="" if decision == "allow" else reason,
    )
    return GuardDecision(
        decision=decision,
        risk_score=risk_score,
        confidence_low=0.8,
        confidence_high=1.0,
        policy_id="policy.cyber_physical.closed_loop",
        reason=reason,
        tenant_safe_explanation=_explanation(reason, len(violations)),
        evidence_refs=evidence_refs,
        verifier_signals=(signal,),
        risk_envelope=risk_envelope,
        attributes={
            "violation_count": str(len(violations)),
            "post_action_verified": str(
                any(v.stage == "post_action" for v in violations)
            ),
        },
    )


def _explanation(reason: str, count: int) -> str:
    if reason == "physical_grounding_consistent":
        return "Physical action is consistent across grounding checks."
    if reason == "physical_budget_exceeded":
        return (
            "Physical action blocked because a physical grounding budget was exhausted."
        )
    if reason == "physical_sensor_unsupported":
        return "Physical action could not be fully grounded because a sensor or adapter is unsupported."
    if reason.startswith("no_go_"):
        return "Physical action blocked by no-go policy."
    return f"Physical action has {count} closed-loop grounding violation(s)."
