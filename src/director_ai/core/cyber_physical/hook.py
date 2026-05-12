# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — GroundingHook

"""Compose a :class:`KinematicModel` with a set of
:class:`PhysicalConstraint` instances into one allow / reject
decision per proposed action.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..safety_event import SafetyEvent
from .budget import PhysicalBudgetExceededError, TenantPhysicalBudget
from .constraints import PhysicalConstraint
from .kinematics import KinematicModel, PhysicalAction, UnsupportedKinematicsError


@dataclass(frozen=True)
class Violation:
    """One failed constraint."""

    constraint: str
    reason: str


@dataclass(frozen=True)
class GroundingVerdict:
    """Result of :meth:`GroundingHook.evaluate`."""

    action: PhysicalAction
    allowed: bool
    violations: tuple[Violation, ...] = field(default_factory=tuple)
    safety_event: SafetyEvent | None = None

    @property
    def any_violation(self) -> bool:
        return bool(self.violations)


class GroundingHook:
    """Bind a model + constraint set into an evaluator.

    Parameters
    ----------
    model :
        The :class:`KinematicModel` to query for reachability /
        collision state.
    constraints :
        Sequence of :class:`PhysicalConstraint`. Order defines
        evaluation order; the hook still evaluates every
        constraint so the returned verdict carries a complete
        violation list for audit.
    reject_on_unreachable :
        When ``True`` (default), the hook calls ``model.inverse``
        on the target and rejects actions whose IK returns
        ``None``. Skipped when the action already carries a
        populated ``joint_angles`` tuple.
    """

    def __init__(
        self,
        *,
        model: KinematicModel,
        constraints: Sequence[PhysicalConstraint],
        reject_on_unreachable: bool = True,
        budget: TenantPhysicalBudget | None = None,
    ) -> None:
        if not constraints:
            raise ValueError("constraints must be non-empty")
        names = [c.name for c in constraints]
        if len(set(names)) != len(names):
            raise ValueError("constraint names must be unique")
        self._model = model
        self._constraints = tuple(constraints)
        self._reject_on_unreachable = reject_on_unreachable
        self._budget = budget

    def evaluate(
        self,
        action: PhysicalAction,
        *,
        tenant_id: str = "",
    ) -> GroundingVerdict:
        violations: list[Violation] = []
        budget_verdict = self._consume_budget(
            action,
            tenant_id=tenant_id,
            counter="action_validations",
        )
        if budget_verdict is not None:
            return budget_verdict
        if self._reject_on_unreachable and not action.joint_angles:
            budget_verdict = self._consume_budget(
                action,
                tenant_id=tenant_id,
                counter="inverse_kinematics",
            )
            if budget_verdict is not None:
                return budget_verdict
            try:
                solution = self._model.inverse(action.target_position)
            except (NotImplementedError, UnsupportedKinematicsError):
                # Model cannot answer reachability — defer to the
                # constraint set. Not a violation by itself.
                solution = ()
            if solution is None:
                violations.append(
                    Violation(
                        constraint="reachability",
                        reason=(
                            f"target {action.target_position} is outside "
                            "the model's reachable workspace"
                        ),
                    )
                )
        for constraint in self._constraints:
            if _constraint_counter(constraint) == "simulation_checks":
                budget_verdict = self._consume_budget(
                    action,
                    tenant_id=tenant_id,
                    counter="simulation_checks",
                )
                if budget_verdict is not None:
                    return budget_verdict
            reason = constraint.evaluate(action, self._model)
            if reason is not None:
                violations.append(Violation(constraint=constraint.name, reason=reason))
        allowed = not violations
        return GroundingVerdict(
            action=action,
            allowed=allowed,
            violations=tuple(violations),
            safety_event=_event_for_verdict(
                allowed,
                tuple(violations),
                tenant_id=tenant_id,
            ),
        )

    @property
    def model(self) -> KinematicModel:
        return self._model

    @property
    def constraints(self) -> tuple[PhysicalConstraint, ...]:
        return self._constraints

    def _consume_budget(
        self,
        action: PhysicalAction,
        *,
        tenant_id: str,
        counter: str,
    ) -> GroundingVerdict | None:
        if self._budget is None:
            return None
        try:
            self._budget.consume(tenant_id, counter)
        except PhysicalBudgetExceededError as exc:
            return _event_for_budget_exhaustion(action, exc)
        return None


def _event_for_verdict(
    allowed: bool,
    violations: tuple[Violation, ...],
    *,
    tenant_id: str = "",
) -> SafetyEvent:
    return SafetyEvent.from_policy_decision(
        hook_id="cyber_physical.grounding",
        hook_scope="cyber_physical",
        policy_decision="allow" if allowed else "block",
        tenant_id=tenant_id,
        halt_reason=(
            "physical_action_allow" if allowed else "physical_constraint_violation"
        ),
        observed_score=1.0 if allowed else 0.0,
        tenant_safe_explanation=(
            "Physical action allowed."
            if allowed
            else f"Physical action blocked by {len(violations)} constraint(s)."
        ),
        evidence_refs=tuple(
            f"physical:{violation.constraint}" for violation in violations
        ),
        attributes={"violation_count": str(len(violations))},
    )


def _event_for_budget_exhaustion(
    action: PhysicalAction,
    exc: PhysicalBudgetExceededError,
) -> GroundingVerdict:
    violation = Violation(
        constraint=f"budget:{exc.counter}",
        reason=str(exc),
    )
    return GroundingVerdict(
        action=action,
        allowed=False,
        violations=(violation,),
        safety_event=SafetyEvent.from_policy_decision(
            hook_id="cyber_physical.grounding",
            hook_scope="cyber_physical",
            policy_decision="block",
            halt_reason="physical_budget_exceeded",
            tenant_id=exc.tenant_id,
            observed_score=0.0,
            tenant_safe_explanation=(
                "Physical action blocked because the tenant budget was exhausted."
            ),
            evidence_refs=(f"physical_budget:{exc.counter}",),
            attributes={
                "budget_counter": exc.counter,
                "budget_limit": str(exc.limit),
                "budget_window_seconds": f"{exc.window_seconds:.3f}",
            },
        ),
    )


def _constraint_counter(constraint: PhysicalConstraint) -> str:
    if constraint.__class__.__name__ == "SpatialConstraint":
        return "simulation_checks"
    return "action_validations"
