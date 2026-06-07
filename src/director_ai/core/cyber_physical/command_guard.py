# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Embodied-AI Robot Command Guard
"""Verify an LLM-planned robot command sequence before it executes.

LLM planners increasingly emit sequences of robot actions. The per-action
:class:`~director_ai.core.cyber_physical.hook.GroundingHook` checks a single
action; this guard checks a whole *plan* before execution and adds **temporal**
safety properties that single-action checks cannot express — bounded per-step
displacement (no teleport jumps) and a bounded total path length.

It stays **warn-only by default**: violations are reported but the plan is not
blocked unless ``high_risk_enabled`` is set, matching the project's "physical
hooks warn-only until an explicit high-risk flag" posture. When enabled, an
unsafe plan is blocked *before* any action runs, with the violated constraint and
the offending step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .constraints import PhysicalConstraint
from .kinematics import KinematicModel, PhysicalAction

STEP_DISPLACEMENT = "max_step_displacement"
PATH_LENGTH = "max_path_length"


@dataclass(frozen=True)
class StepViolation:
    """A safety violation at one step of a planned command sequence."""

    step_index: int
    constraint: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe violation record."""
        return {
            "step_index": self.step_index,
            "constraint": self.constraint,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlanVerdict:
    """The outcome of verifying a planned command sequence."""

    blocked: bool
    warn_only: bool
    violations: tuple[StepViolation, ...]
    step_count: int

    @property
    def safe(self) -> bool:
        """True when the plan raised no safety violation."""
        return not self.violations

    def to_dict(self) -> dict[str, object]:
        """Tenant-safe verdict (constraint names + step indices only)."""
        return {
            "blocked": self.blocked,
            "warn_only": self.warn_only,
            "safe": self.safe,
            "step_count": self.step_count,
            "violations": [v.to_dict() for v in self.violations],
        }


class RobotCommandGuard:
    """Pre-execution guard for an LLM-planned robot action sequence.

    Parameters
    ----------
    constraints:
        Per-action physical constraints (workspace, spatial, velocity, torque, …)
        evaluated against every action in the plan.
    model:
        Kinematic model passed to the constraints (required only by constraints
        that use it, e.g. spatial collision).
    high_risk_enabled:
        When ``False`` (default) the guard is warn-only — violations are reported
        but the plan is not blocked. Set ``True`` for a real high-risk deployment
        so an unsafe plan is blocked before execution.
    max_step_displacement:
        Optional cap on the distance between consecutive action targets (rejects
        teleport-like jumps).
    max_path_length:
        Optional cap on the cumulative path length across the plan.
    """

    def __init__(
        self,
        constraints: Sequence[PhysicalConstraint] = (),
        *,
        model: KinematicModel | None = None,
        high_risk_enabled: bool = False,
        max_step_displacement: float | None = None,
        max_path_length: float | None = None,
    ) -> None:
        if max_step_displacement is not None and max_step_displacement < 0:
            raise ValueError("max_step_displacement must be non-negative")
        if max_path_length is not None and max_path_length < 0:
            raise ValueError("max_path_length must be non-negative")
        self._constraints = tuple(constraints)
        self._model = model
        self._high_risk = high_risk_enabled
        self._max_step = max_step_displacement
        self._max_path = max_path_length

    def verify_plan(self, actions: Sequence[PhysicalAction]) -> PlanVerdict:
        """Verify a planned action sequence before execution.

        Every action is checked against the per-action constraints and the plan
        against the temporal caps. The verdict blocks only when
        ``high_risk_enabled`` is set; otherwise it is warn-only.
        """
        if not actions:
            raise ValueError("plan must contain at least one action")
        violations: list[StepViolation] = []
        for index, action in enumerate(actions):
            for constraint in self._constraints:
                reason = constraint.evaluate(action, self._model)  # type: ignore[arg-type]
                if reason is not None:
                    violations.append(
                        StepViolation(
                            step_index=index,
                            constraint=constraint.name,
                            reason=reason,
                        )
                    )
        violations.extend(self._temporal_violations(actions))
        has_violation = bool(violations)
        return PlanVerdict(
            blocked=has_violation and self._high_risk,
            warn_only=has_violation and not self._high_risk,
            violations=tuple(violations),
            step_count=len(actions),
        )

    def _temporal_violations(
        self, actions: Sequence[PhysicalAction]
    ) -> list[StepViolation]:
        violations: list[StepViolation] = []
        total_path = 0.0
        path_reported = False
        for index in range(1, len(actions)):
            step = actions[index].target_position.distance(
                actions[index - 1].target_position
            )
            if self._max_step is not None and step > self._max_step:
                violations.append(
                    StepViolation(
                        step_index=index,
                        constraint=STEP_DISPLACEMENT,
                        reason=(
                            f"step displacement {step:.3f} exceeds limit "
                            f"{self._max_step:.3f}"
                        ),
                    )
                )
            total_path += step
            if (
                self._max_path is not None
                and total_path > self._max_path
                and not path_reported
            ):
                path_reported = True
                violations.append(
                    StepViolation(
                        step_index=index,
                        constraint=PATH_LENGTH,
                        reason=(
                            f"cumulative path {total_path:.3f} exceeds limit "
                            f"{self._max_path:.3f}"
                        ),
                    )
                )
        return violations
