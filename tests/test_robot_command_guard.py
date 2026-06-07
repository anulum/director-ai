# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Embodied-AI Robot Command Guard Tests
"""Multi-angle tests for pre-execution verification of LLM robot command plans."""

from __future__ import annotations

import pytest

from director_ai.core.cyber_physical import (
    AABB,
    PhysicalAction,
    PlanVerdict,
    RobotCommandGuard,
    StepViolation,
    Vec3,
    VelocityConstraint,
    WorkspaceConstraint,
)
from director_ai.core.cyber_physical.command_guard import (
    PATH_LENGTH,
    STEP_DISPLACEMENT,
)

_ENV = WorkspaceConstraint(
    name="env", envelope=AABB(min_corner=Vec3(0, 0, 0), max_corner=Vec3(1, 1, 1))
)
_VMAX = VelocityConstraint(name="vmax", max_velocity=1.0)


def _act(x: float, y: float, z: float, v: float = 0.0) -> PhysicalAction:
    return PhysicalAction(
        actuator_id="arm", target_position=Vec3(x, y, z), velocity_magnitude=v
    )


class TestValidation:
    def test_empty_plan_rejected(self):
        with pytest.raises(ValueError, match="at least one action"):
            RobotCommandGuard([_ENV]).verify_plan([])

    def test_negative_step_cap_rejected(self):
        with pytest.raises(ValueError, match="max_step_displacement"):
            RobotCommandGuard([], max_step_displacement=-1.0)

    def test_negative_path_cap_rejected(self):
        with pytest.raises(ValueError, match="max_path_length"):
            RobotCommandGuard([], max_path_length=-1.0)


class TestPerActionConstraints:
    def test_safe_plan_passes(self):
        guard = RobotCommandGuard([_ENV, _VMAX], high_risk_enabled=True)
        verdict = guard.verify_plan([_act(0.1, 0.1, 0.1), _act(0.2, 0.2, 0.1, 0.5)])
        assert verdict.safe is True
        assert verdict.blocked is False
        assert verdict.warn_only is False
        assert verdict.step_count == 2

    def test_constraint_violation_carries_step_index(self):
        guard = RobotCommandGuard([_ENV, _VMAX], high_risk_enabled=True)
        verdict = guard.verify_plan([_act(0.1, 0.1, 0.1), _act(5, 5, 5, 9.0)])
        offenders = {(v.step_index, v.constraint) for v in verdict.violations}
        assert (1, "env") in offenders
        assert (1, "vmax") in offenders

    def test_first_step_can_violate(self):
        guard = RobotCommandGuard([_ENV], high_risk_enabled=True)
        verdict = guard.verify_plan([_act(9, 9, 9)])
        assert verdict.violations[0].step_index == 0


class TestTemporalConstraints:
    def test_step_displacement_cap(self):
        guard = RobotCommandGuard([], high_risk_enabled=True, max_step_displacement=0.5)
        verdict = guard.verify_plan([_act(0, 0, 0), _act(10, 0, 0)])
        assert any(v.constraint == STEP_DISPLACEMENT for v in verdict.violations)
        assert verdict.violations[0].step_index == 1

    def test_path_length_cap_reported_once(self):
        guard = RobotCommandGuard([], high_risk_enabled=True, max_path_length=1.0)
        # three 1.0 steps → cumulative 3.0 > 1.0, reported a single time.
        verdict = guard.verify_plan(
            [_act(0, 0, 0), _act(1, 0, 0), _act(2, 0, 0), _act(3, 0, 0)]
        )
        path_hits = [v for v in verdict.violations if v.constraint == PATH_LENGTH]
        assert len(path_hits) == 1

    def test_within_temporal_caps_is_safe(self):
        guard = RobotCommandGuard(
            [], high_risk_enabled=True, max_step_displacement=0.5, max_path_length=2.0
        )
        verdict = guard.verify_plan([_act(0, 0, 0), _act(0.3, 0, 0), _act(0.6, 0, 0)])
        assert verdict.safe is True


class TestWarnOnlyPosture:
    def test_warn_only_by_default(self):
        guard = RobotCommandGuard([_ENV])  # high_risk_enabled defaults False
        verdict = guard.verify_plan([_act(0.1, 0.1, 0.1), _act(9, 9, 9)])
        assert verdict.warn_only is True
        assert verdict.blocked is False
        assert verdict.safe is False  # the violation is still surfaced

    def test_high_risk_blocks(self):
        guard = RobotCommandGuard([_ENV], high_risk_enabled=True)
        verdict = guard.verify_plan([_act(0.1, 0.1, 0.1), _act(9, 9, 9)])
        assert verdict.blocked is True
        assert verdict.warn_only is False


class TestSerialisation:
    def test_to_dict_tenant_safe(self):
        guard = RobotCommandGuard([_ENV], high_risk_enabled=True)
        d = guard.verify_plan([_act(9, 9, 9)]).to_dict()
        assert set(d) == {"blocked", "warn_only", "safe", "step_count", "violations"}
        assert set(d["violations"][0]) == {"step_index", "constraint", "reason"}

    def test_step_violation_to_dict(self):
        v = StepViolation(step_index=2, constraint="c", reason="r")
        assert v.to_dict() == {"step_index": 2, "constraint": "c", "reason": "r"}


class TestGuardWiring:
    def test_guard_robot_command_guard(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        rcg = guard.robot_command_guard([_ENV], high_risk_enabled=True)
        assert isinstance(rcg, RobotCommandGuard)
        verdict = rcg.verify_plan([_act(9, 9, 9)])
        assert isinstance(verdict, PlanVerdict)
        assert verdict.blocked is True
