# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - physical budget tests

from __future__ import annotations

from collections.abc import Sequence

from director_ai.core.agent import CoherenceAgent
from director_ai.core.cyber_physical import (
    AABB,
    GroundingHook,
    JointChain,
    PhysicalAction,
    PhysicalBudgetLimits,
    SimpleKinematicModel,
    SpatialConstraint,
    Sphere,
    TenantPhysicalBudget,
    Vec3,
    VelocityConstraint,
)


class CountingModel(SimpleKinematicModel):
    def __init__(self) -> None:
        super().__init__(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0))
        )
        self.inverse_calls = 0
        self.collision_calls = 0

    def inverse(self, target: Vec3) -> tuple[float, ...] | None:
        self.inverse_calls += 1
        return super().inverse(target)

    def collides_with(
        self,
        point: Vec3,
        obstacles_aabb: Sequence[AABB] = (),
        obstacles_sphere: Sequence[Sphere] = (),
    ) -> bool:
        self.collision_calls += 1
        return super().collides_with(point, obstacles_aabb, obstacles_sphere)


def _action() -> PhysicalAction:
    return PhysicalAction(actuator_id="arm", target_position=Vec3(1.0, 0.0, 0.0))


def test_action_validation_budget_is_per_tenant() -> None:
    budget = TenantPhysicalBudget(
        PhysicalBudgetLimits(
            window_seconds=60.0,
            max_action_validations=1,
            max_inverse_kinematics=10,
            max_simulation_checks=10,
        )
    )
    hook = GroundingHook(
        model=CountingModel(),
        constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
        budget=budget,
    )

    assert hook.evaluate(_action(), tenant_id="tenant-a").allowed
    blocked = hook.evaluate(_action(), tenant_id="tenant-a")
    other_tenant = hook.evaluate(_action(), tenant_id="tenant-b")

    assert not blocked.allowed
    assert blocked.violations[0].constraint == "budget:action_validations"
    assert blocked.safety_event is not None
    assert blocked.safety_event.tenant_id == "tenant-a"
    assert other_tenant.allowed


def test_inverse_budget_blocks_before_solver_call() -> None:
    model = CountingModel()
    budget = TenantPhysicalBudget(
        PhysicalBudgetLimits(
            max_action_validations=10,
            max_inverse_kinematics=0,
            max_simulation_checks=10,
        )
    )
    hook = GroundingHook(
        model=model,
        constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
        budget=budget,
    )

    verdict = hook.evaluate(_action(), tenant_id="tenant-a")

    assert not verdict.allowed
    assert verdict.violations[0].constraint == "budget:inverse_kinematics"
    assert model.inverse_calls == 0
    assert verdict.safety_event is not None
    assert verdict.safety_event.halt_reason == "physical_budget_exceeded"


def test_simulation_budget_blocks_before_collision_call() -> None:
    model = CountingModel()
    budget = TenantPhysicalBudget(
        PhysicalBudgetLimits(
            max_action_validations=10,
            max_inverse_kinematics=10,
            max_simulation_checks=0,
        )
    )
    obstacle = AABB(
        min_corner=Vec3(0.5, -0.5, -0.5),
        max_corner=Vec3(1.5, 0.5, 0.5),
    )
    hook = GroundingHook(
        model=model,
        constraints=[SpatialConstraint(name="obstacle", obstacles_aabb=(obstacle,))],
        reject_on_unreachable=False,
        budget=budget,
    )

    verdict = hook.evaluate(_action(), tenant_id="tenant-a")

    assert not verdict.allowed
    assert verdict.violations[0].constraint == "budget:simulation_checks"
    assert model.collision_calls == 0


def test_budget_window_resets() -> None:
    now = 0.0

    def clock() -> float:
        return now

    budget = TenantPhysicalBudget(
        PhysicalBudgetLimits(
            window_seconds=5.0,
            max_action_validations=1,
            max_inverse_kinematics=10,
            max_simulation_checks=10,
        ),
        clock=clock,
    )
    hook = GroundingHook(
        model=CountingModel(),
        constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
        budget=budget,
    )

    assert hook.evaluate(_action(), tenant_id="tenant-a").allowed
    assert not hook.evaluate(_action(), tenant_id="tenant-a").allowed
    now = 6.0
    assert hook.evaluate(_action(), tenant_id="tenant-a").allowed


def test_agent_warn_mode_keeps_budget_exhaustion_blocking() -> None:
    budget = TenantPhysicalBudget(
        PhysicalBudgetLimits(
            max_action_validations=0,
            max_inverse_kinematics=10,
            max_simulation_checks=10,
        )
    )
    hook = GroundingHook(
        model=CountingModel(),
        constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
        budget=budget,
    )
    agent = CoherenceAgent(grounding_hook=hook)

    verdict = agent.verify_physical_action(_action(), tenant_id="tenant-a")

    assert not verdict.allowed
    assert verdict.safety_event is not None
    assert verdict.safety_event.policy_decision == "block"
