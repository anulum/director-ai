# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Cyber-Physical Real-Surface Tests
"""Real production-surface tests for cyber-physical command guarding."""

from __future__ import annotations

from director_ai.core.config import DirectorConfig
from director_ai.core.cyber_physical import (
    AABB,
    JointChain,
    PhysicalAction,
    RobotCommandGuard,
    SimpleKinematicModel,
    SpatialConstraint,
    Vec3,
    VelocityConstraint,
    WorkspaceConstraint,
)
from director_ai.core.cyber_physical.command_guard import (
    PATH_LENGTH,
    STEP_DISPLACEMENT,
)
from director_ai.guard import ProductionGuard


def _production_guard() -> ProductionGuard:
    """Build the public production facade without loading an external NLI model."""
    return ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))


def _action(
    x: float,
    y: float = 0.0,
    z: float = 0.0,
    *,
    velocity: float = 0.0,
) -> PhysicalAction:
    """Create a public physical action for a deterministic arm target."""
    return PhysicalAction(
        actuator_id="arm",
        target_position=Vec3(x, y, z),
        velocity_magnitude=velocity,
    )


def _workspace() -> WorkspaceConstraint:
    """Return a bounded workcell envelope used by the real-surface workflow."""
    return WorkspaceConstraint(
        name="workcell",
        envelope=AABB(
            min_corner=Vec3(-0.1, -0.2, -0.2),
            max_corner=Vec3(2.0, 0.2, 0.2),
        ),
    )


def _fixture_collision() -> SpatialConstraint:
    """Return a spatial obstacle that intersects the middle target."""
    return SpatialConstraint(
        name="fixture_collision",
        obstacles_aabb=(
            AABB(
                min_corner=Vec3(0.95, -0.05, -0.05),
                max_corner=Vec3(1.05, 0.05, 0.05),
            ),
        ),
    )


def _kinematic_model() -> SimpleKinematicModel:
    """Return the dependency-free kinematic model used by public docs."""
    return SimpleKinematicModel(
        chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0))
    )


def test_robot_command_guard_blocks_missing_spatial_model_via_public_facade() -> None:
    """Spatial constraints without a model fail closed instead of crashing."""
    guard = _production_guard().robot_command_guard(
        [_fixture_collision()],
        high_risk_enabled=True,
    )

    verdict = guard.verify_plan([_action(1.0)])

    assert verdict.blocked is True
    assert verdict.warn_only is False
    assert verdict.safe is False
    assert verdict.step_count == 1
    assert len(verdict.violations) == 1
    violation = verdict.violations[0]
    assert violation.step_index == 0
    assert violation.constraint == "fixture_collision"
    assert "kinematic model" in violation.reason
    assert verdict.to_dict()["violations"] == [violation.to_dict()]


def test_robot_command_guard_blocks_real_spatial_and_temporal_hazards() -> None:
    """The public facade wires model, spatial, velocity, and temporal checks."""
    guard = _production_guard().robot_command_guard(
        [_workspace(), _fixture_collision(), VelocityConstraint("speed_cap", 1.0)],
        model=_kinematic_model(),
        high_risk_enabled=True,
        max_step_displacement=0.5,
        max_path_length=0.75,
    )

    verdict = guard.verify_plan(
        [
            _action(0.2, velocity=0.1),
            _action(1.0, velocity=0.1),
            _action(1.8, velocity=2.0),
        ]
    )

    assert isinstance(guard, RobotCommandGuard)
    assert verdict.blocked is True
    assert verdict.warn_only is False
    assert verdict.safe is False
    assert verdict.step_count == 3
    offenders = {(v.step_index, v.constraint) for v in verdict.violations}
    assert (1, "fixture_collision") in offenders
    assert (1, STEP_DISPLACEMENT) in offenders
    assert (1, PATH_LENGTH) in offenders
    assert (2, "speed_cap") in offenders

    payload = verdict.to_dict()
    assert payload["blocked"] is True
    assert payload["warn_only"] is False
    assert payload["safe"] is False
    assert payload["step_count"] == 3
    violations_payload = payload["violations"]
    assert isinstance(violations_payload, list)
    for violation in violations_payload:
        assert isinstance(violation, dict)
        assert set(violation) == {"step_index", "constraint", "reason"}
