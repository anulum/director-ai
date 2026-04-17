# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cyber-physical grounding tests

"""Multi-angle coverage: Vec3 / AABB / Sphere primitives,
SimpleKinematicModel FK + analytical two-link IK round-trip,
collision checks with margin, every constraint class,
GroundingHook allow / reject + reachability branch, lazy-adapter
import guards."""

from __future__ import annotations

import importlib.util
import math

import pytest

from director_ai.core.cyber_physical import (
    AABB,
    CarlaAdapter,
    GroundingHook,
    GroundingVerdict,
    JointChain,
    MuJoCoAdapter,
    PhysicalAction,
    Ros2Adapter,
    SimpleKinematicModel,
    SpatialConstraint,
    Sphere,
    TorqueConstraint,
    Vec3,
    VelocityConstraint,
    WorkspaceConstraint,
)

# --- Vec3 -----------------------------------------------------------


class TestVec3:
    def test_arithmetic(self):
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)
        assert a + b == Vec3(5.0, 7.0, 9.0)
        assert b - a == Vec3(3.0, 3.0, 3.0)
        assert a * 2.0 == Vec3(2.0, 4.0, 6.0)

    def test_dot_and_norm(self):
        v = Vec3(3.0, 4.0, 0.0)
        assert v.norm() == pytest.approx(5.0)
        assert v.dot(v) == pytest.approx(25.0)

    def test_distance(self):
        a = Vec3(0.0, 0.0, 0.0)
        b = Vec3(1.0, 2.0, 2.0)
        assert a.distance(b) == pytest.approx(3.0)

    def test_clamp(self):
        p = Vec3(5.0, -1.0, 3.0)
        low = Vec3(0.0, 0.0, 0.0)
        high = Vec3(2.0, 2.0, 2.0)
        assert p.clamp(low, high) == Vec3(2.0, 0.0, 2.0)

    def test_clamp_rejects_bad_bounds(self):
        with pytest.raises(ValueError, match="componentwise"):
            Vec3(0.0, 0.0, 0.0).clamp(Vec3(1.0, 0.0, 0.0), Vec3(0.0, 1.0, 1.0))

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            Vec3(math.nan, 0.0, 0.0)


# --- AABB -----------------------------------------------------------


class TestAABB:
    def test_contains_and_intersect(self):
        box = AABB(min_corner=Vec3(0.0, 0.0, 0.0), max_corner=Vec3(1.0, 1.0, 1.0))
        assert box.contains(Vec3(0.5, 0.5, 0.5))
        assert not box.contains(Vec3(2.0, 0.5, 0.5))

    def test_intersects(self):
        a = AABB(min_corner=Vec3(0.0, 0.0, 0.0), max_corner=Vec3(1.0, 1.0, 1.0))
        b = AABB(min_corner=Vec3(0.5, 0.5, 0.5), max_corner=Vec3(2.0, 2.0, 2.0))
        c = AABB(min_corner=Vec3(5.0, 5.0, 5.0), max_corner=Vec3(6.0, 6.0, 6.0))
        assert a.intersects(b)
        assert not a.intersects(c)

    def test_expand(self):
        box = AABB(min_corner=Vec3(1.0, 1.0, 1.0), max_corner=Vec3(2.0, 2.0, 2.0))
        expanded = box.expand(0.5)
        assert expanded.min_corner == Vec3(0.5, 0.5, 0.5)
        assert expanded.max_corner == Vec3(2.5, 2.5, 2.5)

    def test_centre(self):
        box = AABB(min_corner=Vec3(0.0, 0.0, 0.0), max_corner=Vec3(2.0, 4.0, 6.0))
        assert box.centre == Vec3(1.0, 2.0, 3.0)

    def test_invalid_corners(self):
        with pytest.raises(ValueError, match="min_corner"):
            AABB(min_corner=Vec3(1.0, 0.0, 0.0), max_corner=Vec3(0.0, 0.0, 0.0))


# --- Sphere ---------------------------------------------------------


class TestSphere:
    def test_contains(self):
        s = Sphere(centre=Vec3(0.0, 0.0, 0.0), radius=1.0)
        assert s.contains(Vec3(0.5, 0.0, 0.0))
        assert not s.contains(Vec3(2.0, 0.0, 0.0))

    def test_intersects(self):
        a = Sphere(centre=Vec3(0.0, 0.0, 0.0), radius=1.0)
        b = Sphere(centre=Vec3(1.5, 0.0, 0.0), radius=1.0)
        assert a.intersects(b)
        c = Sphere(centre=Vec3(5.0, 0.0, 0.0), radius=0.5)
        assert not a.intersects(c)

    def test_intersects_aabb(self):
        s = Sphere(centre=Vec3(0.0, 0.0, 0.0), radius=0.5)
        inside = AABB(
            min_corner=Vec3(-1.0, -1.0, -1.0), max_corner=Vec3(1.0, 1.0, 1.0)
        )
        away = AABB(min_corner=Vec3(5.0, 5.0, 5.0), max_corner=Vec3(6.0, 6.0, 6.0))
        assert s.intersects_aabb(inside)
        assert not s.intersects_aabb(away)

    def test_negative_radius(self):
        with pytest.raises(ValueError, match="radius"):
            Sphere(centre=Vec3(0.0, 0.0, 0.0), radius=-1.0)


# --- JointChain + SimpleKinematicModel ------------------------------


class TestSimpleKinematicModel:
    def _two_link(self) -> SimpleKinematicModel:
        return SimpleKinematicModel(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0)),
        )

    def test_forward_zero_angles(self):
        model = self._two_link()
        p = model.forward([0.0, 0.0])
        assert p == Vec3(2.0, 0.0, 0.0)

    def test_forward_right_angle(self):
        model = self._two_link()
        p = model.forward([math.pi / 2, 0.0])
        assert p.x == pytest.approx(0.0, abs=1e-9)
        assert p.y == pytest.approx(2.0, abs=1e-9)

    def test_forward_bad_length(self):
        model = self._two_link()
        with pytest.raises(ValueError, match="joint_angles"):
            model.forward([0.0])

    def test_inverse_roundtrip(self):
        model = self._two_link()
        target = Vec3(1.0, 1.0, 0.0)
        solution = model.inverse(target)
        assert solution is not None
        recovered = model.forward(solution)
        assert recovered.distance(target) < 1e-9

    def test_inverse_elbow_down_reaches_same_target(self):
        elbow_down = SimpleKinematicModel(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0)),
            branch="elbow_down",
        )
        target = Vec3(1.0, 1.0, 0.0)
        solution = elbow_down.inverse(target)
        assert solution is not None
        assert elbow_down.forward(solution).distance(target) < 1e-9

    def test_inverse_unreachable_returns_none(self):
        model = self._two_link()
        assert model.inverse(Vec3(10.0, 0.0, 0.0)) is None

    def test_inverse_long_chain_raises(self):
        model = SimpleKinematicModel(
            chain=JointChain(
                base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0, 1.0)
            ),
        )
        with pytest.raises(NotImplementedError):
            model.inverse(Vec3(1.0, 0.0, 0.0))

    def test_collides_with_aabb(self):
        model = self._two_link()
        obstacle = AABB(
            min_corner=Vec3(0.8, -0.2, -0.2), max_corner=Vec3(1.2, 0.2, 0.2)
        )
        assert model.collides_with(Vec3(1.0, 0.0, 0.0), obstacles_aabb=(obstacle,))
        assert not model.collides_with(
            Vec3(2.0, 2.0, 2.0), obstacles_aabb=(obstacle,)
        )

    def test_collision_margin_expands_obstacles(self):
        model = SimpleKinematicModel(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0)),
            collision_margin=0.5,
        )
        obstacle = AABB(
            min_corner=Vec3(1.5, -0.1, -0.1), max_corner=Vec3(1.6, 0.1, 0.1)
        )
        # Without margin the end-effector at (1.0, 0, 0) is >0.4 away;
        # with margin 0.5 the obstacle expands to [1.0, -0.6, -0.6]..
        # [2.1, 0.6, 0.6] and the point is inside.
        assert model.collides_with(
            Vec3(1.0, 0.0, 0.0), obstacles_aabb=(obstacle,)
        )

    def test_collides_with_sphere(self):
        model = self._two_link()
        sphere = Sphere(centre=Vec3(1.0, 0.0, 0.0), radius=0.1)
        assert model.collides_with(
            Vec3(1.05, 0.0, 0.0), obstacles_sphere=(sphere,)
        )

    def test_extra_obstacles_applied(self):
        sphere = Sphere(centre=Vec3(1.0, 0.0, 0.0), radius=0.1)
        model = SimpleKinematicModel(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0)),
            extra_obstacles_sphere=(sphere,),
        )
        assert model.collides_with(Vec3(1.0, 0.0, 0.0))

    def test_bad_branch(self):
        with pytest.raises(ValueError, match="branch"):
            SimpleKinematicModel(
                chain=JointChain(
                    base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0)
                ),
                branch="diagonal",
            )


# --- PhysicalAction -------------------------------------------------


class TestPhysicalAction:
    def test_valid(self):
        action = PhysicalAction(
            actuator_id="arm_left",
            target_position=Vec3(1.0, 0.0, 0.0),
            velocity_magnitude=0.5,
            torque_magnitude=2.0,
        )
        assert action.actuator_id == "arm_left"

    def test_empty_actuator(self):
        with pytest.raises(ValueError, match="actuator_id"):
            PhysicalAction(actuator_id="", target_position=Vec3(0.0, 0.0, 0.0))

    def test_negative_velocity(self):
        with pytest.raises(ValueError, match="velocity"):
            PhysicalAction(
                actuator_id="arm",
                target_position=Vec3(0.0, 0.0, 0.0),
                velocity_magnitude=-1.0,
            )

    def test_negative_torque(self):
        with pytest.raises(ValueError, match="torque"):
            PhysicalAction(
                actuator_id="arm",
                target_position=Vec3(0.0, 0.0, 0.0),
                torque_magnitude=-1.0,
            )


# --- Constraints ----------------------------------------------------


class TestConstraints:
    def _model(self) -> SimpleKinematicModel:
        return SimpleKinematicModel(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0)),
        )

    def test_workspace_allows_inside(self):
        c = WorkspaceConstraint(
            name="cage",
            envelope=AABB(
                min_corner=Vec3(-2.0, -2.0, -2.0), max_corner=Vec3(2.0, 2.0, 2.0)
            ),
        )
        action = PhysicalAction(
            actuator_id="arm", target_position=Vec3(1.0, 0.0, 0.0)
        )
        assert c.evaluate(action, self._model()) is None

    def test_workspace_rejects_outside(self):
        c = WorkspaceConstraint(
            name="cage",
            envelope=AABB(
                min_corner=Vec3(-1.0, -1.0, -1.0), max_corner=Vec3(1.0, 1.0, 1.0)
            ),
        )
        action = PhysicalAction(
            actuator_id="arm", target_position=Vec3(5.0, 0.0, 0.0)
        )
        reason = c.evaluate(action, self._model())
        assert reason is not None
        assert "outside" in reason

    def test_spatial_constraint(self):
        box = AABB(
            min_corner=Vec3(0.9, -0.1, -0.1), max_corner=Vec3(1.1, 0.1, 0.1)
        )
        c = SpatialConstraint(name="table_leg", obstacles_aabb=(box,))
        action = PhysicalAction(
            actuator_id="arm", target_position=Vec3(1.0, 0.0, 0.0)
        )
        assert c.evaluate(action, self._model()) is not None

    def test_spatial_no_obstacle(self):
        with pytest.raises(ValueError, match="obstacle"):
            SpatialConstraint(name="empty")

    def test_velocity_allow(self):
        c = VelocityConstraint(name="cap", max_velocity=1.0)
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1.0, 0.0, 0.0),
            velocity_magnitude=0.5,
        )
        assert c.evaluate(action, self._model()) is None

    def test_velocity_reject(self):
        c = VelocityConstraint(name="cap", max_velocity=1.0)
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1.0, 0.0, 0.0),
            velocity_magnitude=1.5,
        )
        reason = c.evaluate(action, self._model())
        assert reason is not None
        assert "velocity" in reason

    def test_torque_reject(self):
        c = TorqueConstraint(name="motor", max_torque=5.0)
        action = PhysicalAction(
            actuator_id="arm",
            target_position=Vec3(1.0, 0.0, 0.0),
            torque_magnitude=7.0,
        )
        reason = c.evaluate(action, self._model())
        assert reason is not None
        assert "torque" in reason

    def test_negative_velocity_constraint_rejected(self):
        with pytest.raises(ValueError, match="max_velocity"):
            VelocityConstraint(name="c", max_velocity=-1.0)

    def test_negative_torque_constraint_rejected(self):
        with pytest.raises(ValueError, match="max_torque"):
            TorqueConstraint(name="c", max_torque=-1.0)


# --- GroundingHook --------------------------------------------------


class TestGroundingHook:
    def _model(self) -> SimpleKinematicModel:
        return SimpleKinematicModel(
            chain=JointChain(base=Vec3(0.0, 0.0, 0.0), link_lengths=(1.0, 1.0)),
        )

    def test_allows_clean_action(self):
        hook = GroundingHook(
            model=self._model(),
            constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
        )
        verdict = hook.evaluate(
            PhysicalAction(
                actuator_id="arm",
                target_position=Vec3(1.0, 1.0, 0.0),
                velocity_magnitude=0.2,
            )
        )
        assert isinstance(verdict, GroundingVerdict)
        assert verdict.allowed
        assert verdict.violations == ()

    def test_reports_every_violation(self):
        hook = GroundingHook(
            model=self._model(),
            constraints=[
                VelocityConstraint(name="v", max_velocity=0.1),
                TorqueConstraint(name="t", max_torque=0.1),
            ],
        )
        verdict = hook.evaluate(
            PhysicalAction(
                actuator_id="arm",
                target_position=Vec3(1.0, 1.0, 0.0),
                velocity_magnitude=2.0,
                torque_magnitude=5.0,
            )
        )
        assert not verdict.allowed
        constraint_names = {v.constraint for v in verdict.violations}
        assert "v" in constraint_names
        assert "t" in constraint_names

    def test_rejects_unreachable_target(self):
        hook = GroundingHook(
            model=self._model(),
            constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
        )
        verdict = hook.evaluate(
            PhysicalAction(
                actuator_id="arm",
                target_position=Vec3(10.0, 10.0, 0.0),
            )
        )
        assert not verdict.allowed
        assert any(v.constraint == "reachability" for v in verdict.violations)

    def test_skips_reachability_when_flag_off(self):
        hook = GroundingHook(
            model=self._model(),
            constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
            reject_on_unreachable=False,
        )
        verdict = hook.evaluate(
            PhysicalAction(
                actuator_id="arm",
                target_position=Vec3(10.0, 10.0, 0.0),
            )
        )
        assert verdict.allowed

    def test_skips_reachability_when_joint_angles_provided(self):
        hook = GroundingHook(
            model=self._model(),
            constraints=[VelocityConstraint(name="v", max_velocity=1.0)],
        )
        verdict = hook.evaluate(
            PhysicalAction(
                actuator_id="arm",
                target_position=Vec3(1.0, 1.0, 0.0),
                joint_angles=(0.0, 0.0),
            )
        )
        assert verdict.allowed

    def test_empty_constraints_rejected(self):
        with pytest.raises(ValueError, match="constraints"):
            GroundingHook(model=self._model(), constraints=[])

    def test_duplicate_constraint_names_rejected(self):
        with pytest.raises(ValueError, match="unique"):
            GroundingHook(
                model=self._model(),
                constraints=[
                    VelocityConstraint(name="v", max_velocity=1.0),
                    VelocityConstraint(name="v", max_velocity=2.0),
                ],
            )

# --- Adapters (lazy-import guards) ---------------------------------


class TestRos2Adapter:
    def test_direct_ctor_validates(self):
        with pytest.raises(ValueError, match="node"):
            Ros2Adapter(node=None)

    def test_from_ros2_without_package(self):
        if importlib.util.find_spec("rclpy") is not None:
            pytest.skip("rclpy installed; skipping import-error branch")
        with pytest.raises(ImportError, match="ros2"):
            Ros2Adapter.from_ros2()

    def test_collides_with(self):
        # Sentinel node — Ros2Adapter's collides_with only uses
        # caller-supplied obstacles, not the node.
        node = object()
        adapter = Ros2Adapter(node=node)
        box = AABB(
            min_corner=Vec3(0.0, 0.0, 0.0), max_corner=Vec3(1.0, 1.0, 1.0)
        )
        assert adapter.collides_with(
            Vec3(0.5, 0.5, 0.5), obstacles_aabb=(box,)
        )

    def test_forward_raises(self):
        adapter = Ros2Adapter(node=object())
        with pytest.raises(NotImplementedError):
            adapter.forward([0.0])

    def test_inverse_raises(self):
        adapter = Ros2Adapter(node=object())
        with pytest.raises(NotImplementedError):
            adapter.inverse(Vec3(0.0, 0.0, 0.0))


class TestMuJoCoAdapter:
    def test_direct_ctor_validates(self):
        with pytest.raises(ValueError, match="model"):
            MuJoCoAdapter(model=None, data=object())

    def test_from_mjcf_without_package(self):
        if importlib.util.find_spec("mujoco") is not None:
            pytest.skip("mujoco installed")
        with pytest.raises(ImportError, match="mujoco"):
            MuJoCoAdapter.from_mjcf("/tmp/ghost.xml")

    def test_collides_with_user_obstacles(self):
        class _Data:
            ncon = 0

        adapter = MuJoCoAdapter(model=object(), data=_Data())
        sphere = Sphere(centre=Vec3(0.0, 0.0, 0.0), radius=1.0)
        assert adapter.collides_with(
            Vec3(0.5, 0.0, 0.0), obstacles_sphere=(sphere,)
        )


class TestCarlaAdapter:
    def test_direct_ctor_validates(self):
        with pytest.raises(ValueError, match="client"):
            CarlaAdapter(client=None, world=object())

    def test_from_carla_without_package(self):
        if importlib.util.find_spec("carla") is not None:
            pytest.skip("carla installed")
        with pytest.raises(ImportError, match="carla"):
            CarlaAdapter.from_carla()

    def test_from_carla_bad_port(self):
        if importlib.util.find_spec("carla") is None:
            pytest.skip("carla not installed — the port guard fires after the lazy import")
        with pytest.raises(ValueError, match="port"):
            CarlaAdapter.from_carla(port=0)

    def test_collides_with_user_obstacles(self):
        adapter = CarlaAdapter(client=object(), world=object())
        box = AABB(
            min_corner=Vec3(0.0, 0.0, 0.0), max_corner=Vec3(1.0, 1.0, 1.0)
        )
        assert adapter.collides_with(
            Vec3(0.5, 0.5, 0.5), obstacles_aabb=(box,)
        )
