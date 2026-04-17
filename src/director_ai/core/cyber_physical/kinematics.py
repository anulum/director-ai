# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — KinematicModel + SimpleKinematicModel

"""Kinematic modelling Protocol + a dependency-free default.

:class:`SimpleKinematicModel` supports:

* AABB / sphere / AABB-vs-sphere collision probes (from
  :mod:`.geometry`).
* Analytical two-link planar inverse kinematics — exact
  closed-form solution from Spong's *Robot Modeling and
  Control* (Ch. 5), both elbow-up and elbow-down branches.
* Forward kinematics on an arbitrary chain of revolute joints
  in a plane — the 3-D general case is a drop-in extension via
  the same :class:`JointChain` data model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .geometry import AABB, Sphere, Vec3

try:
    from backfire_kernel import rust_two_link_ik as _rust_two_link_ik

    _RUST_IK_AVAILABLE = True
except ImportError:  # pragma: no cover — optional accelerator
    _RUST_IK_AVAILABLE = False


@dataclass(frozen=True)
class PhysicalAction:
    """One actuation the agent wants to execute.

    Coordinates are metres, velocities m/s, torques N·m. The
    action carries both end-effector target (``target_position``)
    and optionally the full joint vector (``joint_angles``); the
    hook will reject actions whose joint vector would place the
    end-effector outside the reachable workspace.
    """

    actuator_id: str
    target_position: Vec3
    velocity_magnitude: float = 0.0
    torque_magnitude: float = 0.0
    joint_angles: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.actuator_id:
            raise ValueError("actuator_id must be non-empty")
        if self.velocity_magnitude < 0:
            raise ValueError("velocity_magnitude must be non-negative")
        if self.torque_magnitude < 0:
            raise ValueError("torque_magnitude must be non-negative")


@dataclass(frozen=True)
class JointChain:
    """Series of co-planar revolute joints.

    ``base`` is the fixed-link anchor. ``link_lengths`` are the
    lengths of each link, in order. The chain operates in the
    x-y plane — the z component of all end-effector positions is
    the base's z.
    """

    base: Vec3
    link_lengths: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.link_lengths:
            raise ValueError("link_lengths must be non-empty")
        for length in self.link_lengths:
            if length <= 0:
                raise ValueError(
                    f"link_lengths must all be positive; got {length!r}"
                )

    @property
    def reach(self) -> float:
        return sum(self.link_lengths)


@runtime_checkable
class KinematicModel(Protocol):
    """Any model that can score a :class:`PhysicalAction` against
    the current world state."""

    def forward(self, joint_angles: Sequence[float]) -> Vec3: ...

    def inverse(self, target: Vec3) -> tuple[float, ...] | None: ...

    def collides_with(
        self,
        point: Vec3,
        obstacles_aabb: Sequence[AABB] = (),
        obstacles_sphere: Sequence[Sphere] = (),
    ) -> bool: ...


@dataclass
class SimpleKinematicModel:
    """Closed-form 2-link / N-link planar chain + AABB/sphere
    collision.

    Two-link analytical IK uses the law of cosines; N-link FK
    walks the chain additively. Collision checks delegate to the
    :mod:`.geometry` primitives.
    """

    chain: JointChain
    branch: str = "elbow_up"
    collision_margin: float = 0.0
    # Optional end-effector envelope used during collision checks.
    end_effector_radius: float = 0.0
    extra_obstacles_aabb: tuple[AABB, ...] = field(default_factory=tuple)
    extra_obstacles_sphere: tuple[Sphere, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.branch not in {"elbow_up", "elbow_down"}:
            raise ValueError(
                "branch must be 'elbow_up' or 'elbow_down'; "
                f"got {self.branch!r}"
            )
        if self.collision_margin < 0:
            raise ValueError("collision_margin must be non-negative")
        if self.end_effector_radius < 0:
            raise ValueError("end_effector_radius must be non-negative")

    def forward(self, joint_angles: Sequence[float]) -> Vec3:
        """Forward kinematics for an arbitrary planar chain.

        Starts at the base and adds each link in the direction
        of the cumulative joint angle. Returns the end-effector
        position.
        """
        if len(joint_angles) != len(self.chain.link_lengths):
            raise ValueError(
                f"joint_angles length {len(joint_angles)} does not match "
                f"chain link count {len(self.chain.link_lengths)}"
            )
        x = self.chain.base.x
        y = self.chain.base.y
        cumulative = 0.0
        for angle, length in zip(
            joint_angles, self.chain.link_lengths, strict=True
        ):
            cumulative += angle
            x += length * math.cos(cumulative)
            y += length * math.sin(cumulative)
        return Vec3(x, y, self.chain.base.z)

    def inverse(self, target: Vec3) -> tuple[float, ...] | None:
        """Analytical IK.

        Supports the exact closed-form solution for two-link
        chains. Longer chains raise :class:`NotImplementedError`
        — the caller should plug in a numerical solver or a
        dedicated :class:`KinematicModel` subclass for those.

        Returns ``None`` when the target is unreachable.
        """
        links = self.chain.link_lengths
        if len(links) != 2:
            raise NotImplementedError(
                "SimpleKinematicModel.inverse supports two-link chains; "
                "extend with a numerical solver for longer chains"
            )
        l1, l2 = links
        elbow_up = self.branch == "elbow_up"
        if _RUST_IK_AVAILABLE:
            result = _rust_two_link_ik(
                l1,
                l2,
                (self.chain.base.x, self.chain.base.y),
                (target.x, target.y),
                elbow_up,
            )
            return None if result is None else result
        dx = target.x - self.chain.base.x
        dy = target.y - self.chain.base.y
        distance = math.sqrt(dx * dx + dy * dy)
        if distance > l1 + l2 or distance < abs(l1 - l2):
            return None
        cos_theta2 = (distance * distance - l1 * l1 - l2 * l2) / (
            2.0 * l1 * l2
        )
        cos_theta2 = max(-1.0, min(1.0, cos_theta2))
        theta2 = math.acos(cos_theta2)
        if not elbow_up:
            theta2 = -theta2
        k1 = l1 + l2 * math.cos(theta2)
        k2 = l2 * math.sin(theta2)
        theta1 = math.atan2(dy, dx) - math.atan2(k2, k1)
        return (theta1, theta2)

    def collides_with(
        self,
        point: Vec3,
        obstacles_aabb: Sequence[AABB] = (),
        obstacles_sphere: Sequence[Sphere] = (),
    ) -> bool:
        """Return ``True`` when the end-effector at ``point``
        intersects any obstacle, honouring the
        :attr:`collision_margin` and
        :attr:`end_effector_radius`.

        Extra obstacles declared on the model (from
        :attr:`extra_obstacles_aabb` / :attr:`extra_obstacles_sphere`)
        are checked in addition to the caller's list.
        """
        radius = self.end_effector_radius + self.collision_margin
        envelope = Sphere(centre=point, radius=max(0.0, radius))
        all_aabb = tuple(obstacles_aabb) + self.extra_obstacles_aabb
        if any(
            envelope.intersects_aabb(box.expand(self.collision_margin))
            for box in all_aabb
        ):
            return True
        all_sphere = tuple(obstacles_sphere) + self.extra_obstacles_sphere
        return any(envelope.intersects(sphere) for sphere in all_sphere)
