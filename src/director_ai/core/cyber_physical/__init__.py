# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cyber-physical grounding

"""Physical-world consistency checks for agent actions.

Every actuation an agent wants to perform is first grounded
against a kinematic model: collision bounds, velocity limits,
torque limits, workspace boundaries. The package is laid out
so the same :class:`GroundingHook` orchestrator works with:

* :class:`SimpleKinematicModel` — pure-Python AABB / sphere
  collision checker + analytical two-link inverse kinematics +
  forward kinematics for 2-D / 3-D chains. Zero external deps;
  every computation is a real closed-form geometry primitive.
* :class:`Ros2Adapter` — lazy-loaded ``rclpy`` adapter that
  queries a running ROS 2 node for collision state. Raises
  :class:`ImportError` with install instructions when ``rclpy``
  is missing.
* :class:`MuJoCoAdapter` — lazy-loaded ``mujoco`` adapter that
  evaluates forward dynamics against an MJCF model.
* :class:`CarlaAdapter` — lazy-loaded ``carla`` adapter that
  queries the driving simulator's world snapshot.

:class:`GroundingHook` composes a :class:`KinematicModel` with
a set of :class:`PhysicalConstraint` instances (spatial,
velocity, torque, workspace) and returns a
:class:`GroundingVerdict` with an ``allowed`` flag and the full
list of violations.
"""

from .adapters import CarlaAdapter, MuJoCoAdapter, Ros2Adapter
from .constraints import (
    PhysicalConstraint,
    SpatialConstraint,
    TorqueConstraint,
    VelocityConstraint,
    WorkspaceConstraint,
)
from .geometry import AABB, Sphere, Vec3
from .hook import GroundingHook, GroundingVerdict, Violation
from .kinematics import (
    JointChain,
    KinematicModel,
    PhysicalAction,
    SimpleKinematicModel,
)

__all__ = [
    "AABB",
    "CarlaAdapter",
    "GroundingHook",
    "GroundingVerdict",
    "JointChain",
    "KinematicModel",
    "MuJoCoAdapter",
    "PhysicalAction",
    "PhysicalConstraint",
    "Ros2Adapter",
    "SimpleKinematicModel",
    "SpatialConstraint",
    "Sphere",
    "TorqueConstraint",
    "Vec3",
    "VelocityConstraint",
    "Violation",
    "WorkspaceConstraint",
]
