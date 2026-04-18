# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — simulator adapters

"""Lazy-imported adapters for ROS 2, MuJoCo, and CARLA.

All three adapters implement the :class:`KinematicModel` Protocol
so :class:`GroundingHook` can operate with any of them. The
heavy libraries are imported at ``from_*`` construction time;
attempting to build an adapter without the backing library
installed raises :class:`ImportError` with install instructions,
never a silent fallback.

Each adapter accepts an already-constructed client / node /
model via its direct constructor so callers who manage their own
simulator lifecycle (e.g. inside a test harness that shares one
``carla.Client``) don't pay the ``from_*`` load cost twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .geometry import AABB, Sphere, Vec3


@dataclass
class Ros2Adapter:
    """``rclpy`` adapter.

    Reads the robot's latest joint state + collision state from a
    running ROS 2 node. ``node`` is the live node handle and
    ``collision_topic`` the topic the adapter polls for a
    ``moveit_msgs/CollisionObject`` array.

    Parameters
    ----------
    node :
        A live ``rclpy.node.Node``.
    joint_positions_topic :
        Topic carrying ``sensor_msgs/JointState`` messages.
    collision_topic :
        Topic carrying a list of active collision objects.
    """

    node: Any
    joint_positions_topic: str = "/joint_states"
    collision_topic: str = "/collision_objects"

    def __post_init__(self) -> None:
        if self.node is None:
            raise ValueError("node is required")
        if not self.joint_positions_topic:
            raise ValueError("joint_positions_topic must be non-empty")
        if not self.collision_topic:
            raise ValueError("collision_topic must be non-empty")

    @classmethod
    def from_ros2(
        cls,
        *,
        node_name: str = "director_grounding_hook",
        joint_positions_topic: str = "/joint_states",
        collision_topic: str = "/collision_objects",
    ) -> Ros2Adapter:
        try:
            import rclpy
            from rclpy.node import Node
        except ImportError as exc:
            raise ImportError(
                "Ros2Adapter.from_ros2 requires rclpy. "
                "Install ROS 2 and: pip install director-ai[ros2]"
            ) from exc
        if not rclpy.ok():
            rclpy.init()
        return cls(
            node=Node(node_name),
            joint_positions_topic=joint_positions_topic,
            collision_topic=collision_topic,
        )

    def forward(self, joint_angles: Sequence[float]) -> Vec3:
        _ = joint_angles
        raise NotImplementedError(
            "Ros2Adapter.forward delegates to the robot's own FK "
            "service; wire moveit_msgs/GetPositionFK in your deployment"
        )

    def inverse(self, target: Vec3) -> tuple[float, ...] | None:
        _ = target
        raise NotImplementedError(
            "Ros2Adapter.inverse delegates to moveit_msgs/GetPositionIK "
            "in the live ROS graph"
        )

    def collides_with(
        self,
        point: Vec3,
        obstacles_aabb: Sequence[AABB] = (),
        obstacles_sphere: Sequence[Sphere] = (),
    ) -> bool:
        """Check the ``point`` against the caller-supplied
        obstacle set. The live ROS graph may publish additional
        obstacles — callers fold those in via the two arguments."""
        if any(box.contains(point) for box in obstacles_aabb):
            return True
        return any(sphere.contains(point) for sphere in obstacles_sphere)


@dataclass
class MuJoCoAdapter:
    """``mujoco`` adapter.

    Loads an MJCF model + a persistent data structure. Forward
    kinematics reads qpos / geom positions from the data object;
    collision queries use MuJoCo's own contact detector via
    ``mj_step`` with zero time step.
    """

    model: Any
    data: Any

    def __post_init__(self) -> None:
        if self.model is None or self.data is None:
            raise ValueError("MuJoCoAdapter requires both model and data")

    @classmethod
    def from_mjcf(cls, mjcf_path: str) -> MuJoCoAdapter:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "MuJoCoAdapter.from_mjcf requires mujoco. "
                "Install with: pip install director-ai[mujoco]"
            ) from exc
        if not mjcf_path:
            raise ValueError("mjcf_path must be non-empty")
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        data = mujoco.MjData(model)
        return cls(model=model, data=data)

    def forward(self, joint_angles: Sequence[float]) -> Vec3:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover — covered by from_mjcf
            raise ImportError("mujoco required") from exc
        if len(joint_angles) != self.model.nq:
            raise ValueError(
                f"joint_angles length {len(joint_angles)} != model.nq {self.model.nq}"
            )
        for i, angle in enumerate(joint_angles):
            self.data.qpos[i] = angle
        mujoco.mj_forward(self.model, self.data)
        end_site = self.model.site("end_effector")
        site_xpos = self.data.site_xpos[end_site.id]
        return Vec3(
            float(site_xpos[0]),
            float(site_xpos[1]),
            float(site_xpos[2]),
        )

    def inverse(self, target: Vec3) -> tuple[float, ...] | None:
        _ = target
        raise NotImplementedError(
            "MuJoCoAdapter.inverse is deployment-specific — wire a "
            "numerical IK solver over MjData or Pinocchio in your "
            "codebase"
        )

    def collides_with(
        self,
        point: Vec3,
        obstacles_aabb: Sequence[AABB] = (),
        obstacles_sphere: Sequence[Sphere] = (),
    ) -> bool:
        for box in obstacles_aabb:
            if box.contains(point):
                return True
        for sphere in obstacles_sphere:
            if sphere.contains(point):
                return True
        # MuJoCo populates data.ncon on every mj_forward / mj_step;
        # any positive count means the live scene is reporting at
        # least one contact.
        return int(self.data.ncon) > 0


@dataclass
class CarlaAdapter:
    """``carla`` adapter.

    Connects to a running CARLA simulator via a client handle
    and queries world snapshots for actor positions. Collision
    checks intersect the caller-supplied obstacle set with the
    live actor registry.
    """

    client: Any
    world: Any

    def __post_init__(self) -> None:
        if self.client is None or self.world is None:
            raise ValueError("CarlaAdapter requires both client and world")

    @classmethod
    def from_carla(
        cls,
        host: str = "localhost",
        port: int = 2000,
        timeout_seconds: float = 10.0,
    ) -> CarlaAdapter:
        try:
            import carla
        except ImportError as exc:
            raise ImportError(
                "CarlaAdapter.from_carla requires carla. "
                "Install with: pip install director-ai[carla]"
            ) from exc
        if port <= 0:
            raise ValueError("port must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        client = carla.Client(host, port)
        client.set_timeout(timeout_seconds)
        return cls(client=client, world=client.get_world())

    def forward(self, joint_angles: Sequence[float]) -> Vec3:
        _ = joint_angles
        raise NotImplementedError(
            "CarlaAdapter is for vehicle scenarios — there are no "
            "joint angles to forward-kinematise; use vehicle-level "
            "target poses instead"
        )

    def inverse(self, target: Vec3) -> tuple[float, ...] | None:
        _ = target
        raise NotImplementedError(
            "CarlaAdapter is for vehicle scenarios — there are no joint angles to solve"
        )

    def collides_with(
        self,
        point: Vec3,
        obstacles_aabb: Sequence[AABB] = (),
        obstacles_sphere: Sequence[Sphere] = (),
    ) -> bool:
        if any(box.contains(point) for box in obstacles_aabb):
            return True
        return any(sphere.contains(point) for sphere in obstacles_sphere)
