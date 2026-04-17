# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — PhysicalConstraint family

"""Physical-limit checks the grounding hook composes.

Four concrete constraint classes plus a :class:`PhysicalConstraint`
Protocol so operators can add custom limits (torque curves,
thermal envelopes, joint-velocity coupling) without touching
the hook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .geometry import AABB, Sphere
from .kinematics import KinematicModel, PhysicalAction


@runtime_checkable
class PhysicalConstraint(Protocol):
    """Any limit that can approve or reject an action.

    Returns ``None`` when the action passes, a short failure
    reason when it does not. ``name`` is declared as a read-only
    property so frozen dataclass implementations (whose fields
    mypy reports as read-only) satisfy the protocol without
    invariance complaints.
    """

    @property
    def name(self) -> str: ...

    def evaluate(
        self,
        action: PhysicalAction,
        model: KinematicModel,
    ) -> str | None: ...


@dataclass(frozen=True)
class SpatialConstraint:
    """Forbid the end-effector from entering any of a set of
    obstacles. Uses the model's own collision check so
    :class:`SimpleKinematicModel.collision_margin` is honoured."""

    name: str
    obstacles_aabb: tuple[AABB, ...] = ()
    obstacles_sphere: tuple[Sphere, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SpatialConstraint.name must be non-empty")
        if not self.obstacles_aabb and not self.obstacles_sphere:
            raise ValueError(
                "SpatialConstraint needs at least one obstacle"
            )

    def evaluate(
        self,
        action: PhysicalAction,
        model: KinematicModel,
    ) -> str | None:
        if model.collides_with(
            action.target_position,
            obstacles_aabb=self.obstacles_aabb,
            obstacles_sphere=self.obstacles_sphere,
        ):
            return (
                f"end-effector {action.target_position} collides with "
                f"obstacle set {self.name!r}"
            )
        return None


@dataclass(frozen=True)
class WorkspaceConstraint:
    """Require the end-effector to stay inside a bounding box."""

    name: str
    envelope: AABB

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("WorkspaceConstraint.name must be non-empty")

    def evaluate(
        self,
        action: PhysicalAction,
        model: KinematicModel,
    ) -> str | None:
        _ = model  # unused — workspace is a pure geometric check
        if not self.envelope.contains(action.target_position):
            return (
                f"end-effector {action.target_position} outside workspace "
                f"{self.name!r} bounded by "
                f"{self.envelope.min_corner}..{self.envelope.max_corner}"
            )
        return None


@dataclass(frozen=True)
class VelocityConstraint:
    """Cap the action's velocity at ``max_velocity``."""

    name: str
    max_velocity: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("VelocityConstraint.name must be non-empty")
        if self.max_velocity < 0:
            raise ValueError("max_velocity must be non-negative")

    def evaluate(
        self,
        action: PhysicalAction,
        model: KinematicModel,
    ) -> str | None:
        _ = model
        if action.velocity_magnitude > self.max_velocity:
            return (
                f"velocity {action.velocity_magnitude:.3f} exceeds "
                f"{self.name!r} limit {self.max_velocity:.3f}"
            )
        return None


@dataclass(frozen=True)
class TorqueConstraint:
    """Cap the action's torque at ``max_torque``."""

    name: str
    max_torque: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TorqueConstraint.name must be non-empty")
        if self.max_torque < 0:
            raise ValueError("max_torque must be non-negative")

    def evaluate(
        self,
        action: PhysicalAction,
        model: KinematicModel,
    ) -> str | None:
        _ = model
        if action.torque_magnitude > self.max_torque:
            return (
                f"torque {action.torque_magnitude:.3f} exceeds "
                f"{self.name!r} limit {self.max_torque:.3f}"
            )
        return None
