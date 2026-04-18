# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — geometry primitives

"""3-D geometry primitives used by the grounding hook.

When ``backfire_kernel`` (the Rust FFI) is installed, the
containment / intersection tests dispatch to ``rust_*`` functions
that are bit-exact with the Python reference; otherwise the pure-
Python path runs. The dispatcher is resolved at import time so
the hot path does no per-call ``hasattr`` lookups.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

try:
    from backfire_kernel import (
        rust_aabb_contains as _rust_aabb_contains,
    )
    from backfire_kernel import (
        rust_sphere_contains as _rust_sphere_contains,
    )
    from backfire_kernel import (
        rust_sphere_intersects_aabb as _rust_sphere_intersects_aabb,
    )
    from backfire_kernel import (
        rust_sphere_intersects_sphere as _rust_sphere_intersects_sphere,
    )

    _RUST_GEOM_AVAILABLE = True
except ImportError:  # pragma: no cover — optional accelerator
    _RUST_GEOM_AVAILABLE = False


@dataclass(frozen=True)
class Vec3:
    """Immutable 3-D point / vector."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for component in (self.x, self.y, self.z):
            if not math.isfinite(component):
                raise ValueError(f"Vec3 components must be finite; got {component!r}")

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def distance(self, other: Vec3) -> float:
        return (self - other).norm()

    def clamp(self, low: Vec3, high: Vec3) -> Vec3:
        """Return a new vector with each component clamped into
        ``[low, high]``. Used by workspace bounding."""
        if low.x > high.x or low.y > high.y or low.z > high.z:
            raise ValueError("low must be componentwise <= high")
        return Vec3(
            max(low.x, min(self.x, high.x)),
            max(low.y, min(self.y, high.y)),
            max(low.z, min(self.z, high.z)),
        )


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box."""

    min_corner: Vec3
    max_corner: Vec3

    def __post_init__(self) -> None:
        if (
            self.min_corner.x > self.max_corner.x
            or self.min_corner.y > self.max_corner.y
            or self.min_corner.z > self.max_corner.z
        ):
            raise ValueError("AABB.min_corner must be componentwise <= max_corner")

    def contains(self, point: Vec3) -> bool:
        if _RUST_GEOM_AVAILABLE:
            # cast documents the narrowing at the PyO3 boundary —
            # the Rust function signature returns ``bool`` but the
            # FFI binding is untyped at the Python level.
            return cast(
                bool,
                _rust_aabb_contains(
                    (self.min_corner.x, self.min_corner.y, self.min_corner.z),
                    (self.max_corner.x, self.max_corner.y, self.max_corner.z),
                    (point.x, point.y, point.z),
                ),
            )
        return (
            self.min_corner.x <= point.x <= self.max_corner.x
            and self.min_corner.y <= point.y <= self.max_corner.y
            and self.min_corner.z <= point.z <= self.max_corner.z
        )

    def intersects(self, other: AABB) -> bool:
        return not (
            self.max_corner.x < other.min_corner.x
            or self.min_corner.x > other.max_corner.x
            or self.max_corner.y < other.min_corner.y
            or self.min_corner.y > other.max_corner.y
            or self.max_corner.z < other.min_corner.z
            or self.min_corner.z > other.max_corner.z
        )

    def expand(self, margin: float) -> AABB:
        """Return a new AABB uniformly expanded by ``margin``
        in every direction. Negative margins shrink."""
        delta = Vec3(margin, margin, margin)
        return AABB(
            min_corner=self.min_corner - delta,
            max_corner=self.max_corner + delta,
        )

    @property
    def centre(self) -> Vec3:
        return Vec3(
            (self.min_corner.x + self.max_corner.x) / 2.0,
            (self.min_corner.y + self.max_corner.y) / 2.0,
            (self.min_corner.z + self.max_corner.z) / 2.0,
        )


@dataclass(frozen=True)
class Sphere:
    """Solid sphere in 3-D."""

    centre: Vec3
    radius: float

    def __post_init__(self) -> None:
        if self.radius < 0:
            raise ValueError(f"Sphere.radius must be non-negative; got {self.radius!r}")

    def contains(self, point: Vec3) -> bool:
        if _RUST_GEOM_AVAILABLE:
            return cast(
                bool,
                _rust_sphere_contains(
                    (self.centre.x, self.centre.y, self.centre.z),
                    self.radius,
                    (point.x, point.y, point.z),
                ),
            )
        return self.centre.distance(point) <= self.radius

    def intersects(self, other: Sphere) -> bool:
        if _RUST_GEOM_AVAILABLE:
            return cast(
                bool,
                _rust_sphere_intersects_sphere(
                    (self.centre.x, self.centre.y, self.centre.z),
                    self.radius,
                    (other.centre.x, other.centre.y, other.centre.z),
                    other.radius,
                ),
            )
        return self.centre.distance(other.centre) <= self.radius + other.radius

    def intersects_aabb(self, box: AABB) -> bool:
        """Closest-point-to-sphere test."""
        if _RUST_GEOM_AVAILABLE:
            return cast(
                bool,
                _rust_sphere_intersects_aabb(
                    (self.centre.x, self.centre.y, self.centre.z),
                    self.radius,
                    (box.min_corner.x, box.min_corner.y, box.min_corner.z),
                    (box.max_corner.x, box.max_corner.y, box.max_corner.z),
                ),
            )
        closest = self.centre.clamp(box.min_corner, box.max_corner)
        return self.centre.distance(closest) <= self.radius
