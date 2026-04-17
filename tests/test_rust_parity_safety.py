# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Rust / Python parity tests for safety hooks

"""Verifies the Rust accelerator functions shipped in
``backfire_kernel`` produce bit-exact output against the Python
reference path for every safety-hook primitive. Skipped when the
optional ``backfire_kernel`` wheel is not installed."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import math
from typing import cast

import pytest

_RUST_AVAILABLE = importlib.util.find_spec("backfire_kernel") is not None
pytestmark = pytest.mark.skipif(
    not _RUST_AVAILABLE, reason="backfire_kernel not installed"
)

if _RUST_AVAILABLE:
    from backfire_kernel import (
        rust_aabb_contains,
        rust_derive_challenge_indices,
        rust_merkle_auth_path,
        rust_merkle_root,
        rust_merkle_walk_path,
        rust_sphere_contains,
        rust_sphere_intersects_aabb,
        rust_sphere_intersects_sphere,
        rust_two_link_ik,
    )


# ─── geometry ────────────────────────────────────────────────────────


def _py_aabb_contains(
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
    p: tuple[float, float, float],
) -> bool:
    return (
        mn[0] <= p[0] <= mx[0]
        and mn[1] <= p[1] <= mx[1]
        and mn[2] <= p[2] <= mx[2]
    )


def _py_sphere_contains(
    c: tuple[float, float, float], r: float, p: tuple[float, float, float]
) -> bool:
    dx, dy, dz = c[0] - p[0], c[1] - p[1], c[2] - p[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) <= r


def _py_sphere_intersects_sphere(
    c1: tuple[float, float, float],
    r1: float,
    c2: tuple[float, float, float],
    r2: float,
) -> bool:
    dx, dy, dz = c1[0] - c2[0], c1[1] - c2[1], c1[2] - c2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz) <= r1 + r2


def _py_sphere_intersects_aabb(
    c: tuple[float, float, float],
    r: float,
    mn: tuple[float, float, float],
    mx: tuple[float, float, float],
) -> bool:
    cx = min(max(c[0], mn[0]), mx[0])
    cy = min(max(c[1], mn[1]), mx[1])
    cz = min(max(c[2], mn[2]), mx[2])
    dx, dy, dz = c[0] - cx, c[1] - cy, c[2] - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz) <= r


class TestGeometryParity:
    @pytest.mark.parametrize(
        "point",
        [(0, 0, 0), (1, 1, 1), (0.5, 0.5, 0.5), (-0.01, 0, 0), (1.0, 1.0, 1.0)],
    )
    def test_aabb_contains_matches(self, point):
        mn = (0.0, 0.0, 0.0)
        mx = (1.0, 1.0, 1.0)
        assert rust_aabb_contains(mn, mx, point) == _py_aabb_contains(mn, mx, point)

    @pytest.mark.parametrize(
        "point",
        [(0, 0, 0), (0.9, 0, 0), (1.0, 0, 0), (1.01, 0, 0), (0.5, 0.5, 0.5)],
    )
    def test_sphere_contains_matches(self, point):
        centre = (0.0, 0.0, 0.0)
        assert rust_sphere_contains(centre, 1.0, point) == _py_sphere_contains(
            centre, 1.0, point
        )

    @pytest.mark.parametrize(
        "d",
        [0.0, 1.0, 1.999, 2.0, 2.001, 5.0],
    )
    def test_sphere_intersects_sphere_matches(self, d):
        c1 = (0.0, 0.0, 0.0)
        c2 = (d, 0.0, 0.0)
        assert rust_sphere_intersects_sphere(c1, 1.0, c2, 1.0) == (
            _py_sphere_intersects_sphere(c1, 1.0, c2, 1.0)
        )

    @pytest.mark.parametrize(
        "centre",
        [(0, 0, 0), (1.5, 0, 0), (2.5, 0, 0), (0.5, 0.5, 0.5), (-10, -10, -10)],
    )
    def test_sphere_intersects_aabb_matches(self, centre):
        mn = (0.0, 0.0, 0.0)
        mx = (1.0, 1.0, 1.0)
        assert rust_sphere_intersects_aabb(centre, 1.0, mn, mx) == (
            _py_sphere_intersects_aabb(centre, 1.0, mn, mx)
        )


# ─── kinematics ─────────────────────────────────────────────────────


def _py_two_link_ik(
    l1: float,
    l2: float,
    base: tuple[float, float],
    target: tuple[float, float],
    elbow_up: bool,
) -> tuple[float, float] | None:
    dx = target[0] - base[0]
    dy = target[1] - base[1]
    r = math.sqrt(dx * dx + dy * dy)
    if r > l1 + l2 + 1e-9 or r < abs(l1 - l2) - 1e-9:
        return None
    cos_theta2 = max(-1.0, min(1.0, (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)))
    sin_mag = math.sqrt(max(0.0, 1.0 - cos_theta2 * cos_theta2))
    sin_theta2 = sin_mag if elbow_up else -sin_mag
    theta2 = math.atan2(sin_theta2, cos_theta2)
    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    theta1 = math.atan2(dy, dx) - math.atan2(k2, k1)
    return (theta1, theta2)


class TestTwoLinkIKParity:
    @pytest.mark.parametrize("elbow_up", [True, False])
    @pytest.mark.parametrize(
        "target",
        [
            (1.5, 0.0),
            (0.0, 1.5),
            (1.0, 1.0),
            (2.0, 0.0),  # exactly at reach
            (0.0, 0.1),  # near base
            (-1.0, 0.3),
        ],
    )
    def test_reachable_target_matches(self, target, elbow_up):
        py = _py_two_link_ik(1.0, 1.0, (0.0, 0.0), target, elbow_up)
        rs = rust_two_link_ik(1.0, 1.0, (0.0, 0.0), target, elbow_up)
        if py is None:
            assert rs is None
            return
        assert rs is not None
        # Bit-exactness is not guaranteed across math.atan2 vs Rust's
        # f64::atan2, but they differ by < 1 ULP in practice; use a
        # tight 1e-12 tolerance to allow last-bit rounding.
        assert rs[0] == pytest.approx(py[0], abs=1e-12)
        assert rs[1] == pytest.approx(py[1], abs=1e-12)

    def test_unreachable_target_both_none(self):
        py = _py_two_link_ik(1.0, 1.0, (0.0, 0.0), (100.0, 0.0), True)
        rs = rust_two_link_ik(1.0, 1.0, (0.0, 0.0), (100.0, 0.0), True)
        assert py is None and rs is None

    def test_bad_link_length_raises(self):
        with pytest.raises(ValueError, match="link lengths"):
            rust_two_link_ik(-1.0, 1.0, (0, 0), (0.5, 0), True)


# ─── Merkle / SHA-256 ──────────────────────────────────────────────


def _py_hash_node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _py_next_level(level: list[bytes]) -> list[bytes]:
    out: list[bytes] = []
    i = 0
    while i < len(level):
        left = level[i]
        right = level[i + 1] if i + 1 < len(level) else left
        out.append(_py_hash_node(left, right))
        i += 2
    return out


def _py_merkle_root(leaves: list[bytes]) -> bytes:
    level = list(leaves)
    while len(level) > 1:
        level = _py_next_level(level)
    return level[0]


def _py_merkle_auth_path(leaves: list[bytes], index: int) -> list[bytes]:
    path: list[bytes] = []
    level = list(leaves)
    i = index
    while len(level) > 1:
        sibling = i ^ 1
        path.append(level[sibling] if sibling < len(level) else level[i])
        level = _py_next_level(level)
        i //= 2
    return path


def _py_walk_path(leaf: bytes, index: int, siblings: list[bytes]) -> bytes:
    node = leaf
    i = index
    for sibling in siblings:
        node = (
            _py_hash_node(node, sibling)
            if i % 2 == 0
            else _py_hash_node(sibling, node)
        )
        i //= 2
    return node


class TestMerkleParity:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 16, 33])
    def test_root_matches(self, n):
        leaves = [bytes([i] * 32) for i in range(n)]
        py = _py_merkle_root(leaves)
        rs = bytes(rust_merkle_root(leaves))
        assert py == rs

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 16, 33])
    def test_auth_path_matches(self, n):
        leaves = [bytes([i] * 32) for i in range(n)]
        for idx in range(n):
            py = _py_merkle_auth_path(leaves, idx)
            rs = [bytes(b) for b in rust_merkle_auth_path(leaves, idx)]
            assert py == rs

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8])
    def test_walk_path_round_trip(self, n):
        leaves = [bytes([i] * 32) for i in range(n)]
        root = rust_merkle_root(leaves)
        for idx in range(n):
            siblings = rust_merkle_auth_path(leaves, idx)
            reconstructed = bytes(
                rust_merkle_walk_path(leaves[idx], idx, siblings)
            )
            assert reconstructed == bytes(root)

    def test_empty_leaves_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            rust_merkle_root([])

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            rust_merkle_auth_path([b"x" * 32, b"y" * 32], 5)


# ─── challenge derivation ────────────────────────────────────────────


_CHALLENGE_KEY = b"director-ai/zk-attestation/challenge-derive/v1"


def _py_derive_challenge(
    seed: bytes, sample_count: int, challenge_size: int
) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(indices) < challenge_size and counter < challenge_size * 16:
        block = hmac.new(
            _CHALLENGE_KEY,
            seed + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        for offset in range(0, len(block), 4):
            if len(indices) >= challenge_size:
                break
            chunk = int.from_bytes(block[offset : offset + 4], "big")
            idx = chunk % sample_count
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        counter += 1
    return indices


class TestChallengeDerivationParity:
    @pytest.mark.parametrize("sample_count", [8, 16, 100, 1000])
    @pytest.mark.parametrize("challenge_size", [1, 4, 16, 32])
    def test_indices_match(self, sample_count, challenge_size):
        seed = b"seed-material-" + str((sample_count, challenge_size)).encode()
        py = _py_derive_challenge(seed, sample_count, challenge_size)
        rs = cast(
            list[int],
            rust_derive_challenge_indices(seed, sample_count, challenge_size),
        )
        assert py == rs

    def test_zero_challenge_size_returns_empty(self):
        assert rust_derive_challenge_indices(b"seed", 100, 0) == []

    def test_zero_sample_count_raises(self):
        with pytest.raises(ValueError, match="sample_count"):
            rust_derive_challenge_indices(b"seed", 0, 4)
