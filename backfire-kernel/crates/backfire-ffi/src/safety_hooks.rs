// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — Rust acceleration for cyber-physical + zk-attestation hooks

//! Rust implementations of the compute-heavy primitives inside the
//! :mod:`director_ai.core.cyber_physical` and
//! :mod:`director_ai.core.zk_attestation` subpackages.
//!
//! * geometry: AABB / sphere containment + intersection tests
//! * kinematics: analytical two-link inverse kinematics (planar)
//! * Merkle: SHA-256 tree root, authentication path, walk
//! * challenge derivation: HMAC-SHA256 PRF expansion over a root
//!
//! Every function matches the Python reference bit-exactly so the
//! dispatcher can be hot-swapped without behavioural drift.
//! Parity is enforced by ``tests/test_rust_parity_safety.py``.

use hmac::{Hmac, Mac};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

const NODE_SEP: u8 = 0x01;

// ─── geometry ────────────────────────────────────────────────────────

/// Return True when ``point`` lies within the axis-aligned box
/// ``[min_corner, max_corner]``. All three tuples are xyz in the
/// same frame.
#[pyfunction]
pub fn rust_aabb_contains(
    min_corner: (f64, f64, f64),
    max_corner: (f64, f64, f64),
    point: (f64, f64, f64),
) -> bool {
    point.0 >= min_corner.0
        && point.0 <= max_corner.0
        && point.1 >= min_corner.1
        && point.1 <= max_corner.1
        && point.2 >= min_corner.2
        && point.2 <= max_corner.2
}

/// Return True when ``point`` lies within a sphere of ``radius``
/// centred at ``centre``.
#[pyfunction]
pub fn rust_sphere_contains(centre: (f64, f64, f64), radius: f64, point: (f64, f64, f64)) -> bool {
    let dx = centre.0 - point.0;
    let dy = centre.1 - point.1;
    let dz = centre.2 - point.2;
    (dx * dx + dy * dy + dz * dz).sqrt() <= radius
}

/// Return True when two spheres intersect (centres and radii).
#[pyfunction]
pub fn rust_sphere_intersects_sphere(
    c1: (f64, f64, f64),
    r1: f64,
    c2: (f64, f64, f64),
    r2: f64,
) -> bool {
    let dx = c1.0 - c2.0;
    let dy = c1.1 - c2.1;
    let dz = c1.2 - c2.2;
    (dx * dx + dy * dy + dz * dz).sqrt() <= r1 + r2
}

/// Return True when a sphere ``(centre, radius)`` intersects the
/// AABB ``[min_corner, max_corner]`` using the standard
/// closest-point-on-box test.
#[pyfunction]
pub fn rust_sphere_intersects_aabb(
    centre: (f64, f64, f64),
    radius: f64,
    min_corner: (f64, f64, f64),
    max_corner: (f64, f64, f64),
) -> bool {
    let closest_x = centre.0.clamp(min_corner.0, max_corner.0);
    let closest_y = centre.1.clamp(min_corner.1, max_corner.1);
    let closest_z = centre.2.clamp(min_corner.2, max_corner.2);
    let dx = centre.0 - closest_x;
    let dy = centre.1 - closest_y;
    let dz = centre.2 - closest_z;
    (dx * dx + dy * dy + dz * dz).sqrt() <= radius
}

// ─── kinematics ─────────────────────────────────────────────────────

/// Analytical inverse kinematics for a planar two-link chain.
///
/// ``l1`` / ``l2`` are positive link lengths; ``base_xy`` is the
/// chain's origin in the plane; ``target_xy`` is the requested
/// end-effector position. ``elbow_up`` chooses the upper branch
/// (positive ``theta_2``) when True, the lower branch otherwise.
///
/// Returns ``Some((theta_1, theta_2))`` in radians on success, or
/// ``None`` when the target is unreachable (outside the
/// ``|l1 - l2| <= reach <= l1 + l2`` annulus).
#[pyfunction]
#[allow(clippy::similar_names)]
pub fn rust_two_link_ik(
    l1: f64,
    l2: f64,
    base_xy: (f64, f64),
    target_xy: (f64, f64),
    elbow_up: bool,
) -> PyResult<Option<(f64, f64)>> {
    if !(l1 > 0.0 && l2 > 0.0) {
        return Err(PyValueError::new_err(
            "link lengths must be positive and finite",
        ));
    }
    let dx = target_xy.0 - base_xy.0;
    let dy = target_xy.1 - base_xy.1;
    let r2 = dx * dx + dy * dy;
    let r = r2.sqrt();
    let max_reach = l1 + l2;
    let min_reach = (l1 - l2).abs();
    // Use a small numerical tolerance so targets at the reach
    // boundary (exactly `l1 + l2`) are accepted despite
    // floating-point round-off.
    let eps = 1e-9;
    if r > max_reach + eps || r < min_reach - eps {
        return Ok(None);
    }

    let cos_theta2 = ((r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
    let sin_theta2_abs = (1.0 - cos_theta2 * cos_theta2).max(0.0).sqrt();
    let sin_theta2 = if elbow_up {
        sin_theta2_abs
    } else {
        -sin_theta2_abs
    };
    let theta2 = sin_theta2.atan2(cos_theta2);
    let k1 = l1 + l2 * cos_theta2;
    let k2 = l2 * sin_theta2;
    let theta1 = dy.atan2(dx) - k2.atan2(k1);
    Ok(Some((theta1, theta2)))
}

// ─── Merkle over SHA-256 ────────────────────────────────────────────

fn hash_node(left: &[u8], right: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update([NODE_SEP]);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().to_vec()
}

fn next_level(level: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(level.len().div_ceil(2));
    let mut i = 0;
    while i < level.len() {
        let left = &level[i];
        let right = if i + 1 < level.len() {
            &level[i + 1]
        } else {
            left
        };
        out.push(hash_node(left, right));
        i += 2;
    }
    out
}

/// Merkle root of a non-empty sequence of pre-hashed leaves.
///
/// The tree uses ``SHA256(0x01 || left || right)`` at each
/// internal node and duplicates the last leaf when a level is
/// odd — the same convention the Python reference uses, so
/// ``rust_merkle_root`` produces bit-identical output.
#[pyfunction]
pub fn rust_merkle_root(leaves: Vec<Vec<u8>>) -> PyResult<Vec<u8>> {
    if leaves.is_empty() {
        return Err(PyValueError::new_err("leaves must be non-empty"));
    }
    let mut level: Vec<Vec<u8>> = leaves;
    while level.len() > 1 {
        level = next_level(&level);
    }
    Ok(level.into_iter().next().unwrap())
}

/// Authentication path from ``leaves[index]`` up to the root, as
/// the ordered sibling hashes exclusive of the root. When the
/// sibling does not exist at an odd-tailed level, the node itself
/// is returned (matches the Python reference's duplicate-last rule).
#[pyfunction]
pub fn rust_merkle_auth_path(leaves: Vec<Vec<u8>>, index: usize) -> PyResult<Vec<Vec<u8>>> {
    if leaves.is_empty() {
        return Err(PyValueError::new_err("leaves must be non-empty"));
    }
    if index >= leaves.len() {
        return Err(PyValueError::new_err(format!(
            "index {} out of range (n={})",
            index,
            leaves.len()
        )));
    }
    let mut level = leaves;
    let mut i = index;
    let mut path: Vec<Vec<u8>> = Vec::new();
    while level.len() > 1 {
        let sibling_idx = i ^ 1;
        let sibling = if sibling_idx < level.len() {
            level[sibling_idx].clone()
        } else {
            level[i].clone()
        };
        path.push(sibling);
        level = next_level(&level);
        i /= 2;
    }
    Ok(path)
}

/// Walk a leaf up through the auth path and return the computed
/// root. Bit-exact with the Python ``_walk_path`` reference.
#[pyfunction]
pub fn rust_merkle_walk_path(leaf: Vec<u8>, index: usize, siblings: Vec<Vec<u8>>) -> Vec<u8> {
    let mut node = leaf;
    let mut i = index;
    for sibling in siblings {
        node = if i.is_multiple_of(2) {
            hash_node(&node, &sibling)
        } else {
            hash_node(&sibling, &node)
        };
        i /= 2;
    }
    node
}

// ─── challenge derivation ───────────────────────────────────────────

const CHALLENGE_HMAC_KEY: &[u8] = b"director-ai/zk-attestation/challenge-derive/v1";

/// Expand a commitment root into ``challenge_size`` distinct
/// indices in ``[0, sample_count)`` via HMAC-SHA256 PRF counter
/// expansion. Matches the Python ``_pick_challenge`` helper so
/// both sides derive the same set.
#[pyfunction]
pub fn rust_derive_challenge_indices(
    seed_material: Vec<u8>,
    sample_count: usize,
    challenge_size: usize,
) -> PyResult<Vec<usize>> {
    if sample_count == 0 {
        return Err(PyValueError::new_err("sample_count must be positive"));
    }
    if challenge_size == 0 {
        return Ok(Vec::new());
    }

    let mut indices: Vec<usize> = Vec::with_capacity(challenge_size);
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut counter: u64 = 0;
    let max_counter = (challenge_size as u64) * 16;
    while indices.len() < challenge_size && counter < max_counter {
        let mut mac = HmacSha256::new_from_slice(CHALLENGE_HMAC_KEY)
            .map_err(|e| PyValueError::new_err(format!("HMAC init failed: {e}")))?;
        mac.update(&seed_material);
        mac.update(&counter.to_be_bytes());
        let block = mac.finalize().into_bytes();
        // Each SHA-256 digest gives us 32 bytes = 8 × 4-byte slots.
        let mut offset = 0;
        while offset + 4 <= block.len() {
            if indices.len() >= challenge_size {
                break;
            }
            let chunk = u32::from_be_bytes([
                block[offset],
                block[offset + 1],
                block[offset + 2],
                block[offset + 3],
            ]);
            let idx = (chunk as usize) % sample_count;
            if seen.insert(idx) {
                indices.push(idx);
            }
            offset += 4;
        }
        counter += 1;
    }
    Ok(indices)
}

// ─── module registration ────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_aabb_contains, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sphere_contains, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sphere_intersects_sphere, m)?)?;
    m.add_function(wrap_pyfunction!(rust_sphere_intersects_aabb, m)?)?;
    m.add_function(wrap_pyfunction!(rust_two_link_ik, m)?)?;
    m.add_function(wrap_pyfunction!(rust_merkle_root, m)?)?;
    m.add_function(wrap_pyfunction!(rust_merkle_auth_path, m)?)?;
    m.add_function(wrap_pyfunction!(rust_merkle_walk_path, m)?)?;
    m.add_function(wrap_pyfunction!(rust_derive_challenge_indices, m)?)?;
    Ok(())
}
