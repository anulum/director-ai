// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — Canonical Parameters
// ─────────────────────────────────────────────────────────────────────
//! Canonical Omega_n (natural frequencies) and Knm (coupling matrix)
//! for the 16-layer SCPN hierarchy.
//!
//! Data sources:
//!   - Omega_n: parameter_catalogue_full.yaml lines 542-925
//!   - Knm: HolonomicAtlas/src/knm_tools/knm_matrix_calculator.py

pub const N_LAYERS: usize = 16;

/// Canonical natural frequencies (rad/s) for the 16 SCPN layers.
/// Values from parameter_catalogue_full.yaml lines 542-925.
/// Some coincide with brainwave bands (e.g. L2 ≈ 40 Hz × 2π) but are
/// catalogued constants, not computed from π/τ.
#[allow(clippy::approx_constant)]
pub const OMEGA_N: [f64; N_LAYERS] = [
    1.329,   // L1  — Quantum Biological
    251.327, // L2  — Neurochemical (≈40 Hz × 2π)
    0.628,   // L3  — Genomic-Epigenetic
    31.416,  // L4  — Cellular Synchrony (≈5 Hz × 2π)
    6.283,   // L5  — Intentional Frame (≈1 Hz × 2π)
    49.199,  // L6  — Gaian / Schumann (≈7.83 Hz × 2π)
    3.142,   // L7  — Geometrical-Symbolic
    0.105,   // L8  — Cosmic Information
    1.571,   // L9  — Memory Manifold
    0.942,   // L10 — Identity Manifold
    0.209,   // L11 — Noospheric-Cultural
    0.042,   // L12 — Ecological-Gaian
    0.013,   // L13 — Source-Field
    0.006,   // L14 — Transdimensional
    0.003,   // L15 — Consilium
    0.991,   // L16 — The Director
];

pub const LAYER_NAMES: [&str; N_LAYERS] = [
    "Quantum Biological",
    "Neurochemical",
    "Genomic-Epigenetic",
    "Cellular Synchrony",
    "Intentional Frame",
    "Gaian Embodiment",
    "Geometrical-Symbolic",
    "Cosmic Information",
    "Memory Manifold",
    "Identity Manifold",
    "Noospheric-Cultural",
    "Ecological-Gaian",
    "Source-Field",
    "Transdimensional",
    "Consilium",
    "The Director",
];

const K_BASE: f64 = 0.45;
const DECAY_ALPHA: f64 = 0.3;

/// Calibration anchors (0-indexed layer pairs → coupling value).
const CALIBRATION_ANCHORS: [(usize, usize, f64); 4] = [
    (0, 1, 0.302), // L1 ↔ L2
    (1, 2, 0.201), // L2 ↔ L3
    (2, 3, 0.252), // L3 ↔ L4
    (3, 4, 0.154), // L4 ↔ L5
];

/// Cross-hierarchy boosts (0-indexed).
const CROSS_BOOSTS: [(usize, usize, f64); 2] = [
    (0, 15, 0.05), // L1 ↔ L16: quantum-director bridge
    (4, 6, 0.15),  // L5 ↔ L7: intentional-symbolic bridge
];

/// Build the 16×16 Knm inter-layer coupling matrix.
///
/// Construction:
///   1. Exponential decay baseline: K_nm = K_base * exp(-α * |n-m|)
///   2. Overwrite calibration anchors
///   3. Apply cross-hierarchy boosts
///   4. Symmetrise and zero diagonal
///
pub fn build_knm_matrix() -> [[f64; N_LAYERS]; N_LAYERS] {
    let mut k = [[0.0f64; N_LAYERS]; N_LAYERS];

    // Step 1: exponential-decay baseline
    for (n, row) in k.iter_mut().enumerate() {
        for (m, cell) in row.iter_mut().enumerate() {
            if n != m {
                let dist = n.abs_diff(m);
                *cell = K_BASE * (-DECAY_ALPHA * dist as f64).exp();
            }
        }
    }

    // Step 2: overwrite calibration anchors
    for &(i, j, val) in &CALIBRATION_ANCHORS {
        k[i][j] = val;
        k[j][i] = val;
    }

    // Step 3: apply cross-hierarchy boosts
    for &(i, j, val) in &CROSS_BOOSTS {
        k[i][j] = val;
        k[j][i] = val;
    }

    // Step 4: symmetrise and zero diagonal
    #[allow(clippy::needless_range_loop)]
    for n in 0..N_LAYERS {
        for m in (n + 1)..N_LAYERS {
            let avg = 0.5 * (k[n][m] + k[m][n]);
            k[n][m] = avg;
            k[m][n] = avg;
        }
        k[n][n] = 0.0;
    }

    k
}

/// Return canonical Omega_n as a Vec (convenience for dynamic dispatch).
pub fn load_omega_n() -> Vec<f64> {
    OMEGA_N.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omega_n_length() {
        assert_eq!(OMEGA_N.len(), N_LAYERS);
    }

    #[test]
    fn test_omega_n_positive() {
        assert!(OMEGA_N.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_knm_symmetric() {
        let k = build_knm_matrix();
        for n in 0..N_LAYERS {
            for m in 0..N_LAYERS {
                assert!(
                    (k[n][m] - k[m][n]).abs() < 1e-12,
                    "K[{n},{m}] != K[{m},{n}]"
                );
            }
        }
    }

    #[test]
    fn test_knm_zero_diagonal() {
        let k = build_knm_matrix();
        for n in 0..N_LAYERS {
            assert_eq!(k[n][n], 0.0, "K[{n},{n}] should be 0");
        }
    }

    #[test]
    fn test_knm_non_negative() {
        let k = build_knm_matrix();
        for n in 0..N_LAYERS {
            for m in 0..N_LAYERS {
                assert!(k[n][m] >= 0.0, "K[{n},{m}] = {} < 0", k[n][m]);
            }
        }
    }

    #[test]
    fn test_calibration_anchor_l1_l2() {
        let k = build_knm_matrix();
        assert!((k[0][1] - 0.302).abs() < 1e-9, "K[0,1] = {}", k[0][1]);
    }

    #[test]
    fn test_cross_boost_l1_l16() {
        let k = build_knm_matrix();
        assert!((k[0][15] - 0.05).abs() < 1e-9, "K[0,15] = {}", k[0][15]);
    }

    #[test]
    fn test_layer_names_length() {
        assert_eq!(LAYER_NAMES.len(), N_LAYERS);
    }
}
