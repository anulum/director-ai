// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SSGF NodeSpace
// Mirrors: SCPN-CODEBASE/optimizations/ssgf/node_space.py
// ─────────────────────────────────────────────────────────────────────
//! Pre-allocated state container for the SSGF outer cycle.
//!
//! Stores latent vector z, phases θ, weight matrix W, eigenpairs,
//! target geometry φ_target, and boundary mask.

use backfire_physics::SimpleRng;

/// Central state container with all pre-allocated arrays.
pub struct NodeSpace {
    pub n: usize,
    /// Latent vector z (length = n*(n-1)/2 for gram_softplus).
    pub z: Vec<f64>,
    /// Oscillator phases θ (length = n).
    pub theta: Vec<f64>,
    /// Weight matrix W, n×n row-major.
    pub w: Vec<f64>,
    /// Target eigenvectors φ_target, n×n row-major (columns = eigvecs).
    pub phi_target: Vec<f64>,
    /// Boundary protection mask (length = n).
    pub protected_mask: Vec<bool>,
    /// Eigenvalues of normalised Laplacian (length = n).
    pub eigvals: Vec<f64>,
    /// Eigenvectors of normalised Laplacian, n×n row-major (columns = eigvecs).
    pub eigvecs: Vec<f64>,
}

impl NodeSpace {
    /// Allocate a new NodeSpace for n oscillators.
    ///
    /// All arrays are zeroed. Call `randomise` to initialise with random values.
    pub fn new(n: usize) -> Self {
        let dim_z = n * (n - 1) / 2;
        Self {
            n,
            z: vec![0.0; dim_z],
            theta: vec![0.0; n],
            w: vec![0.0; n * n],
            phi_target: vec![0.0; n * n],
            protected_mask: vec![false; n],
            eigvals: vec![0.0; n],
            eigvecs: vec![0.0; n * n],
        }
    }

    /// Randomise z, theta, and set identity-like phi_target.
    pub fn randomise(&mut self, seed: u64) {
        let mut rng = SimpleRng::new(seed);

        // Random z in [-0.5, 0.5]
        for v in self.z.iter_mut() {
            *v = rng.next_f64() - 0.5;
        }

        // Random theta in [0, 2π)
        let tau = std::f64::consts::TAU;
        for v in self.theta.iter_mut() {
            *v = rng.next_f64() * tau;
        }

        // phi_target: identity-like (scaled Fiedler template)
        let n = self.n;
        for col in 0..n {
            for row in 0..n {
                self.phi_target[row * n + col] = if row == col { 1.0 } else { 0.0 };
            }
        }

        // Default: no boundary protection
        for v in self.protected_mask.iter_mut() {
            *v = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_space_dimensions() {
        let ns = NodeSpace::new(16);
        assert_eq!(ns.z.len(), 120); // 16*15/2
        assert_eq!(ns.theta.len(), 16);
        assert_eq!(ns.w.len(), 256);
        assert_eq!(ns.eigvals.len(), 16);
        assert_eq!(ns.eigvecs.len(), 256);
    }

    #[test]
    fn test_node_space_randomise() {
        let mut ns = NodeSpace::new(16);
        ns.randomise(42);
        // z should be in [-0.5, 0.5]
        assert!(ns.z.iter().all(|&v| (-0.5..=0.5).contains(&v)));
        // theta should be in [0, 2π)
        let tau = std::f64::consts::TAU;
        assert!(ns.theta.iter().all(|&v| v >= 0.0 && v < tau));
    }

    #[test]
    fn test_phi_target_identity() {
        let mut ns = NodeSpace::new(4);
        ns.randomise(1);
        // Diagonal should be 1.0
        for i in 0..4 {
            assert!((ns.phi_target[i * 4 + i] - 1.0).abs() < 1e-12);
        }
        // Off-diagonal should be 0.0
        assert!(ns.phi_target[1].abs() < 1e-12); // [0][1]
    }

    #[test]
    fn test_small_node_space() {
        let ns = NodeSpace::new(2);
        assert_eq!(ns.z.len(), 1); // 2*1/2
        assert_eq!(ns.w.len(), 4);
    }
}
