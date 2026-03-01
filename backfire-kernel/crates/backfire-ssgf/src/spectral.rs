// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SSGF Spectral Bridge
// Mirrors: SCPN-CODEBASE/optimizations/ssgf/spectral.py
// ─────────────────────────────────────────────────────────────────────
//! W → normalised Laplacian → eigenpairs with gauge fixing.
//!
//! Includes a pure-Rust cyclic Jacobi eigensolver for symmetric matrices.
//! For N=16, convergence is fast (typically <10 sweeps).

/// Gauge-fixing method for temporal eigenvector continuity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GaugeMethod {
    /// Align sign of each eigenvector with previous (dot-product sign).
    Sign,
    /// No gauge fixing (first call only).
    None,
}

/// Spectral bridge: computes eigenpairs of the normalised graph Laplacian.
pub struct SpectralBridge {
    n: usize,
    gauge: GaugeMethod,
    prev_eigvecs: Option<Vec<f64>>,
    // Pre-allocated scratch
    l_scratch: Vec<f64>,
    d_inv_sqrt: Vec<f64>,
    // Jacobi scratch
    jac_a: Vec<f64>,
    jac_v: Vec<f64>,
}

impl SpectralBridge {
    pub fn new(n: usize, gauge: GaugeMethod) -> Self {
        Self {
            n,
            gauge,
            prev_eigvecs: None,
            l_scratch: vec![0.0; n * n],
            d_inv_sqrt: vec![0.0; n],
            jac_a: vec![0.0; n * n],
            jac_v: vec![0.0; n * n],
        }
    }

    pub fn default_params(n: usize) -> Self {
        Self::new(n, GaugeMethod::Sign)
    }

    /// Compute normalised Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.
    ///
    /// Isolated nodes (degree ≈ 0) get zero rows/cols.
    /// Result written to internal scratch; returned as slice.
    pub fn normalised_laplacian(&mut self, w: &[f64]) -> &[f64] {
        let n = self.n;

        // Compute degree vector
        for i in 0..n {
            let mut d = 0.0;
            for j in 0..n {
                d += w[i * n + j];
            }
            if d > 1e-12 {
                self.d_inv_sqrt[i] = 1.0 / d.sqrt();
            } else {
                self.d_inv_sqrt[i] = 0.0;
            }
        }

        // L = I - D^{-1/2} W D^{-1/2}
        for i in 0..n {
            for j in 0..n {
                self.l_scratch[i * n + j] =
                    -(self.d_inv_sqrt[i] * w[i * n + j] * self.d_inv_sqrt[j]);
            }
            self.l_scratch[i * n + i] += 1.0;
        }

        // Fix isolated nodes: zero rows/cols
        for i in 0..n {
            if self.d_inv_sqrt[i] == 0.0 {
                for j in 0..n {
                    self.l_scratch[i * n + j] = 0.0;
                    self.l_scratch[j * n + i] = 0.0;
                }
            }
        }

        &self.l_scratch
    }

    /// Compute eigenpairs of the normalised Laplacian.
    ///
    /// Returns (eigenvalues, eigenvectors) with gauge fixing applied.
    /// Eigenvalues are ascending. Eigenvectors are columns of the returned matrix.
    pub fn compute_eigenpairs(
        &mut self,
        w: &[f64],
        eigvals_out: &mut [f64],
        eigvecs_out: &mut [f64],
    ) {
        let n = self.n;

        // Build Laplacian
        self.normalised_laplacian(w);

        // Copy to Jacobi scratch (will be modified in-place)
        self.jac_a.copy_from_slice(&self.l_scratch);

        // Jacobi eigendecomposition
        jacobi_eigen_symmetric(&mut self.jac_a, n, eigvals_out, &mut self.jac_v);

        // Clamp small negative eigenvalues from numerical noise
        for v in eigvals_out.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }

        // Sort eigenvalues ascending (insertion sort, N ≤ 16)
        sort_eigenpairs(eigvals_out, &mut self.jac_v, n);

        // Apply gauge fixing (inline to avoid borrow conflict)
        apply_gauge(
            &mut self.jac_v,
            self.n,
            self.gauge,
            self.prev_eigvecs.as_deref(),
        );

        // Copy to output
        eigvecs_out.copy_from_slice(&self.jac_v);

        // Store for next gauge alignment
        self.prev_eigvecs = Some(self.jac_v.clone());
    }

    /// Algebraic connectivity (second-smallest eigenvalue).
    pub fn fiedler_value(eigvals: &[f64]) -> f64 {
        if eigvals.len() < 2 {
            return 0.0;
        }
        eigvals[1]
    }

    /// Spectral gap ratio λ_1 / λ_2.
    pub fn spectral_gap(eigvals: &[f64]) -> f64 {
        if eigvals.len() < 3 {
            return 0.0;
        }
        let lam2 = eigvals[2];
        if lam2 < 1e-12 {
            return 0.0;
        }
        eigvals[1] / lam2
    }
}

/// Apply gauge fixing to eigenvectors for temporal continuity.
fn apply_gauge(eigvecs: &mut [f64], n: usize, gauge: GaugeMethod, prev_eigvecs: Option<&[f64]>) {
    match prev_eigvecs {
        None => {
            // First call: sign convention (largest-magnitude component positive)
            for col in 0..n {
                let mut max_abs = 0.0;
                let mut max_idx = 0;
                for row in 0..n {
                    let val = eigvecs[row * n + col].abs();
                    if val > max_abs {
                        max_abs = val;
                        max_idx = row;
                    }
                }
                if eigvecs[max_idx * n + col] < 0.0 {
                    for row in 0..n {
                        eigvecs[row * n + col] = -eigvecs[row * n + col];
                    }
                }
            }
        }
        Some(prev) if gauge == GaugeMethod::Sign => {
            for col in 0..n {
                let mut dot = 0.0;
                for row in 0..n {
                    dot += eigvecs[row * n + col] * prev[row * n + col];
                }
                if dot < 0.0 {
                    for row in 0..n {
                        eigvecs[row * n + col] = -eigvecs[row * n + col];
                    }
                }
            }
        }
        _ => {}
    }
}

/// Sort eigenvalues ascending, rearranging eigenvector columns accordingly.
fn sort_eigenpairs(eigvals: &mut [f64], eigvecs: &mut [f64], n: usize) {
    // Build index array
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        eigvals[a]
            .partial_cmp(&eigvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply permutation via temporary buffers
    let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigvals[i]).collect();
    eigvals[..n].copy_from_slice(&sorted_vals);

    let old_vecs = eigvecs.to_vec();
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            eigvecs[row * n + new_col] = old_vecs[row * n + old_col];
        }
    }
}

/// Cyclic Jacobi eigendecomposition for symmetric n×n matrix.
///
/// `a` is n×n row-major (destroyed — diagonal becomes eigenvalues).
/// `eigvals_out` receives the n eigenvalues (unsorted).
/// `v_out` receives the n×n eigenvector matrix (columns = eigvectors).
///
/// For N=16, converges in ≤15 sweeps. Total: O(N³ × sweeps).
fn jacobi_eigen_symmetric(a: &mut [f64], n: usize, eigvals_out: &mut [f64], v_out: &mut [f64]) {
    const MAX_SWEEPS: usize = 50;
    const TOL: f64 = 1e-14;

    // Initialize V = I
    for i in 0..n {
        for j in 0..n {
            v_out[i * n + j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    for _sweep in 0..MAX_SWEEPS {
        // Find max off-diagonal magnitude
        let mut max_off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                let v = a[p * n + q].abs();
                if v > max_off {
                    max_off = v;
                }
            }
        }
        if max_off < TOL {
            break;
        }

        // Threshold for this sweep (Jacobi threshold strategy)
        let threshold = if _sweep < 4 {
            0.2 * max_off / (n * n) as f64
        } else {
            0.0
        };

        // Sweep through all upper-triangle pairs
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < threshold {
                    continue;
                }

                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let diff = aqq - app;

                let t = if diff.abs() < 1e-300 {
                    // Equal diagonal elements: rotate by π/4
                    if apq > 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    let tau = diff / (2.0 * apq);
                    // Choose the smaller root for numerical stability
                    if tau >= 0.0 {
                        1.0 / (tau + (1.0 + tau * tau).sqrt())
                    } else {
                        -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                    }
                };

                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau_rot = s / (1.0 + c); // Rutishauser form

                // Update diagonal
                a[p * n + p] -= t * apq;
                a[q * n + q] += t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;

                // Update off-diagonal rows/cols
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = arp - s * (arq + tau_rot * arp);
                    a[p * n + r] = a[r * n + p];
                    a[r * n + q] = arq + s * (arp - tau_rot * arq);
                    a[q * n + r] = a[r * n + q];
                }

                // Accumulate rotation in eigenvector matrix
                for r in 0..n {
                    let vrp = v_out[r * n + p];
                    let vrq = v_out[r * n + q];
                    v_out[r * n + p] = vrp - s * (vrq + tau_rot * vrp);
                    v_out[r * n + q] = vrq + s * (vrp - tau_rot * vrq);
                }
            }
        }
    }

    // Extract eigenvalues from diagonal
    for i in 0..n {
        eigvals_out[i] = a[i * n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_eigenvalues() {
        let n = 4;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = (i + 1) as f64; // eigenvalues 1, 2, 3, 4
        }
        let mut eigvals = vec![0.0; n];
        let mut eigvecs = vec![0.0; n * n];
        jacobi_eigen_symmetric(&mut a, n, &mut eigvals, &mut eigvecs);
        sort_eigenpairs(&mut eigvals, &mut eigvecs, n);

        for i in 0..n {
            assert!(
                (eigvals[i] - (i + 1) as f64).abs() < 1e-10,
                "eigval[{i}] = {} expected {}",
                eigvals[i],
                i + 1
            );
        }
    }

    #[test]
    fn test_symmetric_matrix_eigenvectors_orthogonal() {
        let n = 4;
        // Build symmetric matrix
        let mut a = vec![
            4.0, 1.0, 0.5, 0.2, 1.0, 3.0, 0.8, 0.3, 0.5, 0.8, 2.0, 0.1, 0.2, 0.3, 0.1, 1.0,
        ];
        let mut eigvals = vec![0.0; n];
        let mut eigvecs = vec![0.0; n * n];
        jacobi_eigen_symmetric(&mut a, n, &mut eigvals, &mut eigvecs);

        // Check orthogonality: V^T V ≈ I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += eigvecs[k * n + i] * eigvecs[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "V^T V[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_laplacian_first_eigenvalue_zero() {
        let n = 4;
        // Connected graph: complete graph weights
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w[i * n + j] = 1.0;
                }
            }
        }

        let mut bridge = SpectralBridge::default_params(n);
        let mut eigvals = vec![0.0; n];
        let mut eigvecs = vec![0.0; n * n];
        bridge.compute_eigenpairs(&w, &mut eigvals, &mut eigvecs);

        assert!(
            eigvals[0].abs() < 1e-10,
            "First eigenvalue should be ~0, got {}",
            eigvals[0]
        );
        assert!(
            eigvals[1] > 0.01,
            "Fiedler value should be positive, got {}",
            eigvals[1]
        );
    }

    #[test]
    fn test_eigenvalue_ascending_order() {
        let n = 4;
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w[i * n + j] = 0.5 + 0.5 * ((i + j) as f64 * 0.3).sin().abs();
                }
            }
        }

        let mut bridge = SpectralBridge::default_params(n);
        let mut eigvals = vec![0.0; n];
        let mut eigvecs = vec![0.0; n * n];
        bridge.compute_eigenpairs(&w, &mut eigvals, &mut eigvecs);

        for i in 1..n {
            assert!(
                eigvals[i] >= eigvals[i - 1] - 1e-10,
                "Eigenvalues not sorted: eigvals[{}]={} < eigvals[{}]={}",
                i,
                eigvals[i],
                i - 1,
                eigvals[i - 1]
            );
        }
    }

    #[test]
    fn test_fiedler_value() {
        let eigvals = vec![0.0, 0.5, 1.2, 1.8];
        assert!((SpectralBridge::fiedler_value(&eigvals) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_spectral_gap() {
        let eigvals = vec![0.0, 0.5, 1.0, 1.5];
        assert!((SpectralBridge::spectral_gap(&eigvals) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_disconnected_graph() {
        // All zeros → no connectivity
        let n = 4;
        let w = vec![0.0; n * n];
        let mut bridge = SpectralBridge::default_params(n);
        let mut eigvals = vec![0.0; n];
        let mut eigvecs = vec![0.0; n * n];
        bridge.compute_eigenpairs(&w, &mut eigvals, &mut eigvecs);

        // All eigenvalues should be 0 for a fully disconnected graph
        for v in &eigvals {
            assert!(
                v.abs() < 1e-10,
                "Disconnected graph eigval should be 0, got {v}"
            );
        }
    }

    #[test]
    fn test_gauge_sign_continuity() {
        let n = 4;
        let mut w = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    w[i * n + j] = 1.0;
                }
            }
        }

        let mut bridge = SpectralBridge::new(n, GaugeMethod::Sign);
        let mut eigvals1 = vec![0.0; n];
        let mut eigvecs1 = vec![0.0; n * n];
        bridge.compute_eigenpairs(&w, &mut eigvals1, &mut eigvecs1);

        // Slightly perturb W
        w[1] = 1.01; // [0][1]
        w[n] = 1.01; // [1][0]

        let mut eigvals2 = vec![0.0; n];
        let mut eigvecs2 = vec![0.0; n * n];
        bridge.compute_eigenpairs(&w, &mut eigvals2, &mut eigvecs2);

        // Eigenvectors should be close (sign-aligned)
        for col in 0..n {
            let mut dot = 0.0;
            for row in 0..n {
                dot += eigvecs1[row * n + col] * eigvecs2[row * n + col];
            }
            // After gauge fixing, dot product should be positive (or zero for degenerate)
            assert!(
                dot >= -0.1,
                "Gauge fixing failed: dot product for col {col} = {dot}"
            );
        }
    }
}
