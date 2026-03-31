// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// Director-Class AI — micro
// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — SSGF Micro-Cycle Engine
// Mirrors: SCPN-CODEBASE/optimizations/ssgf/micro.py
// ─────────────────────────────────────────────────────────────────────
//! Kuramoto micro-cycle with geometry feedback and PGBO coupling.
//!
//!   dθ_n/dt = Ω_n
//!           + Σ_m K_nm sin(θ_m - θ_n)         (baseline coupling)
//!           + σ_g Σ_m W_nm sin(θ_m - θ_n)     (geometry feedback)
//!           + pgbo_w Σ_m h_nm sin(θ_m - θ_n)  (PGBO coupling)
//!           + F cos(θ_n)                       (external field)
//!           + η_n                              (noise)

use backfire_physics::SimpleRng;

/// Micro-cycle engine for SSGF inner loop.
pub struct MicroCycleEngine {
    n: usize,
    omega: Vec<f64>,
    k: Vec<f64>, // n×n baseline coupling, row-major
    dt_micro: f64,
    sigma_g: f64,
    noise_amp: f64,
    field_pressure: f64,
    pgbo_weight: f64,
    // Scratch arrays
    dtheta: Vec<f64>,
    sin_diffs: Vec<f64>, // n×n pre-computed sin(theta[j] - theta[i])
    rng: SimpleRng,
}

/// Dot product of two equal-length slices.
/// Kept as a simple loop so LLVM can auto-vectorize (SSE2/AVX2/NEON).
#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

impl MicroCycleEngine {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        omega: &[f64],
        k: &[f64],
        dt_micro: f64,
        sigma_g: f64,
        noise_amp: f64,
        field_pressure: f64,
        pgbo_weight: f64,
        seed: u64,
    ) -> Self {
        Self {
            n,
            omega: omega.to_vec(),
            k: k.to_vec(),
            dt_micro,
            sigma_g,
            noise_amp,
            field_pressure,
            pgbo_weight,
            dtheta: vec![0.0; n],
            sin_diffs: vec![0.0; n * n],
            rng: SimpleRng::new(seed),
        }
    }

    /// Execute one micro-step (Euler-Maruyama).
    ///
    /// `theta` is modified in-place.
    /// `w` is the geometry weight matrix (n×n row-major).
    /// `h_munu` is the optional PGBO tensor (n×n row-major).
    pub fn step(&mut self, theta: &mut [f64], w: &[f64], h_munu: Option<&[f64]>) {
        let n = self.n;
        let sqrt_dt = self.dt_micro.sqrt();
        let tau = std::f64::consts::TAU;

        // Phase 1: Pre-compute sin(theta[j] - theta[i]) into contiguous buffer.
        // Separating sin() from the weighted sums lets LLVM auto-vectorize
        // the dot-product loops below (SSE2/AVX2/NEON).
        for i in 0..n {
            let theta_i = theta[i];
            let row = i * n;
            for j in 0..n {
                self.sin_diffs[row + j] = (theta[j] - theta_i).sin();
            }
        }

        // Phase 2: Weighted sums as dot products over contiguous slices.
        for i in 0..n {
            let row = i * n;
            let coupling_k = dot_product(&self.k[row..row + n], &self.sin_diffs[row..row + n]);
            let coupling_w = dot_product(&w[row..row + n], &self.sin_diffs[row..row + n]);
            let coupling_h = match h_munu {
                Some(h) => dot_product(&h[row..row + n], &self.sin_diffs[row..row + n]),
                None => 0.0,
            };

            self.dtheta[i] = self.omega[i]
                + coupling_k
                + self.sigma_g * coupling_w
                + self.pgbo_weight * coupling_h
                + self.field_pressure * theta[i].cos()
                + self.noise_amp * sqrt_dt * self.rng.next_normal();
        }

        // Phase 3: Euler update with modular reduction.
        for i in 0..n {
            theta[i] += self.dtheta[i] * self.dt_micro;
            theta[i] = theta[i].rem_euclid(tau);
        }
    }

    /// Run n_micro micro-cycle steps.
    pub fn run_microcycle(
        &mut self,
        theta: &mut [f64],
        w: &[f64],
        n_micro: usize,
        h_munu: Option<&[f64]>,
    ) {
        for _ in 0..n_micro {
            self.step(theta, w, h_munu);
        }
    }

    /// Kuramoto order parameter R = |⟨e^{iθ}⟩| ∈ [0, 1].
    pub fn compute_order_parameter(&self, theta: &[f64]) -> f64 {
        let n = theta.len() as f64;
        if n < 1.0 {
            return 0.0;
        }
        let (sum_sin, sum_cos) = theta
            .iter()
            .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
        let r = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
        r.clamp(0.0, 1.0)
    }

    /// Phase Locking Value between oscillators i and j.
    ///
    /// PLV = |cos(θ_i - θ_j)| (instantaneous proxy).
    pub fn compute_plv(&self, theta: &[f64], i: usize, j: usize) -> f64 {
        if i >= theta.len() || j >= theta.len() {
            return 0.0;
        }
        (theta[i] - theta[j]).cos().abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine(n: usize) -> MicroCycleEngine {
        let omega = vec![1.0; n];
        let k = vec![0.5; n * n]; // uniform coupling
        MicroCycleEngine::new(n, &omega, &k, 0.001, 0.1, 0.0, 0.0, 0.0, 42)
    }

    #[test]
    fn test_step_phases_bounded() {
        let n = 4;
        let mut engine = make_engine(n);
        let mut theta = vec![0.0, 1.0, 2.0, 3.0];
        let w = vec![0.0; n * n];
        engine.run_microcycle(&mut theta, &w, 100, None);
        let tau = std::f64::consts::TAU;
        assert!(theta.iter().all(|&v| v >= 0.0 && v < tau));
    }

    #[test]
    fn test_order_parameter_sync() {
        let engine = make_engine(4);
        let theta = vec![0.5; 4]; // all equal
        let r = engine.compute_order_parameter(&theta);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_order_parameter_desync() {
        let engine = make_engine(4);
        // Evenly spaced → R ≈ 0
        let tau = std::f64::consts::TAU;
        let theta: Vec<f64> = (0..4).map(|i| i as f64 * tau / 4.0).collect();
        let r = engine.compute_order_parameter(&theta);
        assert!(r < 0.1, "R={r} should be near 0 for uniform phases");
    }

    #[test]
    fn test_plv_same_phase() {
        let engine = make_engine(4);
        let theta = vec![1.0, 1.0, 2.0, 3.0];
        let plv = engine.compute_plv(&theta, 0, 1);
        assert!((plv - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_geometry_feedback_nonzero() {
        let n = 4;
        let mut engine = MicroCycleEngine::new(
            n, &[1.0; 4], &[0.0; 16], // no baseline coupling
            0.001, 1.0, // strong geometry feedback
            0.0, 0.0, 0.0, 42,
        );
        let mut theta = vec![0.0, 1.0, 2.0, 3.0];
        // W with non-zero off-diagonal
        let mut w = vec![0.0; n * n];
        w[1] = 1.0; // [0][1]
        w[n] = 1.0; // [1][0]
        let theta_before = theta.clone();
        engine.run_microcycle(&mut theta, &w, 10, None);
        // Phases should have changed due to geometry coupling
        let moved = theta
            .iter()
            .zip(theta_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(moved, "Geometry feedback should affect phases");
    }
}
