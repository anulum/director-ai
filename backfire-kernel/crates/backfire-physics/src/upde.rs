// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — UPDE Kuramoto Integrator
// Mirrors: src/director_ai/research/physics/l16_mechanistic.py
// ─────────────────────────────────────────────────────────────────────
//! Euler-Maruyama integrator for the Unified Phase Dynamics Equation:
//!
//!   dθ_n/dt = Ω_n + Σ_m K_nm sin(θ_m - θ_n) + F cos(θ_n) + η_n
//!
//! Pre-allocated scratch arrays for zero-alloc hot-path execution.

use serde::{Deserialize, Serialize};

use crate::params::{build_knm_matrix, N_LAYERS, OMEGA_N};

/// Snapshot of the 16-layer phase dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UPDEState {
    /// Current phases θ_n (rad).
    pub theta: Vec<f64>,
    /// Simulation time.
    pub t: f64,
    /// Kuramoto order parameter R ∈ [0, 1].
    pub r_global: f64,
    /// Integration step counter.
    pub step_count: u64,
}

impl UPDEState {
    pub fn new(theta: Vec<f64>) -> Self {
        let mut s = Self {
            theta,
            t: 0.0,
            r_global: 0.0,
            step_count: 0,
        };
        s.compute_order_parameter();
        s
    }

    /// Random initial phases in [0, 2π).
    pub fn random(n: usize) -> Self {
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let mut rng = SimpleRng::new(seed);
        let theta: Vec<f64> = (0..n).map(|_| rng.next_f64() * std::f64::consts::TAU).collect();
        Self::new(theta)
    }

    /// Compute the Kuramoto order parameter R = |⟨e^{iθ}⟩|.
    pub fn compute_order_parameter(&mut self) -> f64 {
        let n = self.theta.len() as f64;
        if n < 1.0 {
            self.r_global = 0.0;
            return 0.0;
        }
        let (sum_sin, sum_cos) = self
            .theta
            .iter()
            .fold((0.0, 0.0), |(s, c), &th| (s + th.sin(), c + th.cos()));
        self.r_global = ((sum_sin / n).powi(2) + (sum_cos / n).powi(2)).sqrt();
        self.r_global = self.r_global.clamp(0.0, 1.0);
        self.r_global
    }
}

/// Minimal xorshift64 RNG for noise generation (no external dep).
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEAD_BEEF_CAFE_BABE } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximate standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Euler-Maruyama single-step integrator for the UPDE.
///
/// Mirrors `UPDEStepper` from `l16_mechanistic.py`.
pub struct UPDEStepper {
    pub omega: [f64; N_LAYERS],
    pub knm: [[f64; N_LAYERS]; N_LAYERS],
    pub dt: f64,
    pub field_pressure: f64,
    pub noise_amplitude: f64,
    n: usize,
    // Pre-allocated scratch
    phase_diff: [[f64; N_LAYERS]; N_LAYERS],
    sin_diff: [[f64; N_LAYERS]; N_LAYERS],
    dtheta: [f64; N_LAYERS],
    rng: SimpleRng,
}

impl UPDEStepper {
    pub fn new(dt: f64, field_pressure: f64, noise_amplitude: f64) -> Self {
        Self::with_params(OMEGA_N, build_knm_matrix(), dt, field_pressure, noise_amplitude)
    }

    pub fn with_params(
        omega: [f64; N_LAYERS],
        knm: [[f64; N_LAYERS]; N_LAYERS],
        dt: f64,
        field_pressure: f64,
        noise_amplitude: f64,
    ) -> Self {
        use std::time::SystemTime;
        let seed = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        Self {
            omega,
            knm,
            dt,
            field_pressure,
            noise_amplitude,
            n: N_LAYERS,
            phase_diff: [[0.0; N_LAYERS]; N_LAYERS],
            sin_diff: [[0.0; N_LAYERS]; N_LAYERS],
            dtheta: [0.0; N_LAYERS],
            rng: SimpleRng::new(seed),
        }
    }

    /// Default parameters: dt=0.01, F=0.1, σ=0.05.
    pub fn default_params() -> Self {
        Self::new(0.01, 0.1, 0.05)
    }

    /// Advance the state by one timestep.
    ///
    /// Uses the correct Kuramoto coupling: Σ_m K_nm sin(θ_m - θ_n).
    pub fn step(&mut self, state: &UPDEState) -> Result<UPDEState, &'static str> {
        let theta = &state.theta;
        if theta.len() != self.n {
            return Err("theta length mismatch");
        }

        // Validate input
        for &th in theta.iter() {
            if !th.is_finite() {
                return Err("input theta contains NaN or Inf");
            }
        }

        // Phase difference matrix: phase_diff[n][m] = θ_m - θ_n
        for n in 0..self.n {
            for m in 0..self.n {
                self.phase_diff[n][m] = theta[m] - theta[n];
                self.sin_diff[n][m] = self.phase_diff[n][m].sin();
            }
        }

        // Kuramoto coupling + frequency + field → dθ
        let sqrt_dt = self.dt.sqrt();
        let mut theta_new = vec![0.0f64; self.n];

        for n in 0..self.n {
            // Coupling: Σ_m K_nm sin(θ_m - θ_n)
            let mut coupling = 0.0;
            for m in 0..self.n {
                coupling += self.knm[n][m] * self.sin_diff[n][m];
            }

            // External field
            let field = self.field_pressure * theta[n].cos();

            // Noise
            let noise = self.noise_amplitude * sqrt_dt * self.rng.next_normal();

            // Euler-Maruyama
            self.dtheta[n] = self.omega[n] + coupling + field;
            theta_new[n] = theta[n] + self.dtheta[n] * self.dt + noise;

            // Modular phase reduction: θ ∈ [0, 2π)
            theta_new[n] = theta_new[n].rem_euclid(std::f64::consts::TAU);
        }

        let mut new_state = UPDEState {
            theta: theta_new,
            t: state.t + self.dt,
            r_global: 0.0,
            step_count: state.step_count + 1,
        };
        new_state.compute_order_parameter();
        Ok(new_state)
    }

    /// Run multiple steps.
    pub fn run(&mut self, initial: &UPDEState, n_steps: u64) -> Result<UPDEState, &'static str> {
        let mut state = initial.clone();
        for _ in 0..n_steps {
            state = self.step(&state)?;
        }
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upde_state_order_param() {
        // All phases equal → R = 1
        let mut state = UPDEState::new(vec![0.5; N_LAYERS]);
        let r = state.compute_order_parameter();
        assert!((r - 1.0).abs() < 1e-9, "R={r} should be ~1.0");
    }

    #[test]
    fn test_upde_state_random() {
        let state = UPDEState::random(N_LAYERS);
        assert_eq!(state.theta.len(), N_LAYERS);
        assert!(state.theta.iter().all(|&th| th >= 0.0 && th < std::f64::consts::TAU));
    }

    #[test]
    fn test_stepper_single_step() {
        let mut stepper = UPDEStepper::default_params();
        let state = UPDEState::new(vec![0.0; N_LAYERS]);
        let new_state = stepper.step(&state).unwrap();
        assert_eq!(new_state.step_count, 1);
        assert!((new_state.t - 0.01).abs() < 1e-12);
        assert!(new_state.theta.iter().all(|th| th.is_finite()));
    }

    #[test]
    fn test_stepper_phases_bounded() {
        let mut stepper = UPDEStepper::default_params();
        let state = UPDEState::random(N_LAYERS);
        let new_state = stepper.run(&state, 100).unwrap();
        assert!(new_state.theta.iter().all(|&th| th >= 0.0 && th < std::f64::consts::TAU));
    }

    #[test]
    fn test_stepper_nan_input_rejected() {
        let mut stepper = UPDEStepper::default_params();
        let state = UPDEState {
            theta: vec![f64::NAN; N_LAYERS],
            t: 0.0,
            r_global: 0.0,
            step_count: 0,
        };
        assert!(stepper.step(&state).is_err());
    }

    #[test]
    fn test_stepper_length_mismatch_rejected() {
        let mut stepper = UPDEStepper::default_params();
        let state = UPDEState::new(vec![0.0; 4]);
        assert!(stepper.step(&state).is_err());
    }

    #[test]
    fn test_synchronisation_tendency() {
        // With high coupling, R should increase (tend toward sync)
        let mut knm = build_knm_matrix();
        for n in 0..N_LAYERS {
            for m in 0..N_LAYERS {
                knm[n][m] *= 10.0; // Boost coupling
            }
        }
        // Use identical frequencies to guarantee sync
        let omega = [1.0; N_LAYERS];
        let mut stepper = UPDEStepper::with_params(omega, knm, 0.001, 0.0, 0.0);
        let initial = UPDEState::new(vec![
            0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.14, 0.13, 0.16, 0.17, 0.19, 0.1, 0.12, 0.14,
            0.15, 0.11,
        ]);
        let r_before = initial.r_global;
        let final_state = stepper.run(&initial, 1000).unwrap();
        assert!(
            final_state.r_global > r_before,
            "R should increase: {r_before} → {}",
            final_state.r_global
        );
    }
}
