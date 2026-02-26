// ─────────────────────────────────────────────────────────────────────
// Director-Class AI — L16 Cybernetic Closure
// ─────────────────────────────────────────────────────────────────────
//! L16 Director closure: PI controllers, Lyapunov health, PLV gate, refusal rules.
//!
//! The L16 Director layer provides cybernetic closure over the SSGF engine:
//!   - PI controllers with anti-windup for cost-term weights
//!   - H_rec Lyapunov candidate (attractor alignment + predictive error + entropy flux)
//!   - PLV precedence gate (L7/L9 writes blocked unless sustained PLV > threshold)
//!   - Refusal rules (if H_rec rises K consecutive steps → reduce lr_z, D_theta,
//!     widen tau_d)

use serde::{Deserialize, Serialize};

/// Internal state for a PI controller with anti-windup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIState {
    pub setpoint: f64,
    pub kp: f64,
    pub ki: f64,
    pub integral: f64,
    pub integral_min: f64,
    pub integral_max: f64,
    pub output_min: f64,
    pub output_max: f64,
    pub last_error: f64,
}

impl PIState {
    pub fn new(setpoint: f64, kp: f64, ki: f64) -> Self {
        Self {
            setpoint,
            kp,
            ki,
            integral: 0.0,
            integral_min: -10.0,
            integral_max: 10.0,
            output_min: 0.01,
            output_max: 10.0,
            last_error: 0.0,
        }
    }
}

impl Default for PIState {
    fn default() -> Self {
        Self::new(0.0, 0.5, 0.05)
    }
}

/// Execute one PI controller step with anti-windup clamping.
pub fn pi_step(pi: &mut PIState, measured: f64, dt: f64) -> f64 {
    let error = pi.setpoint - measured;
    pi.integral += error * dt;
    pi.integral = pi.integral.clamp(pi.integral_min, pi.integral_max);
    let output = pi.kp * error + pi.ki * pi.integral;
    let output = output.clamp(pi.output_min, pi.output_max);
    pi.last_error = error;
    output
}

/// Full L16 controller state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L16ControllerState {
    pub h_rec: f64,
    pub h_rec_history: Vec<f64>,
    pub plv_buffer: Vec<f64>,
    pub refusal_active: bool,
    pub refusal_count: u32,
    pub lr_z_scale: f64,
    pub d_theta_scale: f64,
    pub tau_d_scale: f64,
}

impl Default for L16ControllerState {
    fn default() -> Self {
        Self {
            h_rec: 0.0,
            h_rec_history: Vec::new(),
            plv_buffer: Vec::new(),
            refusal_active: false,
            refusal_count: 0,
            lr_z_scale: 1.0,
            d_theta_scale: 1.0,
            tau_d_scale: 1.0,
        }
    }
}

/// Cost inputs for an L16 controller step.
#[derive(Debug, Clone, Default)]
pub struct L16CostInputs {
    pub c7_symbolic: f64,
    pub c8_phase: f64,
    pub c10_boundary: f64,
    pub p_h1: f64,
    pub h_frob: f64,
}

/// Result of an L16 controller step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L16StepResult {
    pub lambda7: f64,
    pub lambda8: f64,
    pub lambda10: f64,
    pub nu_star: f64,
    pub gate_open: bool,
    pub refusal: bool,
    pub h_rec: f64,
    pub avg_plv: f64,
    pub lr_z_scale: f64,
    pub d_theta_scale: f64,
    pub tau_d_scale: f64,
}

/// L16 Director closure controller.
///
/// Mirrors `L16Controller` from `l16_closure.py`.
pub struct L16Controller {
    n: usize,
    plv_threshold: f64,
    plv_window: usize,
    h_rec_window: usize,
    refusal_lr_factor: f64,
    refusal_d_factor: f64,
    refusal_tau_factor: f64,
    // PI controllers for cost-term weights
    pi_lambda7: PIState,
    pi_lambda8: PIState,
    pi_lambda10: PIState,
    pi_nu_star: PIState,
    pub state: L16ControllerState,
}

impl L16Controller {
    pub fn new(
        n: usize,
        plv_threshold: f64,
        plv_window: usize,
        h_rec_window: usize,
        refusal_lr_factor: f64,
        refusal_d_factor: f64,
        refusal_tau_factor: f64,
    ) -> Self {
        Self {
            n,
            plv_threshold,
            plv_window,
            h_rec_window,
            refusal_lr_factor,
            refusal_d_factor,
            refusal_tau_factor,
            pi_lambda7: PIState::new(0.0, 0.3, 0.03),
            pi_lambda8: PIState::new(0.0, 0.3, 0.03),
            pi_lambda10: PIState::new(0.0, 0.3, 0.03),
            pi_nu_star: PIState::new(0.5, 0.5, 0.05),
            state: L16ControllerState::default(),
        }
    }

    /// Default: N=16, plv_threshold=0.6, plv_window=10, h_rec_window=5.
    pub fn default_params() -> Self {
        Self::new(16, 0.6, 10, 5, 0.5, 0.5, 1.5)
    }

    /// Compute H_rec Lyapunov candidate.
    ///
    /// H_rec = alignment_error + predictive_error + entropy_flux
    ///       + h1_deficit + pgbo_energy
    ///
    /// `eigvecs` and `phi_target` are flattened column-major matrices (N×k).
    /// If either is empty, falls back to a simplified formula.
    pub fn compute_h_rec(
        &self,
        theta: &[f64],
        eigvecs: &[f64],
        eigvecs_cols: usize,
        phi_target: &[f64],
        phi_cols: usize,
        r_global: f64,
        p_h1: f64,
        h_frob: f64,
    ) -> f64 {
        let r = r_global.clamp(0.0, 1.0);
        let h1_deficit = (0.72 - p_h1).max(0.0);

        // Simplified path when eigenvector data is unavailable
        if eigvecs.is_empty() || phi_target.is_empty() || eigvecs_cols == 0 || phi_cols == 0 {
            return ((1.0 - r) + h1_deficit).max(0.0);
        }

        // Alignment error: Frobenius norm of (eigvecs - phi_target)
        let k = eigvecs_cols.min(phi_cols);
        let rows = self.n.min(eigvecs.len() / eigvecs_cols.max(1));
        let mut alignment_err = 0.0;
        for col in 0..k {
            for row in 0..rows {
                let ev_idx = col * rows + row; // column-major
                let pt_idx = col * rows + row;
                if ev_idx < eigvecs.len() && pt_idx < phi_target.len() {
                    let diff = eigvecs[ev_idx] - phi_target[pt_idx];
                    alignment_err += diff * diff;
                }
            }
        }

        // Predictive error: 1 - R_global
        let pred_err = 1.0 - r;

        // Entropy flux: phase dispersion
        let n = theta.len() as f64;
        if n < 1.0 {
            return (alignment_err + pred_err + h1_deficit + 0.01 * h_frob).max(0.0);
        }
        let (sum_sin, sum_cos, sum_sin2, sum_cos2) = theta.iter().fold(
            (0.0, 0.0, 0.0, 0.0),
            |(ss, sc, ss2, sc2), &th| {
                let s = th.sin();
                let c = th.cos();
                (ss + s, sc + c, ss2 + s * s, sc2 + c * c)
            },
        );
        let mean_sin = sum_sin / n;
        let mean_cos = sum_cos / n;
        let var_sin = sum_sin2 / n - mean_sin * mean_sin;
        let var_cos = sum_cos2 / n - mean_cos * mean_cos;
        let phase_var = var_sin + var_cos;

        let pgbo_energy = 0.01 * h_frob;

        (alignment_err + pred_err + phase_var + h1_deficit + pgbo_energy).max(0.0)
    }

    /// Add a PLV sample and return windowed average.
    pub fn update_plv(&mut self, plv: f64) -> f64 {
        self.state.plv_buffer.push(plv);
        if self.state.plv_buffer.len() > self.plv_window {
            let start = self.state.plv_buffer.len() - self.plv_window;
            self.state.plv_buffer = self.state.plv_buffer[start..].to_vec();
        }
        let sum: f64 = self.state.plv_buffer.iter().sum();
        sum / self.state.plv_buffer.len().max(1) as f64
    }

    /// Check if PLV precedence gate allows L7/L9 writes.
    pub fn plv_gate_open(&self) -> bool {
        if self.state.plv_buffer.is_empty() {
            return false;
        }
        let sum: f64 = self.state.plv_buffer.iter().sum();
        let avg = sum / self.state.plv_buffer.len() as f64;
        avg > self.plv_threshold
    }

    /// Check if H_rec has been rising for h_rec_window consecutive steps.
    pub fn check_refusal(&mut self) -> bool {
        let hist = &self.state.h_rec_history;
        if hist.len() < self.h_rec_window + 1 {
            return false;
        }

        let start = hist.len() - self.h_rec_window;
        let recent = &hist[start..];
        let rising = recent.windows(2).all(|w| w[0] < w[1]);

        if rising {
            self.state.refusal_active = true;
            self.state.refusal_count += 1;
            self.state.lr_z_scale = self.refusal_lr_factor;
            self.state.d_theta_scale = self.refusal_d_factor;
            self.state.tau_d_scale = self.refusal_tau_factor;
        } else {
            self.state.refusal_active = false;
            self.state.lr_z_scale = 1.0;
            self.state.d_theta_scale = 1.0;
            self.state.tau_d_scale = 1.0;
        }
        self.state.refusal_active
    }

    /// Execute one L16 controller step.
    pub fn step(
        &mut self,
        theta: &[f64],
        eigvecs: &[f64],
        eigvecs_cols: usize,
        phi_target: &[f64],
        phi_cols: usize,
        r_global: f64,
        plv: f64,
        costs: &L16CostInputs,
        dt: f64,
    ) -> L16StepResult {
        let h_rec = self.compute_h_rec(
            theta,
            eigvecs,
            eigvecs_cols,
            phi_target,
            phi_cols,
            r_global,
            costs.p_h1,
            costs.h_frob,
        );
        self.state.h_rec = h_rec;
        self.state.h_rec_history.push(h_rec);
        if self.state.h_rec_history.len() > 100 {
            let start = self.state.h_rec_history.len() - 100;
            self.state.h_rec_history = self.state.h_rec_history[start..].to_vec();
        }

        let avg_plv = self.update_plv(plv);
        let gate_open = self.plv_gate_open();
        let refusal = self.check_refusal();

        let lambda7 = pi_step(&mut self.pi_lambda7, costs.c7_symbolic, dt);
        let lambda8 = pi_step(&mut self.pi_lambda8, costs.c8_phase, dt);
        let lambda10 = pi_step(&mut self.pi_lambda10, costs.c10_boundary, dt);
        let nu_star = pi_step(&mut self.pi_nu_star, r_global, dt);

        L16StepResult {
            lambda7,
            lambda8,
            lambda10,
            nu_star,
            gate_open,
            refusal,
            h_rec,
            avg_plv,
            lr_z_scale: self.state.lr_z_scale,
            d_theta_scale: self.state.d_theta_scale,
            tau_d_scale: self.state.tau_d_scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pi_step_basic() {
        let mut pi = PIState::new(1.0, 0.5, 0.05);
        let out = pi_step(&mut pi, 0.0, 0.01);
        // error=1.0, output = 0.5*1.0 + 0.05*0.01 = 0.5005
        assert!((out - 0.5005).abs() < 1e-9, "PI output={out}");
    }

    #[test]
    fn test_pi_step_anti_windup() {
        let mut pi = PIState::new(100.0, 0.5, 0.05);
        // Drive integral hard
        for _ in 0..10000 {
            pi_step(&mut pi, 0.0, 1.0);
        }
        assert!(
            pi.integral <= pi.integral_max,
            "Integral {} > max {}",
            pi.integral,
            pi.integral_max
        );
    }

    #[test]
    fn test_pi_step_output_clamped() {
        let mut pi = PIState::new(0.0, 0.5, 0.05);
        let out = pi_step(&mut pi, 1000.0, 0.01);
        assert!(
            out >= pi.output_min && out <= pi.output_max,
            "Output {out} out of [{}, {}]",
            pi.output_min,
            pi.output_max
        );
    }

    #[test]
    fn test_h_rec_simplified_path() {
        let ctrl = L16Controller::default_params();
        let theta = vec![0.5; 16];
        let h = ctrl.compute_h_rec(&theta, &[], 0, &[], 0, 0.8, 0.5, 0.0);
        // Simplified: (1 - 0.8) + max(0, 0.72 - 0.5) = 0.2 + 0.22 = 0.42
        assert!((h - 0.42).abs() < 1e-9, "H_rec simplified={h}");
    }

    #[test]
    fn test_h_rec_with_eigvecs() {
        let ctrl = L16Controller::default_params();
        let theta = vec![1.0; 16];
        // 16x2 identity-like eigvecs and target (aligned → alignment_err ≈ 0)
        let eigvecs: Vec<f64> = (0..32).map(|i| if i % 17 == 0 { 1.0 } else { 0.0 }).collect();
        let phi_target = eigvecs.clone();
        let h = ctrl.compute_h_rec(&theta, &eigvecs, 2, &phi_target, 2, 1.0, 0.72, 0.0);
        // alignment_err=0, pred_err=0, h1_deficit=0, pgbo=0, phase_var≈0 (all same phase)
        assert!(h < 0.01, "H_rec aligned should be ~0, got {h}");
    }

    #[test]
    fn test_plv_gate_closed_initially() {
        let ctrl = L16Controller::default_params();
        assert!(!ctrl.plv_gate_open());
    }

    #[test]
    fn test_plv_gate_opens_with_high_plv() {
        let mut ctrl = L16Controller::default_params();
        for _ in 0..10 {
            ctrl.update_plv(0.8);
        }
        assert!(ctrl.plv_gate_open(), "Gate should be open with PLV=0.8");
    }

    #[test]
    fn test_plv_gate_stays_closed_with_low_plv() {
        let mut ctrl = L16Controller::default_params();
        for _ in 0..10 {
            ctrl.update_plv(0.3);
        }
        assert!(!ctrl.plv_gate_open(), "Gate should be closed with PLV=0.3");
    }

    #[test]
    fn test_refusal_activates_on_rising_h_rec() {
        let mut ctrl = L16Controller::default_params();
        // Push 6 monotonically rising values (need h_rec_window+1=6)
        for i in 0..6 {
            ctrl.state.h_rec_history.push(i as f64);
        }
        assert!(ctrl.check_refusal(), "Refusal should activate on rising H_rec");
        assert!(ctrl.state.refusal_active);
        assert_eq!(ctrl.state.refusal_count, 1);
        assert!((ctrl.state.lr_z_scale - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_refusal_deactivates_on_flat_h_rec() {
        let mut ctrl = L16Controller::default_params();
        // Rising → refusal on
        for i in 0..6 {
            ctrl.state.h_rec_history.push(i as f64);
        }
        ctrl.check_refusal();
        assert!(ctrl.state.refusal_active);

        // Now push flat values
        for _ in 0..6 {
            ctrl.state.h_rec_history.push(5.0);
        }
        ctrl.check_refusal();
        assert!(!ctrl.state.refusal_active, "Refusal should deactivate on flat H_rec");
        assert!((ctrl.state.lr_z_scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_full_step() {
        let mut ctrl = L16Controller::default_params();
        let theta = vec![0.5; 16];
        let costs = L16CostInputs {
            c7_symbolic: 0.1,
            c8_phase: 0.2,
            c10_boundary: 0.15,
            p_h1: 0.8,
            h_frob: 0.5,
        };
        let result = ctrl.step(&theta, &[], 0, &[], 0, 0.7, 0.65, &costs, 0.01);
        assert!(result.h_rec >= 0.0, "h_rec should be non-negative");
        assert!(result.lambda7 > 0.0, "lambda7 should be positive");
        assert!(result.lambda8 > 0.0, "lambda8 should be positive");
        assert!(result.lambda10 > 0.0, "lambda10 should be positive");
        assert!(result.nu_star > 0.0, "nu_star should be positive");
    }

    #[test]
    fn test_plv_window_truncation() {
        let mut ctrl = L16Controller::new(16, 0.6, 5, 5, 0.5, 0.5, 1.5);
        for i in 0..20 {
            ctrl.update_plv(i as f64 * 0.1);
        }
        assert_eq!(ctrl.state.plv_buffer.len(), 5, "PLV buffer should be truncated to window");
    }
}
