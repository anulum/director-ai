# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — L16 Cybernetic Closure
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
L16 cybernetic closure: PI controllers, Lyapunov health, PLV gate, refusal rules.

The L16 Director layer provides cybernetic closure over the SSGF engine:
  - PI controllers with anti-windup for cost-term weights
  - H_rec Lyapunov candidate (attractor alignment + predictive error + entropy flux)
  - PLV precedence gate (L7/L9 writes blocked unless sustained PLV > threshold)
  - Refusal rules (if H_rec rises K consecutive steps → reduce lr_z, D_theta,
    widen tau_d)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PIState:
    """Internal state for a PI controller with anti-windup."""

    setpoint: float = 0.0
    Kp: float = 0.5
    Ki: float = 0.05
    integral: float = 0.0
    integral_min: float = -10.0
    integral_max: float = 10.0
    output_min: float = 0.01
    output_max: float = 10.0
    last_error: float = 0.0


def pi_step(pi: PIState, measured: float, dt: float) -> float:
    """Execute one PI controller step with anti-windup clamping.

    Parameters
    ----------
    pi : PIState
    measured : current value of controlled variable.
    dt : timestep.

    Returns
    -------
    output : controller output (weight/gain).
    """
    error = pi.setpoint - measured
    pi.integral += error * dt
    pi.integral = np.clip(pi.integral, pi.integral_min, pi.integral_max)
    output = pi.Kp * error + pi.Ki * pi.integral
    output = np.clip(output, pi.output_min, pi.output_max)
    pi.last_error = error
    return float(output)


@dataclass
class L16ControllerState:
    """Full L16 controller state snapshot."""

    H_rec: float = 0.0
    H_rec_history: list[float] = field(default_factory=list)
    plv_buffer: list[float] = field(default_factory=list)
    refusal_active: bool = False
    refusal_count: int = 0
    lr_z_scale: float = 1.0
    D_theta_scale: float = 1.0
    tau_d_scale: float = 1.0


class L16Controller:
    """L16 Director closure controller.

    Parameters
    ----------
    N : int — number of nodes.
    plv_threshold : float — PLV must exceed this for L7/L9 writes.
    plv_window : int — frames over which PLV is averaged.
    h_rec_window : int — consecutive H_rec rises that trigger refusal.
    refusal_lr_factor : float — lr_z multiplier when refusal is active.
    refusal_D_factor : float — D_theta multiplier when refusal is active.
    refusal_tau_factor : float — tau_d multiplier when refusal is active.
    """

    def __init__(
        self,
        N: int = 16,
        plv_threshold: float = 0.6,
        plv_window: int = 10,
        h_rec_window: int = 5,
        refusal_lr_factor: float = 0.5,
        refusal_D_factor: float = 0.5,
        refusal_tau_factor: float = 1.5,
    ) -> None:
        self.N = N
        self.plv_threshold = plv_threshold
        self.plv_window = plv_window
        self.h_rec_window = h_rec_window
        self.refusal_lr_factor = refusal_lr_factor
        self.refusal_D_factor = refusal_D_factor
        self.refusal_tau_factor = refusal_tau_factor

        # PI controllers for cost-term weights
        self.pi_lambda7 = PIState(setpoint=0.0, Kp=0.3, Ki=0.03)
        self.pi_lambda8 = PIState(setpoint=0.0, Kp=0.3, Ki=0.03)
        self.pi_lambda10 = PIState(setpoint=0.0, Kp=0.3, Ki=0.03)
        self.pi_nu_star = PIState(setpoint=0.5, Kp=0.5, Ki=0.05)

        self.state = L16ControllerState()

    def compute_h_rec(
        self,
        theta: np.ndarray,
        eigvecs: np.ndarray,
        phi_target: np.ndarray,
        R_global: float,
        p_h1: float = 0.0,
        h_frob: float = 0.0,
    ) -> float:
        """Compute H_rec Lyapunov candidate.

        H_rec = attractor_alignment_error + predictive_error + entropy_flux
                + h1_deficit + pgbo_energy

        Should be non-increasing for a healthy system.
        """
        # Clamp R_global to [0, 1]
        R_global = max(0.0, min(1.0, R_global))

        # Attractor alignment: ||eigvecs - phi_target|| (Frobenius)
        k = min(eigvecs.shape[1], phi_target.shape[1])
        alignment_err = float(np.sum((eigvecs[:, :k] - phi_target[:, :k]) ** 2))

        # Predictive error: 1 - R_global
        pred_err = 1.0 - R_global

        # Entropy flux: phase dispersion
        phase_var = float(np.var(np.sin(theta)) + np.var(np.cos(theta)))

        # TCBO contribution: h1 deficit
        h1_deficit = max(0.0, 0.72 - p_h1)

        # PGBO contribution: geometry proxy energy
        pgbo_energy = 0.01 * h_frob

        h_rec = alignment_err + pred_err + phase_var + h1_deficit + pgbo_energy
        return max(0.0, h_rec)

    def update_plv(self, plv: float) -> float:
        """Add a PLV sample and return windowed average."""
        self.state.plv_buffer.append(plv)
        if len(self.state.plv_buffer) > self.plv_window:
            self.state.plv_buffer = self.state.plv_buffer[-self.plv_window :]
        return float(np.mean(self.state.plv_buffer))

    def plv_gate_open(self) -> bool:
        """Check if PLV precedence gate allows L7/L9 writes."""
        if len(self.state.plv_buffer) < 1:
            return False
        return float(np.mean(self.state.plv_buffer)) > self.plv_threshold

    def check_refusal(self) -> bool:
        """Check if H_rec has been rising for h_rec_window consecutive steps."""
        hist = self.state.H_rec_history
        if len(hist) < self.h_rec_window + 1:
            return False
        recent = hist[-self.h_rec_window :]
        rising = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
        if rising:
            self.state.refusal_active = True
            self.state.refusal_count += 1
            self.state.lr_z_scale = self.refusal_lr_factor
            self.state.D_theta_scale = self.refusal_D_factor
            self.state.tau_d_scale = self.refusal_tau_factor
        else:
            self.state.refusal_active = False
            self.state.lr_z_scale = 1.0
            self.state.D_theta_scale = 1.0
            self.state.tau_d_scale = 1.0
        return self.state.refusal_active

    def step(
        self,
        theta: np.ndarray,
        eigvecs: np.ndarray,
        phi_target: np.ndarray,
        R_global: float,
        plv: float,
        costs: dict,
        dt: float,
    ) -> dict:
        """Execute one L16 controller step.

        Parameters
        ----------
        theta : current phases.
        eigvecs : current eigenvectors.
        phi_target : target eigenvectors.
        R_global : global order parameter.
        plv : current phase-locking value.
        costs : dict with C7, C8, C10, p_h1, h_frob values.
        dt : timestep.

        Returns
        -------
        dict with lambda7, lambda8, lambda10, nu_star, gate_open,
        refusal, h_rec, avg_plv, scales.
        """
        p_h1 = costs.get("p_h1", 0.0)
        h_frob = costs.get("h_frob", 0.0)
        h_rec = self.compute_h_rec(theta, eigvecs, phi_target, R_global, p_h1, h_frob)
        self.state.H_rec = h_rec
        self.state.H_rec_history.append(h_rec)
        if len(self.state.H_rec_history) > 100:
            self.state.H_rec_history = self.state.H_rec_history[-100:]

        avg_plv = self.update_plv(plv)
        gate_open = self.plv_gate_open()
        refusal = self.check_refusal()

        c7 = costs.get("C7_symbolic", costs.get("c7", 0.0))
        c8 = costs.get("C8_phase", costs.get("c8", 0.0))
        c10 = costs.get("C10_boundary", costs.get("c10", 0.0))

        lambda7 = pi_step(self.pi_lambda7, c7, dt)
        lambda8 = pi_step(self.pi_lambda8, c8, dt)
        lambda10 = pi_step(self.pi_lambda10, c10, dt)
        nu_star = pi_step(self.pi_nu_star, R_global, dt)

        return {
            "lambda7": lambda7,
            "lambda8": lambda8,
            "lambda10": lambda10,
            "nu_star": nu_star,
            "gate_open": gate_open,
            "refusal": refusal,
            "h_rec": h_rec,
            "avg_plv": avg_plv,
            "scales": {
                "lr_z_scale": self.state.lr_z_scale,
                "D_theta_scale": self.state.D_theta_scale,
                "tau_d_scale": self.state.tau_d_scale,
            },
        }

    def get_state(self) -> dict:
        """Return serialisable controller state."""
        return {
            "H_rec": self.state.H_rec,
            "refusal_active": self.state.refusal_active,
            "refusal_count": self.state.refusal_count,
            "lr_z_scale": self.state.lr_z_scale,
            "D_theta_scale": self.state.D_theta_scale,
            "tau_d_scale": self.state.tau_d_scale,
            "plv_gate_open": self.plv_gate_open(),
            "avg_plv": (
                float(np.mean(self.state.plv_buffer)) if self.state.plv_buffer else 0.0
            ),
        }
