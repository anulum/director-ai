# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — L16 Mechanistic Oversight Dynamics
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Layer 16 mechanistic oversight dynamics.

Implements the Kuramoto-type UPDE integrator for the 16-layer system
with Director (L16) cybernetic closure. The L16 layer sits at the top
of the hierarchy and modulates the entire system via:

  dθ_n/dt = Ω_n + Σ_m K_nm sin(θ_m - θ_n) + F cos(θ_n) + η_n

The Director layer (n=16) has special authority: it monitors the global
order parameter R and can modulate coupling strengths, noise amplitude,
and field pressure in real-time.

This module provides:
  - ``UPDEState``: Snapshot of the 16-layer phase dynamics.
  - ``UPDEStepper``: Single-step integrator (Euler-Maruyama).
  - ``L16OversightLoop``: Full Director oversight loop integrating
    the UPDE with SEC monitoring and L16 authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .scpn_params import N_LAYERS, load_omega_n, build_knm_matrix


@dataclass
class UPDEState:
    """Snapshot of the 16-layer phase dynamics."""

    theta: np.ndarray          # (N,) current phases
    t: float = 0.0             # simulation time
    R_global: float = 0.0      # Kuramoto order parameter
    R_per_layer: Optional[np.ndarray] = None  # (N,) per-layer coherence
    step_count: int = 0

    def compute_order_parameter(self) -> float:
        """Compute the Kuramoto order parameter R."""
        self.R_global = float(np.abs(np.mean(np.exp(1j * self.theta))))
        return self.R_global


class UPDEStepper:
    """Euler-Maruyama single-step integrator for the UPDE.

    Parameters
    ----------
    omega : ndarray (N,) — natural frequencies.
    knm : ndarray (N, N) — coupling matrix.
    dt : float — timestep.
    field_pressure : float — external field strength F.
    noise_amplitude : float — stochastic noise σ.
    """

    def __init__(
        self,
        omega: Optional[np.ndarray] = None,
        knm: Optional[np.ndarray] = None,
        dt: float = 0.01,
        field_pressure: float = 0.1,
        noise_amplitude: float = 0.05,
    ) -> None:
        self.omega = omega if omega is not None else load_omega_n()
        self.knm = knm if knm is not None else build_knm_matrix()
        self.N = len(self.omega)
        self.dt = dt
        self.field_pressure = field_pressure
        self.noise_amplitude = noise_amplitude

        # Pre-allocated scratch arrays
        self._phase_diff = np.zeros((self.N, self.N), dtype=np.float64)
        self._sin_diff = np.zeros((self.N, self.N), dtype=np.float64)
        self._dtheta = np.zeros(self.N, dtype=np.float64)

    def step(self, state: UPDEState) -> UPDEState:
        """Advance the state by one timestep.

        Uses the correct Kuramoto coupling: Σ_m K_nm sin(θ_m - θ_n).
        """
        theta = state.theta

        # Phase difference matrix: phase_diff[n, m] = θ_m - θ_n
        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        np.sin(self._phase_diff, out=self._sin_diff)

        # Kuramoto coupling: Σ_m K_nm sin(θ_m - θ_n) for each n
        coupling = np.sum(self.knm * self._sin_diff, axis=1)

        # External field
        field_term = self.field_pressure * np.cos(theta)

        # Noise
        noise = self.noise_amplitude * np.sqrt(self.dt) * np.random.randn(self.N)

        # Euler-Maruyama step
        np.add(self.omega, coupling, out=self._dtheta)
        self._dtheta += field_term
        theta_new = theta + self._dtheta * self.dt + noise

        new_state = UPDEState(
            theta=theta_new,
            t=state.t + self.dt,
            step_count=state.step_count + 1,
        )
        new_state.compute_order_parameter()
        return new_state


@dataclass
class OversightSnapshot:
    """Snapshot from one L16 oversight iteration."""

    state: UPDEState
    R_global: float
    coherence_score: float        # SEC-derived (1 - V_norm)
    dV_dt: float                  # Lyapunov derivative
    is_stable: bool               # dV/dt ≤ 0
    intervention: Optional[str]   # What L16 did (if anything)


class L16OversightLoop:
    """Full L16 Director oversight loop.

    Integrates the UPDE stepper with SEC monitoring and L16 authority:
      1. Step the UPDE
      2. Evaluate SEC Lyapunov functional
      3. If dV/dt > 0 for K consecutive steps → intervene
      4. Interventions: increase coupling, reduce noise, tighten threshold

    Parameters
    ----------
    stepper : UPDEStepper — the UPDE integrator.
    sec : SECFunctional — the Lyapunov evaluator.
    intervention_window : int — consecutive instability steps to trigger.
    coupling_boost : float — multiplicative boost on intervention.
    noise_damping : float — multiplicative damping on intervention.
    """

    def __init__(
        self,
        stepper: Optional[UPDEStepper] = None,
        sec=None,
        intervention_window: int = 5,
        coupling_boost: float = 1.2,
        noise_damping: float = 0.8,
    ) -> None:
        self.stepper = stepper if stepper is not None else UPDEStepper()

        if sec is not None:
            self.sec = sec
        else:
            from .sec_functional import SECFunctional
            self.sec = SECFunctional(
                knm=self.stepper.knm,
                omega=self.stepper.omega,
            )

        self.intervention_window = intervention_window
        self.coupling_boost = coupling_boost
        self.noise_damping = noise_damping

        self._instability_count: int = 0
        self._history: List[OversightSnapshot] = []

    def step(self, state: UPDEState) -> tuple[UPDEState, OversightSnapshot]:
        """Execute one oversight iteration.

        Returns (new_state, snapshot).
        """
        prev_theta = state.theta.copy()

        # 1. UPDE step
        new_state = self.stepper.step(state)

        # 2. SEC evaluation
        sec_result = self.sec.evaluate(new_state.theta, prev_theta, self.stepper.dt)

        # 3. Stability check
        stable = self.sec.is_stable(sec_result)
        intervention = None

        if not stable:
            self._instability_count += 1
        else:
            self._instability_count = 0

        # 4. L16 intervention if sustained instability
        if self._instability_count >= self.intervention_window:
            self.stepper.knm *= self.coupling_boost
            self.stepper.noise_amplitude *= self.noise_damping
            intervention = (
                f"L16: coupling ×{self.coupling_boost}, "
                f"noise ×{self.noise_damping}"
            )
            self._instability_count = 0

        snapshot = OversightSnapshot(
            state=new_state,
            R_global=new_state.R_global,
            coherence_score=sec_result.coherence_score,
            dV_dt=sec_result.dV_dt,
            is_stable=stable,
            intervention=intervention,
        )
        self._history.append(snapshot)
        return new_state, snapshot

    def run(
        self,
        initial_theta: Optional[np.ndarray] = None,
        n_steps: int = 100,
    ) -> List[OversightSnapshot]:
        """Run the oversight loop for n_steps.

        Parameters
        ----------
        initial_theta : initial phases (random if None).
        n_steps : number of integration steps.

        Returns
        -------
        List of OversightSnapshot (one per step).
        """
        if initial_theta is None:
            initial_theta = np.random.uniform(0, 2 * np.pi, self.stepper.N)

        state = UPDEState(theta=initial_theta)
        state.compute_order_parameter()
        self._history.clear()

        for _ in range(n_steps):
            state, _ = self.step(state)

        return list(self._history)

    @property
    def history(self) -> List[OversightSnapshot]:
        return list(self._history)
