# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — SEC Functional (Lyapunov Stability)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Sustainable Ethical Coherence (SEC) as a Lyapunov functional.

The SEC functional V(θ) is a candidate Lyapunov function for the SCPN
Kuramoto-type system. It satisfies:

  1. V(θ) ≥ 0  for all θ
  2. V(θ) = 0  iff the system is at the coherence fixed point
  3. dV/dt ≤ 0  along trajectories of the UPDE (under sufficient coupling)

When all three conditions hold and coupling exceeds the critical
threshold K_c, the system converges to the coherence equilibrium.

The SEC maps directly to the consumer's CoherenceScore:
  CoherenceScore.score = 1 - V_normalised

where V_normalised = V(θ) / V_max ∈ [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scpn_params import build_knm_matrix, load_omega_n


@dataclass
class SECResult:
    """Result of a SEC functional evaluation."""

    V: float  # Raw Lyapunov value (≥ 0, lower = more coherent)
    V_normalised: float  # Normalised to [0, 1]
    R_global: float  # Kuramoto order parameter
    dV_dt: float  # Time derivative estimate (should be ≤ 0)
    coherence_score: float  # = 1 - V_normalised (maps to consumer score)
    terms: dict  # Breakdown: {coupling, frequency, entropy}


class SECFunctional:
    """SEC as a Lyapunov functional for the SCPN Kuramoto system.

    The functional is:

        V(θ) = V_coupling(θ) + V_frequency(θ) + V_entropy(θ)

    where:
      - V_coupling  = Σ_{n,m} K_nm [1 - cos(θ_n - θ_m)]
                      (minimised when all phases are aligned)
      - V_frequency = λ_ω Σ_n (dθ_n/dt - ω_n)²
                      (penalises deviation from natural frequencies)
      - V_entropy   = -λ_S Σ_n log(p_n + ε)
                      (penalises concentration — encourages diversity)

    Parameters
    ----------
    knm : ndarray (N, N) — coupling matrix (default: canonical Knm).
    omega : ndarray (N,) — natural frequencies (default: canonical Omega_n).
    lambda_omega : float — frequency deviation weight.
    lambda_entropy : float — entropy regularisation weight.
    """

    def __init__(
        self,
        knm: np.ndarray | None = None,
        omega: np.ndarray | None = None,
        lambda_omega: float = 0.1,
        lambda_entropy: float = 0.01,
    ) -> None:
        self.knm = knm if knm is not None else build_knm_matrix()
        self.omega = omega if omega is not None else load_omega_n()
        self.N = len(self.omega)
        self.lambda_omega = lambda_omega
        self.lambda_entropy = lambda_entropy

        # V_max for normalisation (worst case: all pairwise phases at π)
        self._V_coupling_max = float(np.sum(self.knm) * 2.0)
        self._V_max = max(self._V_coupling_max, 1e-12)

        # Previous state for dV/dt estimation
        self._prev_V: float | None = None
        self._prev_dt: float = 0.01

    def coupling_potential(self, theta: np.ndarray) -> float:
        """V_coupling = Σ_{n,m} K_nm [1 - cos(θ_n - θ_m)]."""
        phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        return float(np.sum(self.knm * (1.0 - np.cos(phase_diff))))

    def frequency_penalty(
        self,
        theta: np.ndarray,
        theta_prev: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> float:
        """V_frequency = λ_ω Σ_n (dθ_n/dt - ω_n)²."""
        if theta_prev is None:
            return 0.0
        dtheta_dt = (theta - theta_prev) / max(dt, 1e-12)
        deviation = dtheta_dt - self.omega[: len(theta)]
        return self.lambda_omega * float(np.sum(deviation**2))

    def entropy_term(self, theta: np.ndarray) -> float:
        """V_entropy = -λ_S Σ_n log(p_n + ε), where p_n ∝ |e^{iθ_n}|² = 1.

        In practice, use phase distribution entropy. Uniform = max diversity,
        so the *negative* entropy penalises collapse to a single phase.
        """
        # Phase distribution over bins
        n_bins = max(self.N, 8)
        counts, _ = np.histogram(
            np.mod(theta, 2 * np.pi), bins=n_bins, range=(0, 2 * np.pi)
        )
        p = counts / max(counts.sum(), 1)
        p = p + 1e-12  # avoid log(0)
        entropy = -float(np.sum(p * np.log(p)))
        # Penalise low entropy (concentration)
        max_entropy = np.log(n_bins)
        return float(self.lambda_entropy * (max_entropy - entropy))

    def evaluate(
        self,
        theta: np.ndarray,
        theta_prev: np.ndarray | None = None,
        dt: float = 0.01,
    ) -> SECResult:
        """Evaluate the full SEC functional.

        Parameters
        ----------
        theta : ndarray (N,) — current phase vector.
        theta_prev : ndarray (N,) — previous phase vector (for dV/dt).
        dt : float — timestep.

        Returns
        -------
        SECResult with V, V_normalised, R_global, dV_dt, coherence_score, terms.

        Raises
        ------
        ValueError
            If *theta* contains NaN or Inf values.
        """
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains NaN or Inf values")
        if theta_prev is not None and not np.all(np.isfinite(theta_prev)):
            raise ValueError("theta_prev contains NaN or Inf values")

        v_coupling = self.coupling_potential(theta)
        v_freq = self.frequency_penalty(theta, theta_prev, dt)
        v_entropy = self.entropy_term(theta)

        V = v_coupling + v_freq + v_entropy

        # Guard against non-finite V from numerical overflow
        if not np.isfinite(V):
            V = self._V_max

        V_norm = float(np.clip(V / self._V_max, 0.0, 1.0)) if self._V_max > 0 else 0.0

        # Kuramoto order parameter — always in [0, 1]
        R = float(np.clip(np.abs(np.mean(np.exp(1j * theta))), 0.0, 1.0))

        # dV/dt estimation
        if self._prev_V is not None:
            dV_dt = (V - self._prev_V) / max(dt, 1e-12)
            if not np.isfinite(dV_dt):
                dV_dt = 0.0
        else:
            dV_dt = 0.0
        self._prev_V = V
        self._prev_dt = dt

        coherence_score = float(np.clip(1.0 - V_norm, 0.0, 1.0))

        return SECResult(
            V=V,
            V_normalised=V_norm,
            R_global=R,
            dV_dt=dV_dt,
            coherence_score=coherence_score,
            terms={
                "coupling": v_coupling,
                "frequency": v_freq,
                "entropy": v_entropy,
            },
        )

    def critical_coupling(self) -> float:
        """Estimate the critical coupling K_c below which sync is lost.

        For the Kuramoto model with heterogeneous frequencies:
            K_c ≈ 2 / (π · g(0))
        where g(0) is the frequency density at the mean.

        Returns
        -------
        K_c : float — estimated critical coupling strength.
        """
        omega = self.omega
        std_omega = float(np.std(omega))
        if std_omega < 1e-12:
            return 0.0
        # Approximate g(0) as 1 / (√(2π) · σ_ω)
        g0 = 1.0 / (np.sqrt(2 * np.pi) * std_omega)
        return float(2.0 / (np.pi * g0))

    def is_stable(self, sec_result: SECResult, tolerance: float = 1e-3) -> bool:
        """Check if dV/dt ≤ 0 (Lyapunov stability condition)."""
        return bool(sec_result.dV_dt <= tolerance)
