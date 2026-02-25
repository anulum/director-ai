# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — PGBO (Phase→Geometry Bridge Operator)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Phase→Geometry Bridge Operator (PGBO).

Converts coherent phase dynamics into a symmetric rank-2 tensor field
h_munu that modulates propagation, coupling, and effective curvature.

Key equations:
    u_mu   = dphi_mu - α · A_mu          (covariant phase-flow drive)
    h_munu = κ · u_mu ⊗ u_mu             (induced geometry proxy)

Properties:
  - h_munu is symmetric, smooth, differentiable
  - grows with coherence intensity (proportional to |u|²)
  - encodes directionality (anisotropy) automatically
  - optional traceless projection isolates shear effects

SCPN connections:
  - L8 (phase-locking) stabilises φ → coherent u_mu → structured h_munu
  - L10 (boundary control) injects A_mu to steer phase-flow
  - L16 (cybernetic closure) adapts α and κ via PI controllers
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PGBOConfig:
    """Configuration for the Phase→Geometry Bridge Operator."""

    D: int = 4  # dimensions (time + 3 spatial, or just N)
    alpha: float = 0.5  # coupling strength (boundary authority)
    kappa: float = 0.3  # geometry gain (phase→geometry transfer)
    alpha_min: float = 0.0
    alpha_max: float = 5.0
    kappa_min: float = 0.0
    kappa_max: float = 5.0
    u_cap: float = 10.0  # saturation cap for |u|
    traceless: bool = False  # remove trace for pure shear


class PGBOEngine:
    """Phase→Geometry Bridge Operator engine.

    Parameters
    ----------
    N : int — number of oscillator nodes (SCPN layers).
    config : PGBOConfig — operator configuration.
    """

    def __init__(
        self,
        N: int = 16,
        config: PGBOConfig | None = None,
    ) -> None:
        self.N = N
        self.cfg = config if config is not None else PGBOConfig()
        self.D = N

        # Background metric (Euclidean/flat by default)
        self.g0 = np.eye(self.D, dtype=np.float64)
        self.g0_inv = np.eye(self.D, dtype=np.float64)

        # Boundary potential (zero by default)
        self.A_mu = np.zeros(self.D, dtype=np.float64)

        # Pre-allocated outputs
        self.u_mu = np.zeros(self.D, dtype=np.float64)
        self.h_munu = np.zeros((self.D, self.D), dtype=np.float64)

        # Cached scalars
        self.u_norm: float = 0.0
        self.h_trace: float = 0.0
        self.h_frob: float = 0.0

        self._prev_theta: np.ndarray | None = None

    def set_boundary_potential(self, A_mu: np.ndarray) -> None:
        """Set the boundary injection potential (L10 handle)."""
        np.copyto(self.A_mu, A_mu[: self.D])

    def set_background_metric(self, g0: np.ndarray) -> None:
        """Set the background metric and its inverse."""
        if abs(np.linalg.det(g0)) < 1e-12:
            raise ValueError("singular metric: det(g0) ≈ 0")
        np.copyto(self.g0, g0)
        self.g0_inv = np.linalg.inv(g0)

    def compute(
        self,
        theta: np.ndarray,
        dt: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the PGBO from current phases.

        Parameters
        ----------
        theta : ndarray (N,) — current oscillator phases.
        dt : float — timestep for finite-difference gradient.

        Returns
        -------
        u_mu : ndarray (D,) — covariant phase-flow drive.
        h_munu : ndarray (D, D) — induced geometry proxy tensor.
        """
        if self._prev_theta is not None:
            dphi_mu = (theta[: self.D] - self._prev_theta[: self.D]) / max(dt, 1e-12)
        else:
            dphi_mu = np.zeros(self.D, dtype=np.float64)

        self._prev_theta = theta.copy()

        # Covariant drive
        np.copyto(self.u_mu, dphi_mu)
        self.u_mu -= self.cfg.alpha * self.A_mu

        # Saturation cap
        self.u_norm = float(np.linalg.norm(self.u_mu))
        if self.u_norm > 1e-12:
            scale = 1.0 / (1.0 + self.u_norm / self.cfg.u_cap)
            self.u_mu *= scale
            self.u_norm *= scale

        # Induced metric perturbation
        np.outer(self.u_mu, self.u_mu, out=self.h_munu)
        self.h_munu *= self.cfg.kappa

        # Optional traceless projection
        if self.cfg.traceless:
            self.h_trace = float(np.einsum("mn,mn->", self.g0_inv, self.h_munu))
            self.h_munu -= (self.h_trace / self.D) * self.g0
        else:
            self.h_trace = float(np.trace(self.h_munu))

        self.h_frob = float(np.sqrt(np.sum(self.h_munu * self.h_munu)))

        return self.u_mu, self.h_munu

    def extract_spatial(self) -> np.ndarray:
        """Extract spatial part of h_munu for propagation modulation."""
        return self.h_munu

    def extract_scalar_source(self) -> float:
        """Extract scalar backreaction source S = Tr(g_inv h)."""
        return self.h_trace

    def get_state(self) -> dict:
        """Return serialisable engine state."""
        return {
            "alpha": round(self.cfg.alpha, 6),
            "kappa": round(self.cfg.kappa, 6),
            "u_norm": round(self.u_norm, 6),
            "h_trace": round(self.h_trace, 6),
            "h_frob": round(self.h_frob, 6),
            "traceless": self.cfg.traceless,
            "D": self.D,
        }

    def reset(self) -> None:
        """Reset engine state."""
        self.u_mu[:] = 0.0
        self.h_munu[:] = 0.0
        self.A_mu[:] = 0.0
        self.u_norm = 0.0
        self.h_trace = 0.0
        self.h_frob = 0.0
        self._prev_theta = None


# ── Standalone function ──────────────────────────────────────────────


def phase_geometry_bridge(
    dphi_mu: np.ndarray,
    A_mu: np.ndarray,
    alpha: float,
    kappa: float,
    g0: np.ndarray | None = None,
    g0_inv: np.ndarray | None = None,
    traceless: bool = False,
    u_cap: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase→Geometry Bridge Operator (standalone function).

    Supports batch operation with (..., D) shaped inputs.

    Parameters
    ----------
    dphi_mu : ndarray (..., D) — phase gradient.
    A_mu : ndarray (..., D) — boundary potential.
    alpha : float — coupling strength.
    kappa : float — geometry gain.
    g0, g0_inv : ndarray (D, D) — background metric (optional).
    traceless : bool — remove trace relative to g0.
    u_cap : float — saturation cap for |u|.

    Returns
    -------
    u_mu : ndarray (..., D)
    h_munu : ndarray (..., D, D)
    """
    u_mu = dphi_mu - alpha * A_mu

    # Saturation
    u_norm = np.linalg.norm(u_mu, axis=-1, keepdims=True)
    u_mu = u_mu / (1.0 + u_norm / u_cap)

    h_munu = kappa * np.einsum("...m,...n->...mn", u_mu, u_mu)

    if traceless:
        if g0 is None or g0_inv is None:
            raise ValueError("g0 and g0_inv required for traceless projection.")
        D = h_munu.shape[-1]
        tr = np.einsum("mn,...mn->...", g0_inv, h_munu)
        h_munu = h_munu - (tr[..., None, None] / D) * g0

    return u_mu, h_munu
