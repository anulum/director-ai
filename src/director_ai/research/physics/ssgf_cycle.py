# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — SSGF Outer Cycle (Geometry Learning)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
SSGF-style outer geometry learning cycle integrating TCBO and PGBO.

The Stochastic Synthesis of Geometric Fields (SSGF) outer cycle:
  1. Run microcycle (Kuramoto + geometry feedback)
  2. Compute TCBO consciousness observable p_h1
  3. Compute PGBO geometry tensor h_munu
  4. Evaluate composite cost U_total
  5. Update latent geometry z via gradient descent

This is a self-contained Director-AI implementation that captures the
essential SSGF dynamics without requiring the full SCPN-CODEBASE engine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..consciousness import (
    PGBOConfig,
    PGBOEngine,
    TCBOConfig,
    TCBOController,
    TCBOObserver,
)
from .l16_closure import L16Controller
from .scpn_params import build_knm_matrix, load_omega_n

logger = logging.getLogger("DirectorAI.SSGF")


@dataclass
class SSGFState:
    """State of the SSGF outer cycle."""

    z: np.ndarray  # Latent geometry vector
    W: np.ndarray  # Decoded weight matrix (N x N)
    theta: np.ndarray  # Current phases
    R_global: float = 0.0
    p_h1: float = 0.0
    gate_open: bool = False
    U_total: float = 0.0
    costs: dict = field(default_factory=dict)
    step: int = 0


@dataclass
class SSGFConfig:
    """Configuration for the SSGF outer cycle."""

    N: int = 16
    n_micro: int = 10  # Micro-cycle steps per outer step
    dt: float = 0.01
    lr_z: float = 0.01  # Latent geometry learning rate
    sigma_g: float = 0.3  # Geometry feedback strength
    pgbo_weight: float = 0.1  # PGBO coupling weight
    noise_std: float = 0.05  # Micro-cycle noise
    field_pressure: float = 0.1  # External field
    # Cost weights
    w_micro: float = 1.0
    w_reg: float = 0.01  # Graph regularisation
    w_tcbo: float = 0.5  # TCBO cost term
    w_pgbo: float = 0.1  # PGBO cost term


def _decode_gram_softplus(z: np.ndarray, N: int) -> np.ndarray:
    """Decode latent z into a valid weight matrix W (symmetric, non-neg, zero diag).

    Uses Gram-softplus decoder: W = softplus(A) where A is symmetric from z.
    """
    # z has N*(N-1)/2 elements (upper triangle)
    A = np.zeros((N, N), dtype=np.float64)
    idx = np.triu_indices(N, k=1)
    n_params = N * (N - 1) // 2
    z_clip = z[:n_params]
    A[idx] = z_clip
    A = A + A.T  # Symmetrise
    # Softplus: log(1 + exp(x))
    W: np.ndarray = np.log1p(np.exp(np.clip(A, -20, 20)))
    np.fill_diagonal(W, 0.0)
    return W


def _spectral_bridge(W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalised Laplacian eigenpairs from W."""
    D = np.sum(W, axis=1)
    D_safe = np.maximum(D, 1e-12)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_safe))
    L_sym = np.eye(len(W)) - D_inv_sqrt @ W @ D_inv_sqrt
    L_sym = 0.5 * (L_sym + L_sym.T)  # Ensure symmetry
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    return eigvals, eigvecs


class SSGFEngine:
    """SSGF outer cycle engine integrating TCBO, PGBO, and L16 closure.

    Parameters
    ----------
    config : SSGFConfig — engine configuration.
    """

    def __init__(self, config: SSGFConfig | None = None) -> None:
        self.cfg = config if config is not None else SSGFConfig()
        N = self.cfg.N

        # Canonical SCPN parameters
        self.omega = load_omega_n()[:N]
        self.knm = build_knm_matrix()[:N, :N]

        # Latent geometry
        n_params = N * (N - 1) // 2
        self.z = np.random.randn(n_params) * 0.1

        # Decode initial W
        self.W = _decode_gram_softplus(self.z, N)

        # Phase state
        self.theta = np.random.uniform(0, 2 * np.pi, N)

        # TCBO
        tcbo_cfg = TCBOConfig(
            window_size=20, embed_dim=3, tau_delay=1, compute_every_n=1
        )
        self.tcbo_observer = TCBOObserver(N=N, config=tcbo_cfg)
        self.tcbo_controller = TCBOController()
        self.kappa = 0.3

        # PGBO
        self.pgbo = PGBOEngine(N=N, config=PGBOConfig(kappa=0.3))

        # L16 controller
        self.l16 = L16Controller(N=N)

        # Pre-allocated scratch
        self._phase_diff = np.zeros((N, N), dtype=np.float64)
        self._sin_diff = np.zeros((N, N), dtype=np.float64)

        # History (capped to prevent unbounded memory growth)
        self._MAX_HISTORY = 500
        self._cost_history: list[dict] = []
        self._state_history: list[SSGFState] = []
        self._step_count = 0

    def _micro_step(self) -> None:
        """Single Kuramoto micro-step with geometry and PGBO feedback."""
        N = self.cfg.N
        theta = self.theta

        # Phase differences
        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        np.sin(self._phase_diff, out=self._sin_diff)

        # Standard Kuramoto coupling
        coupling = np.sum(self.knm * self._sin_diff, axis=1)

        # Geometry feedback: σ_g * Σ_m W_nm sin(θ_m - θ_n)
        geo_coupling = self.cfg.sigma_g * np.sum(self.W * self._sin_diff, axis=1)

        # PGBO coupling
        u_mu, h_munu = self.pgbo.compute(theta, self.cfg.dt)
        pgbo_coupling = self.cfg.pgbo_weight * np.sum(h_munu * self._sin_diff, axis=1)

        # External field + noise
        field_term = self.cfg.field_pressure * np.cos(theta)
        noise = self.cfg.noise_std * np.sqrt(self.cfg.dt) * np.random.randn(N)

        # Euler-Maruyama step
        dtheta = self.omega + coupling + geo_coupling + pgbo_coupling + field_term
        theta_new = theta + dtheta * self.cfg.dt + noise

        # Modular phase reduction — keep phases in [0, 2π)
        self.theta = np.mod(theta_new, 2.0 * np.pi)

    def _compute_costs(self) -> dict:
        """Compute all cost terms for the outer cycle."""
        # C_micro: mean phase velocity deviation from natural frequencies
        # (proxy — use order parameter deficit)
        R = float(np.abs(np.mean(np.exp(1j * self.theta))))
        c_micro = 1.0 - R

        # Graph regularisation: ||W||_F
        r_graph = float(np.sum(self.W**2))

        # TCBO cost: deficit from consciousness threshold
        c_tcbo = max(0.0, 0.72 - self.tcbo_observer.p_h1) ** 2

        # PGBO cost: Frobenius norm of h_munu
        c_pgbo = self.pgbo.h_frob**2

        costs = {
            "c_micro": c_micro,
            "r_graph": r_graph,
            "c_tcbo": c_tcbo,
            "c_pgbo": c_pgbo,
            "R_global": R,
        }

        U_total = (
            self.cfg.w_micro * c_micro
            + self.cfg.w_reg * r_graph
            + self.cfg.w_tcbo * c_tcbo
            + self.cfg.w_pgbo * c_pgbo
        )
        # Guard against NaN/Inf in cost terms
        if not np.isfinite(U_total):
            U_total = 1e6  # Large but finite sentinel
        costs["U_total"] = U_total

        return costs

    def _gradient_step(self) -> None:
        """Update latent z via finite-difference gradient descent."""
        eps = 1e-4
        n_params = len(self.z)
        grad = np.zeros(n_params, dtype=np.float64)

        # Current cost
        W0 = _decode_gram_softplus(self.z, self.cfg.N)
        # Use simple regularisation gradient as proxy
        for i in range(min(n_params, 30)):  # Cap to keep fast
            z_plus = self.z.copy()
            z_plus[i] += eps
            W_plus = _decode_gram_softplus(z_plus, self.cfg.N)

            cost_0 = self.cfg.w_reg * float(np.sum(W0**2))
            cost_plus = self.cfg.w_reg * float(np.sum(W_plus**2))
            grad[i] = (cost_plus - cost_0) / eps

        # Gradient norm check — truncate if exploding
        grad_norm = float(np.linalg.norm(grad))
        max_grad_norm = 10.0
        if not np.isfinite(grad_norm) or grad_norm > max_grad_norm:
            if np.isfinite(grad_norm) and grad_norm > 0:
                grad = grad * (max_grad_norm / grad_norm)
                logger.warning(
                    "SSGF gradient truncated: %.4f → %.4f", grad_norm, max_grad_norm
                )
            else:
                logger.warning("SSGF gradient non-finite, zeroing")
                grad[:] = 0.0

        # Apply gradient with learning rate
        lr = self.cfg.lr_z
        if self.l16.state.refusal_active:
            lr *= self.l16.state.lr_z_scale
        self.z -= lr * grad

        # Re-decode W
        self.W = _decode_gram_softplus(self.z, self.cfg.N)

    def step(self) -> SSGFState:
        """Execute one SSGF outer-cycle step.

        Runs n_micro micro-steps, then computes TCBO/PGBO, costs,
        L16 closure, and gradient update.

        Returns
        -------
        SSGFState snapshot.
        """
        # 1. Run micro-cycle
        for _ in range(self.cfg.n_micro):
            self._micro_step()

        # 2. TCBO observation
        self.tcbo_observer.push_and_compute(self.theta, force=True)
        p_h1 = self.tcbo_observer.p_h1
        gate_open = self.tcbo_controller.is_gate_open(p_h1)

        # 3. TCBO controller → update kappa
        self.kappa = self.tcbo_controller.step(p_h1, self.kappa, self.cfg.dt)

        # 4. Compute costs
        costs = self._compute_costs()
        R_global = costs["R_global"]

        # 5. Spectral bridge (guard against degenerate W)
        try:
            _eigvals, eigvecs = _spectral_bridge(self.W)
        except np.linalg.LinAlgError:
            logger.warning(
                "Spectral bridge failed on degenerate W; using identity fallback"
            )
            eigvecs = np.eye(self.cfg.N)

        # 6. L16 closure
        phi_target = np.eye(self.cfg.N, 4)
        self.l16.step(
            theta=self.theta,
            eigvecs=eigvecs[:, :4],
            phi_target=phi_target,
            R_global=R_global,
            plv=R_global,  # Use R as PLV proxy
            costs={
                "c7": costs.get("c_micro", 0),
                "c8": 0.0,
                "c10": 0.0,
                "p_h1": p_h1,
                "h_frob": self.pgbo.h_frob,
            },
            dt=self.cfg.dt,
        )

        # 7. Gradient update on latent z
        self._gradient_step()

        # 8. Build state snapshot
        self._step_count += 1
        state = SSGFState(
            z=self.z.copy(),
            W=self.W.copy(),
            theta=self.theta.copy(),
            R_global=R_global,
            p_h1=p_h1,
            gate_open=gate_open,
            U_total=costs["U_total"],
            costs=costs,
            step=self._step_count,
        )

        self._cost_history.append(costs)
        if len(self._cost_history) > self._MAX_HISTORY:
            self._cost_history = self._cost_history[-self._MAX_HISTORY :]
        self._state_history.append(state)
        if len(self._state_history) > self._MAX_HISTORY:
            self._state_history = self._state_history[-self._MAX_HISTORY :]

        return state

    def run(self, n_steps: int = 50) -> list[SSGFState]:
        """Run the SSGF outer cycle for n_steps.

        Returns list of SSGFState snapshots.
        """
        states = []
        for _ in range(n_steps):
            states.append(self.step())
        return states

    def get_cost_history(self) -> list[dict]:
        return list(self._cost_history)

    @property
    def fiedler_value(self) -> float:
        """Algebraic connectivity (second smallest eigenvalue of Laplacian)."""
        eigvals, _ = _spectral_bridge(self.W)
        return float(eigvals[1]) if len(eigvals) > 1 else 0.0
