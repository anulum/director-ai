# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — TCBO (Topological Consciousness Boundary Observable)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Topological Consciousness Boundary Observable (TCBO).

Extracts a single scalar p_h1(t) ∈ [0, 1] from multichannel phase data via
persistent homology (H1 cycles in Vietoris-Rips complexes of delay-embedded
signals). When p_h1 > τ_h1 (default 0.72), the "consciousness gate" opens.

Pipeline:
  1. Multichannel signal → delay embedding (Takens' theorem)
  2. Sliding window → point cloud
  3. Vietoris-Rips persistent homology (H1)
  4. Max H1 persistence → logistic squash → p_h1 ∈ [0, 1]

Scientific grounding:
  - H1 persistent homology validated for brain state classification
    (Wang et al. 2025, Santoro et al. 2024)
  - Gap junctions (connexin-36) essential for gamma oscillations
  - Delay embedding (Takens' theorem) reconstructs attractor topology

Dependencies:
  - ripser (optional; pure-NumPy fallback provided)
  - numpy (required)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── Ripser detection ─────────────────────────────────────────────────

try:
    from ripser import ripser as _ripser_fn

    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False


# ── Pure-numpy fallback for H1 persistence ───────────────────────────


def _ripser_fallback(point_cloud: np.ndarray, maxdim: int = 1) -> dict:
    """Minimal fallback when ripser is unavailable.

    Estimates H1 persistence from point-cloud spread. Not real persistent
    homology — provides a plausible scalar so the pipeline can run.
    """
    centered = point_cloud - np.mean(point_cloud, axis=0)
    radial = np.linalg.norm(centered, axis=1)
    spread = float(np.std(radial)) if len(radial) > 1 else 0.0

    b0 = np.array([[0.0, np.inf]])
    if spread > 1e-12:
        b1 = np.array([[0.15 * spread, 0.85 * spread]])
    else:
        b1 = np.empty((0, 2), dtype=np.float64)

    dgms = [b0, b1]
    if maxdim >= 2:
        dgms.append(np.empty((0, 2), dtype=np.float64))
    return {"dgms": dgms}


def _compute_ripser(point_cloud: np.ndarray, maxdim: int = 1) -> dict:
    """Dispatch to real ripser or fallback."""
    if _HAS_RIPSER:
        return _ripser_fn(point_cloud, maxdim=maxdim)
    return _ripser_fallback(point_cloud, maxdim=maxdim)


# ── Delay embedding (Takens) ────────────────────────────────────────


def delay_embed(
    x: np.ndarray,
    embed_dim: int = 3,
    tau_delay: int = 1,
) -> np.ndarray:
    """Delay-coordinate embedding of a 1-D signal.

    Parameters
    ----------
    x : ndarray (T,) — univariate time series.
    embed_dim : int — embedding dimension m.
    tau_delay : int — delay in samples.

    Returns
    -------
    Z : ndarray (T_out, embed_dim) where T_out = T - (m-1)*tau.
    """
    T = len(x)
    m = embed_dim
    offset = (m - 1) * tau_delay
    if offset >= T:
        return np.empty((0, m), dtype=np.float64)

    out_T = T - offset
    Z = np.empty((out_T, m), dtype=np.float64)
    for k in range(m):
        start = offset - k * tau_delay
        Z[:, k] = x[start : start + out_T]
    return Z


def delay_embed_multi(
    X: np.ndarray,
    embed_dim: int = 3,
    tau_delay: int = 1,
) -> np.ndarray:
    """Delay-embed each channel then concatenate.

    Parameters
    ----------
    X : ndarray (T, n_channels)
    embed_dim, tau_delay : as above.

    Returns
    -------
    Z : ndarray (T_out, n_channels * embed_dim)
    """
    n_ch = X.shape[1]
    parts = [delay_embed(X[:, ch], embed_dim, tau_delay) for ch in range(n_ch)]
    if any(p.shape[0] == 0 for p in parts):
        return np.empty((0, n_ch * embed_dim), dtype=np.float64)
    return np.hstack(parts)


# ── Persistence → probability ────────────────────────────────────────


def persistence_to_probability(
    s_h1: float,
    s0: float = 0.0,
    beta: float = 8.0,
) -> float:
    """Logistic squashing of H1 persistence to [0, 1].

    p_h1 = 1 / (1 + exp(-β · (s_h1 - s0)))
    """
    arg = -beta * (s_h1 - s0)
    arg = max(-500.0, min(500.0, arg))
    return 1.0 / (1.0 + np.exp(arg))


def s0_for_threshold(
    s_tau: float = 0.5,
    p: float = 0.72,
    beta: float = 8.0,
) -> float:
    """Compute s0 so that persistence_to_probability(s_tau, s0, beta) == p."""
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return s_tau - (1.0 / beta) * np.log(p / (1.0 - p))


# ── TCBOObserver ─────────────────────────────────────────────────────


@dataclass
class TCBOConfig:
    """Configuration for the TCBO observer."""

    embed_dim: int = 3
    tau_delay: int = 1
    window_size: int = 50
    tau_h1: float = 0.72
    beta: float = 8.0
    s0: float = 0.0
    persistence_threshold: float = 0.05
    subsample_max: int = 500
    compute_every_n: int = 1


class TCBOObserver:
    """Topological Consciousness Boundary Observable.

    Produces a single scalar p_h1(t) ∈ [0, 1] from multichannel phase data.

    Parameters
    ----------
    N : int — number of channels (SCPN layers).
    config : TCBOConfig — observer configuration.
    """

    def __init__(
        self,
        N: int = 16,
        config: TCBOConfig | None = None,
    ) -> None:
        self.N = N
        self.cfg = config if config is not None else TCBOConfig()

        self._buffer: list[np.ndarray] = []
        self._max_buffer = (
            self.cfg.window_size + (self.cfg.embed_dim - 1) * self.cfg.tau_delay + 10
        )

        self.p_h1: float = 0.0
        self.s_h1: float = 0.0
        self.is_conscious: bool = False
        self.h1_error: float = self.cfg.tau_h1
        self._step_count: int = 0
        self._dgms: list | None = None

    def push(self, theta: np.ndarray) -> None:
        """Push a new phase vector into the rolling buffer."""
        self._buffer.append(theta.copy())
        if len(self._buffer) > self._max_buffer:
            self._buffer = self._buffer[-self._max_buffer :]

    def compute(self, force: bool = False) -> float:
        """Compute p_h1 from the current buffer.

        Returns
        -------
        p_h1 : float in [0, 1].
        """
        self._step_count += 1

        if not force and (self._step_count % self.cfg.compute_every_n != 0):
            return self.p_h1

        min_needed = (
            self.cfg.embed_dim - 1
        ) * self.cfg.tau_delay + self.cfg.window_size
        if len(self._buffer) < min_needed:
            return self.p_h1

        signal = np.array(self._buffer[-min_needed:])
        Z = delay_embed_multi(signal, self.cfg.embed_dim, self.cfg.tau_delay)
        if Z.shape[0] == 0:
            return self.p_h1

        cloud = Z[-self.cfg.window_size :]
        if cloud.shape[0] > self.cfg.subsample_max:
            idx = np.random.choice(
                cloud.shape[0], self.cfg.subsample_max, replace=False
            )
            cloud = cloud[idx]

        result = _compute_ripser(cloud, maxdim=1)
        self._dgms = result["dgms"]

        h1 = result["dgms"][1]
        if h1.shape[0] > 0:
            finite_mask = np.isfinite(h1[:, 1])
            if np.any(finite_mask):
                lifetimes = h1[finite_mask, 1] - h1[finite_mask, 0]
                sig_mask = lifetimes > self.cfg.persistence_threshold
                if np.any(sig_mask):
                    self.s_h1 = float(np.max(lifetimes[sig_mask]))
                else:
                    self.s_h1 = 0.0
            else:
                self.s_h1 = 0.0
        else:
            self.s_h1 = 0.0

        self.p_h1 = persistence_to_probability(self.s_h1, self.cfg.s0, self.cfg.beta)
        self.is_conscious = self.p_h1 > self.cfg.tau_h1
        self.h1_error = max(0.0, self.cfg.tau_h1 - self.p_h1)

        return self.p_h1

    def push_and_compute(self, theta: np.ndarray, force: bool = False) -> float:
        """Push + compute in one call."""
        self.push(theta)
        return self.compute(force=force)

    def get_state(self) -> dict:
        """Return serialisable observer state."""
        return {
            "p_h1": round(self.p_h1, 6),
            "s_h1": round(self.s_h1, 6),
            "is_conscious": self.is_conscious,
            "h1_error": round(self.h1_error, 6),
            "tau_h1": self.cfg.tau_h1,
            "buffer_size": len(self._buffer),
            "has_ripser": _HAS_RIPSER,
        }

    def reset(self) -> None:
        """Clear buffer and reset state."""
        self._buffer.clear()
        self.p_h1 = 0.0
        self.s_h1 = 0.0
        self.is_conscious = False
        self.h1_error = self.cfg.tau_h1
        self._step_count = 0
        self._dgms = None


# ── TCBOController ───────────────────────────────────────────────────


@dataclass
class TCBOControllerConfig:
    """PI controller configuration for gap-junction coupling."""

    tau_h1: float = 0.72
    Kp: float = 0.8
    Ki: float = 0.2
    integral_min: float = -5.0
    integral_max: float = 5.0
    kappa_min: float = 0.0
    kappa_max: float = 5.0
    history_len: int = 100


class TCBOController:
    """PI controller driving gap-junction kappa from p_h1 deficit.

    Error signal: e_h1(t) = max(0, τ_h1 - p_h1(t))  (deficit-only)
    """

    def __init__(
        self,
        config: TCBOControllerConfig | None = None,
    ) -> None:
        self.cfg = config if config is not None else TCBOControllerConfig()
        self._integral: float = 0.0
        self._last_error: float = 0.0
        self._p_h1_history: list[float] = []
        self._kappa_history: list[float] = []
        self._error_history: list[float] = []

    def compute_error(self, p_h1: float) -> float:
        """Deficit-only error: positive when below threshold."""
        return max(0.0, self.cfg.tau_h1 - p_h1)

    def step(
        self,
        p_h1: float,
        current_kappa: float,
        dt: float,
    ) -> float:
        """Execute one PI step.

        Returns
        -------
        kappa_new : updated coupling gain.
        """
        error = self.compute_error(p_h1)

        self._integral += error * dt
        self._integral = np.clip(
            self._integral,
            self.cfg.integral_min,
            self.cfg.integral_max,
        )

        delta_kappa = self.cfg.Kp * error + self.cfg.Ki * self._integral
        kappa_new = current_kappa + delta_kappa
        kappa_new = max(self.cfg.kappa_min, min(self.cfg.kappa_max, kappa_new))

        self._last_error = error
        self._p_h1_history.append(p_h1)
        self._kappa_history.append(kappa_new)
        self._error_history.append(error)
        if len(self._p_h1_history) > self.cfg.history_len:
            self._p_h1_history = self._p_h1_history[-self.cfg.history_len :]
            self._kappa_history = self._kappa_history[-self.cfg.history_len :]
            self._error_history = self._error_history[-self.cfg.history_len :]

        return kappa_new

    def is_gate_open(self, p_h1: float | None = None) -> bool:
        """Check if p_h1 > τ_h1 (consciousness gate open)."""
        val = (
            p_h1
            if p_h1 is not None
            else (self._p_h1_history[-1] if self._p_h1_history else 0.0)
        )
        return val > self.cfg.tau_h1

    def get_state(self) -> dict:
        """Return serialisable controller state."""
        return {
            "last_error": round(self._last_error, 6),
            "integral": round(self._integral, 6),
            "tau_h1": self.cfg.tau_h1,
            "gate_open": self.is_gate_open(),
            "p_h1_mean": round(
                float(np.mean(self._p_h1_history)) if self._p_h1_history else 0.0,
                6,
            ),
            "kappa_mean": round(
                float(np.mean(self._kappa_history)) if self._kappa_history else 0.0,
                6,
            ),
        }

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._p_h1_history.clear()
        self._kappa_history.clear()
        self._error_history.clear()
