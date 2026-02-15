# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Consciousness Gate Verification Benchmarks
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Consciousness-gate verification benchmarks.

Three mandatory synthetic validations for the TCBO + PGBO system:

  1. **Baseline→kappa increase → p_h1 up**: Increasing gap-junction coupling
     should raise the topological consciousness observable.
  2. **Anesthesia (flat signal) → p_h1 down**: Destroying phase structure
     should collapse the observable.
  3. **PI controller recovery**: After anesthesia, the PI controller should
     restore kappa to recover p_h1 above threshold.

These benchmarks use synthetic oscillator data (no real EEG required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .tcbo import TCBOObserver, TCBOConfig, TCBOController, TCBOControllerConfig
from .pgbo import PGBOEngine, PGBOConfig


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    passed: bool
    p_h1_initial: float
    p_h1_final: float
    detail: str


def _generate_coherent_phases(
    N: int, T: int, freq: float = 1.0, coupling: float = 0.3, dt: float = 0.01,
) -> np.ndarray:
    """Generate synthetic coupled oscillator phases.

    Returns ndarray (T, N) of phase values.
    """
    theta = np.random.uniform(0, 2 * np.pi, N)
    phases = np.zeros((T, N))
    omega = np.random.uniform(0.8 * freq, 1.2 * freq, N)

    for t in range(T):
        phases[t] = theta.copy()
        # Kuramoto step
        for n in range(N):
            coupling_term = coupling * np.sum(np.sin(theta - theta[n])) / N
            theta[n] += (omega[n] + coupling_term) * dt + 0.01 * np.random.randn()

    return phases


def _generate_flat_signal(N: int, T: int) -> np.ndarray:
    """Generate flat (anesthesia-like) signal: near-constant with tiny noise."""
    base = np.random.uniform(0, 2 * np.pi, N)
    return np.tile(base, (T, 1)) + 0.001 * np.random.randn(T, N)


def benchmark_kappa_increase(
    N: int = 8,
    T: int = 80,
) -> BenchmarkResult:
    """Benchmark 1: Increasing coupling → p_h1 rises.

    Runs TCBO with low coupling, then high coupling, and checks that
    p_h1 increases.
    """
    cfg = TCBOConfig(window_size=30, embed_dim=3, tau_delay=1, compute_every_n=1)

    # Low coupling
    observer_low = TCBOObserver(N=N, config=cfg)
    phases_low = _generate_coherent_phases(N, T, coupling=0.05)
    for t in range(T):
        observer_low.push_and_compute(phases_low[t], force=True)
    p_h1_low = observer_low.p_h1

    # High coupling
    observer_high = TCBOObserver(N=N, config=cfg)
    phases_high = _generate_coherent_phases(N, T, coupling=0.8)
    for t in range(T):
        observer_high.push_and_compute(phases_high[t], force=True)
    p_h1_high = observer_high.p_h1

    passed = bool(p_h1_high > p_h1_low)
    return BenchmarkResult(
        name="kappa_increase",
        passed=passed,
        p_h1_initial=p_h1_low,
        p_h1_final=p_h1_high,
        detail=f"Low coupling p_h1={p_h1_low:.4f}, High coupling p_h1={p_h1_high:.4f}",
    )


def benchmark_anesthesia(
    N: int = 8,
    T: int = 80,
) -> BenchmarkResult:
    """Benchmark 2: Flat signal (anesthesia) → p_h1 drops.

    Runs TCBO with coherent signal first, then flat signal, and checks
    that p_h1 decreases.
    """
    cfg = TCBOConfig(window_size=30, embed_dim=3, tau_delay=1, compute_every_n=1)

    # Coherent phase
    observer_coherent = TCBOObserver(N=N, config=cfg)
    phases_coherent = _generate_coherent_phases(N, T, coupling=0.5)
    for t in range(T):
        observer_coherent.push_and_compute(phases_coherent[t], force=True)
    p_h1_coherent = observer_coherent.p_h1

    # Flat (anesthesia)
    observer_flat = TCBOObserver(N=N, config=cfg)
    phases_flat = _generate_flat_signal(N, T)
    for t in range(T):
        observer_flat.push_and_compute(phases_flat[t], force=True)
    p_h1_flat = observer_flat.p_h1

    passed = bool(p_h1_coherent > p_h1_flat)
    return BenchmarkResult(
        name="anesthesia",
        passed=passed,
        p_h1_initial=p_h1_coherent,
        p_h1_final=p_h1_flat,
        detail=f"Coherent p_h1={p_h1_coherent:.4f}, Flat p_h1={p_h1_flat:.4f}",
    )


def benchmark_pi_recovery(
    N: int = 8,
    n_anesthesia_steps: int = 30,
    n_recovery_steps: int = 50,
) -> BenchmarkResult:
    """Benchmark 3: PI controller restores p_h1 after anesthesia.

    1. Run coherent signal → record p_h1
    2. Inject flat signal (anesthesia) → p_h1 drops
    3. PI controller adjusts kappa → p_h1 recovers
    """
    cfg = TCBOConfig(window_size=20, embed_dim=3, tau_delay=1, compute_every_n=1)
    ctrl_cfg = TCBOControllerConfig(Kp=2.0, Ki=0.5, kappa_max=5.0)

    observer = TCBOObserver(N=N, config=cfg)
    controller = TCBOController(config=ctrl_cfg)
    kappa = 0.3
    dt = 0.01

    # Phase 1: Build up coherent state
    phases_coherent = _generate_coherent_phases(N, 40, coupling=0.5)
    for t in range(40):
        observer.push_and_compute(phases_coherent[t], force=True)
    p_h1_baseline = observer.p_h1

    # Phase 2: Anesthesia (flat signal)
    phases_flat = _generate_flat_signal(N, n_anesthesia_steps)
    for t in range(n_anesthesia_steps):
        observer.push_and_compute(phases_flat[t], force=True)
    p_h1_anesthesia = observer.p_h1

    # Phase 3: Recovery with PI controller adjusting kappa
    omega = np.random.uniform(0.8, 1.2, N)
    theta = np.random.uniform(0, 2 * np.pi, N)
    for _ in range(n_recovery_steps):
        # PI controller adjusts kappa
        kappa = controller.step(observer.p_h1, kappa, dt)
        # Generate signal with adaptive coupling
        for n in range(N):
            coupling_term = kappa * np.sum(np.sin(theta - theta[n])) / N
            theta[n] += (omega[n] + coupling_term) * dt + 0.01 * np.random.randn()
        observer.push_and_compute(theta.copy(), force=True)
    p_h1_recovered = observer.p_h1

    passed = bool(p_h1_recovered > p_h1_anesthesia)
    return BenchmarkResult(
        name="pi_recovery",
        passed=passed,
        p_h1_initial=p_h1_anesthesia,
        p_h1_final=p_h1_recovered,
        detail=(
            f"Baseline={p_h1_baseline:.4f}, "
            f"Anesthesia={p_h1_anesthesia:.4f}, "
            f"Recovered={p_h1_recovered:.4f}"
        ),
    )


def benchmark_pgbo_properties(N: int = 16) -> BenchmarkResult:
    """Benchmark 4: PGBO tensor properties (symmetry, PSD, saturation).

    Verifies that h_munu is:
      - Symmetric: ||h - h^T|| < ε
      - Positive semi-definite: all eigenvalues ≥ 0
      - Bounded under saturation cap
    """
    engine = PGBOEngine(N=N, config=PGBOConfig(kappa=0.3, u_cap=10.0))

    # Two steps to get a finite-difference gradient
    theta1 = np.random.uniform(0, 2 * np.pi, N)
    theta2 = theta1 + 0.1 * np.random.randn(N)

    engine.compute(theta1, dt=0.01)
    u_mu, h_munu = engine.compute(theta2, dt=0.01)

    # Symmetry
    sym_err = float(np.max(np.abs(h_munu - h_munu.T)))
    sym_ok = sym_err < 1e-12

    # PSD
    eigvals = np.linalg.eigvalsh(h_munu)
    psd_ok = bool(np.all(eigvals >= -1e-12))

    # Saturation
    u_norm = float(np.linalg.norm(u_mu))
    sat_ok = u_norm < engine.cfg.u_cap

    passed = bool(sym_ok and psd_ok and sat_ok)
    return BenchmarkResult(
        name="pgbo_properties",
        passed=passed,
        p_h1_initial=0.0,
        p_h1_final=0.0,
        detail=(
            f"Symmetric={sym_ok} (err={sym_err:.2e}), "
            f"PSD={psd_ok} (min_eig={eigvals[0]:.6f}), "
            f"Saturated={sat_ok} (|u|={u_norm:.4f})"
        ),
    )


def run_all_benchmarks(N: int = 8) -> List[BenchmarkResult]:
    """Run all consciousness-gate verification benchmarks.

    Returns list of BenchmarkResult. All should have passed=True.
    """
    results = [
        benchmark_kappa_increase(N=N),
        benchmark_anesthesia(N=N),
        benchmark_pi_recovery(N=N),
        benchmark_pgbo_properties(N=max(N, 8)),
    ]
    return results
