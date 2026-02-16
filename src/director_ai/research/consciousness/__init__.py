# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Consciousness Gate (Track 2)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Consciousness gate and topological observables.

Modules:
    - ``tcbo``      — Topological Consciousness Boundary Observable + controller
    - ``pgbo``      — Phase→Geometry Bridge Operator
    - ``benchmark`` — Consciousness-gate verification benchmarks
"""

from .benchmark import (
    BenchmarkResult,
    benchmark_anesthesia,
    benchmark_kappa_increase,
    benchmark_pgbo_properties,
    benchmark_pi_recovery,
    run_all_benchmarks,
)
from .pgbo import (
    PGBOConfig,
    PGBOEngine,
    phase_geometry_bridge,
)
from .tcbo import (
    TCBOConfig,
    TCBOController,
    TCBOControllerConfig,
    TCBOObserver,
    delay_embed,
    delay_embed_multi,
    persistence_to_probability,
    s0_for_threshold,
)

__all__ = [
    # TCBO
    "TCBOConfig",
    "TCBOObserver",
    "TCBOControllerConfig",
    "TCBOController",
    "delay_embed",
    "delay_embed_multi",
    "persistence_to_probability",
    "s0_for_threshold",
    # PGBO
    "PGBOConfig",
    "PGBOEngine",
    "phase_geometry_bridge",
    # Benchmarks
    "BenchmarkResult",
    "run_all_benchmarks",
    "benchmark_kappa_increase",
    "benchmark_anesthesia",
    "benchmark_pi_recovery",
    "benchmark_pgbo_properties",
]
