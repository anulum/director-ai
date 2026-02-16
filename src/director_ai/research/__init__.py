# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Research Extensions (SCPN Framework)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
SCPN Research Extensions for Director-Class AI.

These modules require the ``[research]`` optional dependencies::

    pip install director-ai[research]

Subpackages:
    - ``physics``       — SCPN Layer 16 mechanistic physics (Track 1)
    - ``consciousness`` — Consciousness gate / TCBO / PGBO (Track 2)
    - ``consilium``     — L15 Ethical Functional & active inference agent
"""

try:
    import scipy  # noqa: F401 — research extras gate
except ImportError as err:
    raise ImportError(
        "Research extensions require additional dependencies. "
        "Install them with:  pip install director-ai[research]"
    ) from err

# Track 2 — Consciousness Gate
from .consciousness import (
    BenchmarkResult,
    PGBOConfig,
    PGBOEngine,
    TCBOConfig,
    TCBOController,
    TCBOControllerConfig,
    TCBOObserver,
    phase_geometry_bridge,
    run_all_benchmarks,
)
from .consilium import ConsiliumAgent, EthicalFunctional, SystemState

# Track 1 — L16 Physics
from .physics import (
    LAYER_NAMES,
    N_LAYERS,
    OMEGA_N,
    L16Controller,
    L16OversightLoop,
    ProofResult,
    SECFunctional,
    SECResult,
    SSGFConfig,
    SSGFEngine,
    SSGFState,
    TorchUPDEConfig,
    TorchUPDEStepper,
    UPDEState,
    UPDEStepper,
    build_knm_matrix,
    load_omega_n,
    run_all_proofs,
)

__all__ = [
    # Consilium
    "ConsiliumAgent",
    "EthicalFunctional",
    "SystemState",
    # Physics (Track 1)
    "N_LAYERS",
    "OMEGA_N",
    "LAYER_NAMES",
    "load_omega_n",
    "build_knm_matrix",
    "SECFunctional",
    "SECResult",
    "UPDEState",
    "UPDEStepper",
    "L16OversightLoop",
    "L16Controller",
    # Consciousness (Track 2)
    "TCBOObserver",
    "TCBOConfig",
    "TCBOController",
    "TCBOControllerConfig",
    "PGBOEngine",
    "PGBOConfig",
    "phase_geometry_bridge",
    "run_all_benchmarks",
    "BenchmarkResult",
    # SSGF + Lyapunov
    "SSGFConfig",
    "SSGFEngine",
    "SSGFState",
    "ProofResult",
    "run_all_proofs",
    # GPU UPDE
    "TorchUPDEStepper",
    "TorchUPDEConfig",
]
