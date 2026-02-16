# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — SCPN Layer 16 Physics (Track 1)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
SCPN Layer 16 mechanistic physics extensions.

Modules:
    - ``scpn_params``       — Canonical Omega_n / Knm for 16-layer hierarchy
    - ``sec_functional``    — SEC as a Lyapunov functional with stability proof
    - ``l16_mechanistic``   — Lyapunov-based oversight dynamics (UPDE integrator)
    - ``l16_closure``       — L16 cybernetic closure (PI controllers, refusal rules)
"""

from .gpu_upde import TorchUPDEConfig, TorchUPDEStepper
from .l16_closure import (
    L16Controller,
    L16ControllerState,
    PIState,
    pi_step,
)
from .l16_mechanistic import (
    L16OversightLoop,
    OversightSnapshot,
    UPDEState,
    UPDEStepper,
)
from .lyapunov_proof import ProofResult, run_all_proofs
from .scpn_params import (
    CALIBRATION_ANCHORS,
    CROSS_BOOSTS,
    DECAY_ALPHA,
    K_BASE,
    LAYER_NAMES,
    N_LAYERS,
    OMEGA_N,
    build_knm_matrix,
    load_omega_n,
)
from .sec_functional import SECFunctional, SECResult
from .ssgf_cycle import SSGFConfig, SSGFEngine, SSGFState

__all__ = [
    # Parameters
    "N_LAYERS",
    "OMEGA_N",
    "LAYER_NAMES",
    "K_BASE",
    "DECAY_ALPHA",
    "CALIBRATION_ANCHORS",
    "CROSS_BOOSTS",
    "load_omega_n",
    "build_knm_matrix",
    # SEC Functional
    "SECFunctional",
    "SECResult",
    # L16 Mechanistic
    "UPDEState",
    "UPDEStepper",
    "OversightSnapshot",
    "L16OversightLoop",
    # L16 Closure
    "PIState",
    "pi_step",
    "L16ControllerState",
    "L16Controller",
    # SSGF
    "SSGFConfig",
    "SSGFEngine",
    "SSGFState",
    # Lyapunov proof
    "ProofResult",
    "run_all_proofs",
    # GPU UPDE
    "TorchUPDEStepper",
    "TorchUPDEConfig",
]
