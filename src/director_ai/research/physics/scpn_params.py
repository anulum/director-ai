# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — SCPN Canonical Parameters
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Canonical Omega_n (natural frequencies) and Knm (coupling matrix)
for the 16-layer SCPN hierarchy.

Data sources:
  - Omega_n: parameter_catalogue_full.yaml lines 542-925
    (validated, source: numerical_fit, last_updated: 2025-12-25)
  - Knm: HolonomicAtlas/src/knm_tools/knm_matrix_calculator.py
    (K_base=0.45, decay alpha=0.3, 4 calibration anchors + cross boosts)
"""

from __future__ import annotations

import numpy as np

N_LAYERS = 16

# ============================================================================
# Natural frequencies Omega_n (rad/s) for the 16 SCPN layers
# ============================================================================
#
# Each frequency characterises the dominant timescale of the layer:
#   L1  Quantum Biological    ~ THz substrate down-converted
#   L2  Neurochemical         ~ gamma peak (≈40 Hz → 251.3 rad/s)
#   L3  Genomic-Epigenetic    ~ gene regulation timescale
#   L4  Cellular Synchrony    ~ gap-junction coupling
#   L5  Intentional Frame     ~ active-inference update
#   L6  Gaian Embodiment      ~ Schumann resonance (≈7.83 Hz)
#   L7  Geometrical-Symbolic  ~ symbolic processing
#   L8  Cosmic Information    ~ helio/geomagnetic
#   L9  Memory Manifold       ~ engram consolidation
#   L10 Identity Manifold     ~ self-model
#   L11 Noospheric-Cultural   ~ collective dynamics
#   L12 Ecological-Gaian      ~ planetary processes
#   L13 Source-Field           ~ Planck-scale SOC
#   L14 Transdimensional      ~ KK mode coupling
#   L15 Consilium             ~ oversoul integration
#   L16 The Director          ~ cybernetic closure

OMEGA_N = np.array(
    [
        1.329,  # L1  - Quantum Biological
        251.327,  # L2  - Neurochemical (≈40 Hz × 2π)
        0.628,  # L3  - Genomic-Epigenetic
        31.416,  # L4  - Cellular Synchrony (≈5 Hz × 2π)
        6.283,  # L5  - Intentional Frame (≈1 Hz × 2π)
        49.199,  # L6  - Gaian / Schumann (≈7.83 Hz × 2π)
        3.142,  # L7  - Geometrical-Symbolic
        0.105,  # L8  - Cosmic Information
        1.571,  # L9  - Memory Manifold
        0.942,  # L10 - Identity Manifold
        0.209,  # L11 - Noospheric-Cultural
        0.042,  # L12 - Ecological-Gaian
        0.013,  # L13 - Source-Field
        0.006,  # L14 - Transdimensional
        0.003,  # L15 - Consilium
        0.991,  # L16 - The Director
    ],
    dtype=np.float64,
)

LAYER_NAMES = [
    "Quantum Biological",
    "Neurochemical",
    "Genomic-Epigenetic",
    "Cellular Synchrony",
    "Intentional Frame",
    "Gaian Embodiment",
    "Geometrical-Symbolic",
    "Cosmic Information",
    "Memory Manifold",
    "Identity Manifold",
    "Noospheric-Cultural",
    "Ecological-Gaian",
    "Source-Field",
    "Transdimensional",
    "Consilium",
    "The Director",
]


def load_omega_n() -> np.ndarray:
    """Return the canonical 16-layer natural frequency vector (rad/s)."""
    result: np.ndarray = OMEGA_N.copy()
    return result


# ============================================================================
# Knm coupling matrix builder
# ============================================================================
#
# Construction (from HolonomicAtlas knm_matrix_calculator.py):
#   1. Exponential-decay base:  K_nm = K_base * exp(-alpha * |n - m|)
#   2. Overwrite with 4 calibration anchors
#   3. Apply cross-hierarchy boosts
#   4. Symmetrise: K_nm = K_mn
#   5. Zero diagonal (no self-coupling)

K_BASE = 0.45
DECAY_ALPHA = 0.3

# Calibration anchors (1-indexed layer pairs → coupling value)
CALIBRATION_ANCHORS = {
    (1, 2): 0.302,
    (2, 3): 0.201,
    (3, 4): 0.252,
    (4, 5): 0.154,
}

# Cross-hierarchy boosts (1-indexed)
CROSS_BOOSTS = {
    (1, 16): 0.05,  # L1 ↔ L16  quantum-director bridge
    (5, 7): 0.15,  # L5 ↔ L7   intentional-symbolic bridge
}


def build_knm_matrix(n_layers: int = 16) -> np.ndarray:
    """Build the 16×16 Knm inter-layer coupling matrix.

    Construction:
      1. Exponential decay baseline  K_nm = K_base * exp(-alpha * |n-m|)
      2. Overwrite calibration anchors
      3. Apply cross-hierarchy boosts
      4. Symmetrise and zero diagonal

    Returns:
        ndarray of shape (n_layers, n_layers).
    """
    K = np.zeros((n_layers, n_layers), dtype=np.float64)

    # Step 1: exponential-decay baseline
    for n in range(n_layers):
        for m in range(n_layers):
            if n != m:
                K[n, m] = K_BASE * np.exp(-DECAY_ALPHA * abs(n - m))

    # Step 2: overwrite calibration anchors (1-indexed → 0-indexed)
    for (i, j), val in CALIBRATION_ANCHORS.items():
        K[i - 1, j - 1] = val
        K[j - 1, i - 1] = val

    # Step 3: apply cross-hierarchy boosts
    for (i, j), val in CROSS_BOOSTS.items():
        K[i - 1, j - 1] = val
        K[j - 1, i - 1] = val

    # Step 4: ensure symmetry and zero diagonal
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)

    result: np.ndarray = K
    return result
