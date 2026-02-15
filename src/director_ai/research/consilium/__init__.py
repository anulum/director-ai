# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Consilium (L15 Ethical Functional)
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact: www.anulum.li | protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Consilium subsystem — Ethical Functional optimizer and active inference agent.

Implements the 'Oversoul' ethical controller for the SCPN system,
optimizing system state based on Layer 15 (Teleology) dynamics.
"""

from .director_core import ConsiliumAgent, EthicalFunctional, SystemState

__all__ = [
    "ConsiliumAgent",
    "EthicalFunctional",
    "SystemState",
]
