# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence packet package

"""One-command, tamper-evident evidence packet for the narrow grounding demo.

Runs the whole loop — load policy facts, approve a grounded answer, block a
hallucinated one, attach the Answer BOM and eval-trace evidence — and emits a
SHA-256-sealed packet a reviewer can verify without re-running the guard.
"""

from __future__ import annotations

from .packet import (
    DEMO_FACTS,
    EVIDENCE_PACKET_VERSION,
    build_evidence_packet,
    verify_evidence_packet,
)

__all__ = [
    "DEMO_FACTS",
    "EVIDENCE_PACKET_VERSION",
    "build_evidence_packet",
    "verify_evidence_packet",
]
