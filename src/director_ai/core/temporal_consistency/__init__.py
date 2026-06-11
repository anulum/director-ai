# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Temporal Consistency Graph
"""Structured-claim temporal consistency tracking across sessions and documents."""

from __future__ import annotations

from .graph import (
    FUNCTIONAL_VALUE,
    POLARITY,
    TemporalClaim,
    TemporalConsistencyGraph,
    TemporalContradiction,
)

__all__ = [
    "FUNCTIONAL_VALUE",
    "POLARITY",
    "TemporalClaim",
    "TemporalConsistencyGraph",
    "TemporalContradiction",
]
