# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Canary package

"""Counterfactual canary facts.

Plant tenant-scoped, uniquely-marked false facts in a knowledge base and detect
when one surfaces in a model's output (leakage) or in the retrieved evidence
(citation) — a direct signal of leakage, injection, or KB poisoning.
"""

from __future__ import annotations

from .detector import CanaryDetector, CanarySignal
from .registry import CANARY_FLAG, CANARY_ID_KEY, CanaryFact, CanaryRegistry

__all__ = [
    "CANARY_FLAG",
    "CANARY_ID_KEY",
    "CanaryDetector",
    "CanaryFact",
    "CanaryRegistry",
    "CanarySignal",
]
