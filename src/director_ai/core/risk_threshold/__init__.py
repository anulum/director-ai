# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Risk-adaptive threshold package

"""Risk-adaptive thresholding.

Compute a per-request approval threshold from a documented risk profile — user
role, tenant risk, domain, retrieval confidence, action reversibility, external
exposure, PII presence, evidence freshness, and historical false-halt rate —
deterministically, with every factor's contribution recorded.
"""

from __future__ import annotations

from .adapter import RiskAdaptiveThreshold, RiskThresholdDecision
from .factors import RiskFactors
from .policy import RiskThresholdPolicy

__all__ = [
    "RiskAdaptiveThreshold",
    "RiskFactors",
    "RiskThresholdDecision",
    "RiskThresholdPolicy",
]
