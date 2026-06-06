# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Active labelling cockpit package

"""Active-labelling cockpit.

Rank the most informative guard decisions to label, split labelled outcomes into
false halts vs missed hallucinations, sweep the threshold trade-off curve,
recommend a per-domain threshold, and export a deterministic train/eval packet.
"""

from __future__ import annotations

from .cockpit import (
    ActiveLabellingCockpit,
    ErrorBreakdown,
    ThresholdPoint,
    ThresholdRecommendation,
)
from .items import GROUNDED, HALLUCINATION, LABELS, LabelItem

__all__ = [
    "GROUNDED",
    "HALLUCINATION",
    "LABELS",
    "ActiveLabellingCockpit",
    "ErrorBreakdown",
    "LabelItem",
    "ThresholdPoint",
    "ThresholdRecommendation",
]
