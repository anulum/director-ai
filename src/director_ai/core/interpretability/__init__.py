# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — mechanistic interpretability package

"""Mechanistic interpretability hooks for hallucination attribution.

:class:`MechanisticAttributor` implements the ReDeEP decoupling of parametric
Knowledge FFNs from external-context Copying Heads, so a hallucination signal
can be traced to the specific layers and attention heads that produced it. The
signals are injected, so the attribution logic runs without an ML stack; a real
deployment feeds in a transformer's MLP activations and attention maps.
"""

from .redeep import (
    ActivationProvider,
    HeadContribution,
    HeadSignal,
    LayerContribution,
    LayerSignal,
    MechanisticAttributionReport,
    MechanisticAttributor,
)

__all__ = [
    "ActivationProvider",
    "HeadContribution",
    "HeadSignal",
    "LayerContribution",
    "LayerSignal",
    "MechanisticAttributionReport",
    "MechanisticAttributor",
]
