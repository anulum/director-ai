# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — machine-learning bill of materials

"""Supply-chain provenance for the ML system (ML-BOM, OWASP ASVS).

Record every model, dataset, and dependency with a SHA-256 digest and provenance
at a known-good point, then re-verify the deployed artefacts to detect a swapped
or poisoned component. :class:`MachineLearningBOM` is the inventory (itself
tamper-evident via :attr:`~MachineLearningBOM.bom_digest`);
:class:`MLBOMComponent` is one pinned component, and
:class:`VerificationReport` is the intact / tampered / unverified result of a
re-verification pass.
"""

from .bom import MachineLearningBOM, VerificationReport
from .components import ComponentType, MLBOMComponent, compute_sha256

__all__ = [
    "ComponentType",
    "MLBOMComponent",
    "MachineLearningBOM",
    "VerificationReport",
    "compute_sha256",
]
