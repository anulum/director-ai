# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-model consensus package

"""Cross-model consensus and divergence explanation.

See :mod:`director_ai.core.consensus.cross_model_consensus`.
"""

from director_ai.core.consensus.cross_model_consensus import (
    ConsensusResult,
    ContradictionEngine,
    CrossModelConsensus,
    Divergence,
    ModelResponse,
)

__all__ = [
    "ConsensusResult",
    "ContradictionEngine",
    "CrossModelConsensus",
    "Divergence",
    "ModelResponse",
]
