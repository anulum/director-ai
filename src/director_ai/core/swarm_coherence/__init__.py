# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — swarm coherence package

"""Real-time cross-agent contradiction detection with cascade halt.

See :mod:`director_ai.core.swarm_coherence.cascade_monitor`.
"""

from director_ai.core.swarm_coherence.cascade_monitor import (
    AgentMessage,
    CascadeContradiction,
    CoherenceUpdate,
    ContradictionEngine,
    SwarmCoherenceMonitor,
)

__all__ = [
    "AgentMessage",
    "CascadeContradiction",
    "CoherenceUpdate",
    "ContradictionEngine",
    "SwarmCoherenceMonitor",
]
