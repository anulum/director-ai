# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — execution rings (graduated authorisation)

"""Graduated human authorisation for agent actions (execution rings).

Each action an agent attempts is classified into an ordered :class:`ExecutionRing`
— read, write, delete, execute, exfiltrate — and :class:`ExecutionRingGate`
allows it only when the human authorisation factors that ring demands have been
collected. The model bounds the blast radius of a prompt-injected agent: even a
fully bypassed guardrail cannot delete or exfiltrate without out-of-band human
confirmation (approval, cooling period, second operator, CISO notification).
"""

from .authorization import (
    AuthorizationEvidence,
    AuthorizationFactor,
    satisfied_factors,
)
from .gate import ExecutionRingGate, RingDecision
from .rings import RING_REQUIRED_FACTORS, ExecutionRing, classify_operation

__all__ = [
    "RING_REQUIRED_FACTORS",
    "AuthorizationEvidence",
    "AuthorizationFactor",
    "ExecutionRing",
    "ExecutionRingGate",
    "RingDecision",
    "classify_operation",
    "satisfied_factors",
]
