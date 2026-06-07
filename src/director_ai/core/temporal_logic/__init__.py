# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Temporal Logic Runtime Monitoring
"""Linear Temporal Logic runtime monitoring for agent trajectories.

Public surface:

* formula algebra and constructors (``atom``, ``G``, ``F``, ``X``, ``U``,
  ``and_``, ``or_``, ``not_``, ``implies``);
* the three-valued :class:`LTLMonitor` and its :class:`Verdict`;
* the agent-domain :class:`TrajectorySafetyMonitor`, :class:`StepObservation`,
  and the built-in :func:`default_agent_safety_specs`.
"""

from __future__ import annotations

from .agent_specs import (
    COHERENCE_CHECK,
    EVIDENCE_RETRIEVED,
    FACT_CLAIM,
    HANDOFF,
    INJECTION_DETECTED,
    OUTPUT_EMITTED,
    TOOL_CALL,
    VERIFICATION_PASSED,
    SpecStatus,
    StepObservation,
    TrajectorySafetyMonitor,
    default_agent_safety_specs,
)
from .formula import (
    BOTTOM,
    TOP,
    Atom,
    F,
    Formula,
    G,
    U,
    X,
    and_,
    atom,
    implies,
    not_,
    or_,
    progress,
    value_at_end,
)
from .monitor import LTLMonitor, Verdict

__all__ = [
    "BOTTOM",
    "COHERENCE_CHECK",
    "EVIDENCE_RETRIEVED",
    "FACT_CLAIM",
    "HANDOFF",
    "INJECTION_DETECTED",
    "OUTPUT_EMITTED",
    "TOOL_CALL",
    "TOP",
    "VERIFICATION_PASSED",
    "Atom",
    "F",
    "Formula",
    "G",
    "LTLMonitor",
    "SpecStatus",
    "StepObservation",
    "TrajectorySafetyMonitor",
    "U",
    "Verdict",
    "X",
    "and_",
    "atom",
    "default_agent_safety_specs",
    "implies",
    "not_",
    "or_",
    "progress",
    "value_at_end",
]
