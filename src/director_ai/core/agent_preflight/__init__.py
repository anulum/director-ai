# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Agent preflight package

"""Agent / MCP preflight guard.

Five gates for the seams of an agent loop — before a tool call, after a tool
result, before the final answer, before a cross-agent handoff, and before an
irreversible action — each returning a tenant-safe, evidence- and policy-tied
:class:`PreflightDecision`.
"""

from __future__ import annotations

from .decision import DECISIONS, PreflightDecision
from .guard import AgentPreflightGuard
from .policy import PreflightPolicy

__all__ = [
    "DECISIONS",
    "AgentPreflightGuard",
    "PreflightDecision",
    "PreflightPolicy",
]
