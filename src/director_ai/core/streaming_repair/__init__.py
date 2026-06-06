# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming repair package

"""Streaming repair: rewrite an unsupported clause instead of only halting.

Turns a coherence halt into a corrective pass — find the unsupported clause,
retrieve corrective evidence, rewrite only that clause (or redact it), and
resume with the rest of the answer intact, emitting a tenant-safe repair event
per fix.
"""

from __future__ import annotations

from .clauses import join_clauses, split_clauses
from .repair import RepairAction, RepairResult, StreamingRepairer

__all__ = [
    "RepairAction",
    "RepairResult",
    "StreamingRepairer",
    "join_clauses",
    "split_clauses",
]
