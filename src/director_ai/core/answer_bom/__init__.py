# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Answer Bill of Materials package

"""Answer Bill of Materials.

A versioned, machine-readable manifest of one guarded answer: model, scorer,
threshold, and a per-claim record of evidence, support strength, and verdict.
Built from the scorer's existing claim-level provenance and round-trips through
an audit log.
"""

from __future__ import annotations

from .builder import build_answer_bom
from .manifest import (
    ANSWER_BOM_SCHEMA_VERSION,
    CLAIM_VERDICTS,
    AnswerBOM,
    ClaimRecord,
    new_answer_id,
    utc_timestamp,
)

__all__ = [
    "ANSWER_BOM_SCHEMA_VERSION",
    "CLAIM_VERDICTS",
    "AnswerBOM",
    "ClaimRecord",
    "build_answer_bom",
    "new_answer_id",
    "utc_timestamp",
]
