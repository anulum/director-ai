# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Preregistration Approvals

"""The threshold approver's approval of a preregistration and its pool.

Custody ingests the approval as its own ``preregistration_approval`` blob,
whose bytes are the canonical JSON of :class:`PreregistrationApproval`.
Only the programme's threshold approver role may ingest it, and its custody
sequence must precede every candidate's ingest and the look.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import require_choice, require_sha256, require_text, require_utc_timestamp
from .errors import CustodyRecordError


@dataclass(frozen=True, slots=True)
class PreregistrationApproval:
    """The threshold approver's approval of a preregistration and its pool.

    Attributes
    ----------
    schema : Literal["preregistration-approval.v1"]
        Record schema.
    preregistration_digest : str
        Record digest of the approved preregistration.
    pool_id : str
        The approved sealed split pool.
    approved_at_utc : str
        Informational; ordering uses the approval's custody sequence.
    """

    schema: Literal["preregistration-approval.v1"]
    preregistration_digest: str
    pool_id: str
    approved_at_utc: str


def validate_preregistration_approval(
    approval: PreregistrationApproval,
) -> PreregistrationApproval:
    """Check a preregistration approval's fields.

    Parameters
    ----------
    approval : PreregistrationApproval
        The approval to check.

    Returns
    -------
    PreregistrationApproval
        The same approval.

    Raises
    ------
    CustodyRecordError
        If a field is malformed.
    MalformedDigestError
        If the preregistration digest is malformed.
    """
    if not isinstance(approval, PreregistrationApproval):
        raise CustodyRecordError("a PreregistrationApproval is required")
    require_choice(
        approval.schema, "schema", {"preregistration-approval.v1"}, CustodyRecordError
    )
    require_sha256(approval.preregistration_digest, "preregistration_digest")
    require_text(approval.pool_id, "pool_id", CustodyRecordError)
    require_utc_timestamp(
        approval.approved_at_utc, "approved_at_utc", CustodyRecordError
    )
    return approval
