# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Sealed Look Admission

"""Admission of a sealed look against custody records and the journal.

:func:`admit_sealed_look` runs every check before a split is opened, so a
refusal consumes nothing. It binds the request to its preregistration and
approval by record digest, requires the pool's splits to come from the
programme's custodian role and the approval from its threshold approver,
orders the preregistration, the approval and the candidates by custody
sequence, checks that custody witnesses every journal entry and opening,
derives the lineage from custody and checks the look budget. Build the
admitted look's opening with :func:`~.look_openings.look_opening_for`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from ._checks import require_int, require_text
from .custody_records import (
    MODEL_CUSTODY_KINDS,
    CustodyIngestRecord,
    validate_custody_ingest_record,
)
from .digests import record_digest
from .errors import (
    CustodyRecordError,
    EvaluationContractError,
    PreregistrationOrderError,
    SealedPoolCustodyError,
    UnverifiedCustodyArtefactError,
)
from .evaluation_contracts import PreregistrationRecord, validate_preregistration
from .journal_custody import check_journal_custody
from .journal_entries import check_journal_order, select_study_entries
from .lineage import LineageComponent, derive_proposer_lineage, exempt_base_digests
from .look_budget import check_look_budget
from .preregistration_approvals import (
    PreregistrationApproval,
    validate_preregistration_approval,
)
from .sealed_looks import (
    SealedLookJournalEntry,
    SealedLookRequest,
    validate_look_request,
)


@dataclass(frozen=True, slots=True)
class CustodyProgrammeRoles:
    """The roles a programme's root configuration names.

    Attributes
    ----------
    programme : str
        Programme name.
    custodian_role : str
        The only role that may ingest the programme's sealed splits.
    threshold_approver_role : str
        The only role whose preregistration approvals count.
    deployment_operator_role : str
        The only role whose tier pins count for a promotion.
    """

    programme: str
    custodian_role: str
    threshold_approver_role: str
    deployment_operator_role: str


def _record(
    records: Mapping[str, CustodyIngestRecord],
    digest: str,
    kinds: Collection[str],
    what: str,
) -> CustodyIngestRecord:
    """Return the custody record of ``digest``, which must have one of ``kinds``."""
    record = records.get(digest)
    if record is None:
        raise UnverifiedCustodyArtefactError(f"{what} has no custody ingest record")
    if record.kind not in kinds:
        raise EvaluationContractError(
            f"{what} is custody kind {record.kind!r}, not {sorted(kinds)}"
        )
    return record


def _validated_records(
    records: Mapping[str, CustodyIngestRecord],
) -> Mapping[str, CustodyIngestRecord]:
    """Validate every record and its key, returning the mapping."""
    for digest, record in records.items():
        validate_custody_ingest_record(record)
        if record.content_digest != digest:
            raise CustodyRecordError(
                "custody records must be keyed by their content_digest"
            )
    return records


def admit_sealed_look(
    request: SealedLookRequest,
    *,
    preregistration: PreregistrationRecord,
    approval: PreregistrationApproval,
    custody_records: Mapping[str, CustodyIngestRecord],
    roles: CustodyProgrammeRoles,
    prior_journal: Sequence[SealedLookJournalEntry],
    look_custody_seq: int,
) -> LineageComponent:
    """Admit a sealed look or refuse it before any split is opened.

    Parameters
    ----------
    request : SealedLookRequest
        The look request.
    preregistration : PreregistrationRecord
        The preregistration whose record digest the request names.
    approval : PreregistrationApproval
        The threshold approver's approval of that preregistration.
    custody_records : Mapping[str, CustodyIngestRecord]
        Every custody record, keyed by content digest.
    roles : CustodyProgrammeRoles
        The request programme's configured roles.
    prior_journal : Sequence[SealedLookJournalEntry]
        The programme's whole journal before this look, in order. Custody
        must hold each entry and each opening (:mod:`.journal_custody`); the
        trusts left are that ``custody_records`` is complete and that custody
        ingests an opening before it releases the look's split.
    look_custody_seq : int
        The custody sequence watermark at the time of the look.

    Returns
    -------
    LineageComponent
        The custody-derived lineage of the look.

    Raises
    ------
    ModelLifecycleError
        A named subclass for every refusal: malformed records, missing
        custody, pool or approval roles, ordering, or the look budget.
    """
    validate_look_request(request)
    validate_preregistration(preregistration)
    validate_preregistration_approval(approval)
    records = _validated_records(custody_records)
    check_journal_order(prior_journal, programme=request.programme)
    require_text(roles.custodian_role, "custodian_role", SealedPoolCustodyError)
    require_text(
        roles.threshold_approver_role, "threshold_approver_role", SealedPoolCustodyError
    )
    if roles.programme != request.programme:
        raise SealedPoolCustodyError("the configured roles belong to another programme")
    watermark = require_int(
        look_custody_seq, "look_custody_seq", PreregistrationOrderError, minimum=0
    )

    if record_digest(preregistration) != request.preregistration_digest:
        raise EvaluationContractError(
            "the preregistration does not match the request's digest"
        )
    prereg = _record(
        records,
        request.preregistration_digest,
        {"preregistration"},
        "the preregistration",
    )
    pool = preregistration.split_pool
    if request.split_digest not in pool.split_digests:
        raise EvaluationContractError(
            "the requested split is not in the preregistered pool"
        )
    for split in pool.split_digests:
        split_record = _record(records, split, {"split"}, "a pool split")
        if (
            split_record.programme != request.programme
            or split_record.submitting_role != roles.custodian_role
        ):
            raise SealedPoolCustodyError(
                "every pool split must be ingested by the programme's custodian role"
            )
    if set(request.frozen_artefact_digests) != set(
        preregistration.frozen_look_candidate_digests
    ):
        raise EvaluationContractError(
            "the look must evaluate exactly the preregistered frozen artefacts"
        )
    for digest in request.frozen_artefact_digests:
        _record(records, digest, MODEL_CUSTODY_KINDS, "a frozen artefact")

    approval_record = _record(
        records, record_digest(approval), {"preregistration_approval"}, "the approval"
    )
    if approval_record.submitting_role != roles.threshold_approver_role:
        raise SealedPoolCustodyError(
            "the approval must be ingested by the threshold approver role"
        )
    if (
        approval.preregistration_digest != request.preregistration_digest
        or approval.pool_id != pool.pool_id
    ):
        raise SealedPoolCustodyError(
            "the approval names another preregistration or pool"
        )
    for record in (prereg, approval_record):
        if record.programme != request.programme:
            raise SealedPoolCustodyError(
                "the preregistration and approval must belong to the programme"
            )

    models = {
        digest: record
        for digest, record in records.items()
        if record.kind in MODEL_CUSTODY_KINDS
    }
    exempt = exempt_base_digests(
        preregistration.base_checkpoint_digests,
        preregistration_seq=prereg.custody_ingest_seq,
        split_digest=request.split_digest,
        records=models,
        edges=request.declared_lineage_edges,
        prior_journal=prior_journal,
    )
    candidates = [d for d in request.frozen_artefact_digests if d not in exempt]
    if candidates and not any(
        records[d].submitting_role == request.requester_role for d in candidates
    ):
        raise EvaluationContractError(
            "the requester role ingested none of the look's candidates"
        )
    if approval_record.custody_ingest_seq > watermark:
        raise PreregistrationOrderError(
            "the approval was ingested after the look's watermark"
        )
    for digest in request.frozen_artefact_digests:
        seq = records[digest].custody_ingest_seq
        if seq > watermark:
            raise PreregistrationOrderError(
                "a frozen artefact was ingested after the look's watermark"
            )
        if digest in exempt:
            continue
        if seq <= prereg.custody_ingest_seq:
            raise PreregistrationOrderError(
                "a candidate was ingested before its preregistration"
            )
        if seq <= approval_record.custody_ingest_seq:
            raise PreregistrationOrderError(
                "a candidate was ingested before the approval"
            )
    check_journal_custody(
        prior_journal,
        programme=request.programme,
        custody_records=records,
        look_custody_seq=watermark,
        study_id=preregistration.study_id,
        preregistration_digest=request.preregistration_digest,
    )
    study_entries = select_study_entries(
        prior_journal,
        study_id=preregistration.study_id,
        programme=request.programme,
        custody_records=records,
    )
    if any(
        prereg.custody_ingest_seq > entry.custody_seq_at_open for entry in study_entries
    ):
        raise PreregistrationOrderError(
            "the preregistration was amended after a look of its study opened"
        )

    component = derive_proposer_lineage(
        request.frozen_artefact_digests,
        exempt_bases=exempt,
        records=models,
        edges=request.declared_lineage_edges,
        prior_journal=prior_journal,
    )
    check_look_budget(
        component,
        split_digest=request.split_digest,
        look_kind=request.look_kind,
        prior_journal=prior_journal,
    )
    return component
