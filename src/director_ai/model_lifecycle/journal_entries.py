# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Journal Entries

"""Building journal entries and reading a study's looks from the journal.

The kernel builds every journal entry from the look it admitted, so an
entry's study is the study of the preregistration it was admitted under.
Admission then selects a study's earlier looks from the whole journal by
``study_id`` instead of trusting a caller to select them.

The journal is per programme, starts at ``JOURNAL_ORIGIN_SEQ`` and its
``journal_seq`` rises by exactly one, so a missing or repeated entry
anywhere before the tail is refused, its head included. A truncated tail is
not detectable here: that the journal passed in is complete up to the look
stays a custody trust, as it is for the look budget.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .custody_records import CustodyIngestRecord
from .digests import record_digest
from .errors import EvaluationContractError, PreregistrationOrderError
from .evaluation_contracts import PreregistrationRecord, validate_preregistration
from .sealed_looks import (
    JOURNAL_ORIGIN_SEQ,
    SealedLookJournalEntry,
    SealedLookOutcomeStatus,
    SealedLookRequest,
    validate_journal_entry,
    validate_look_request,
)


def journal_entry_for(
    request: SealedLookRequest,
    preregistration: PreregistrationRecord,
    *,
    journal_seq: int,
    custody_seq_at_release: int,
    derived_proposer_lineage_id: str,
    status: SealedLookOutcomeStatus,
    fault_attestation_digest: str,
    recorded_at_utc: str,
) -> SealedLookJournalEntry:
    """Build the journal entry of an admitted look from its request and plan.

    Parameters
    ----------
    request : SealedLookRequest
        The admitted request.
    preregistration : PreregistrationRecord
        The preregistration the request names.
    journal_seq : int
        The entry's place in the programme's journal.
    custody_seq_at_release : int
        The custody watermark when the look finished.
    derived_proposer_lineage_id : str
        The lineage that admission derived.
    status : SealedLookOutcomeStatus
        The look's outcome.
    fault_attestation_digest : str
        The attestation of a fault outcome, else ``""``.
    recorded_at_utc : str
        When the entry was recorded.

    Returns
    -------
    SealedLookJournalEntry
        The validated v2 entry.

    Raises
    ------
    EvaluationContractError
        If the preregistration is not the one the request names, or a
        record or field is malformed.
    """
    validate_look_request(request)
    validate_preregistration(preregistration)
    if record_digest(preregistration) != request.preregistration_digest:
        raise EvaluationContractError(
            "the preregistration does not match the request's digest"
        )
    return validate_journal_entry(
        SealedLookJournalEntry(
            schema="sealed-look-journal-entry.v2",
            journal_seq=journal_seq,
            custody_seq_at_release=custody_seq_at_release,
            derived_proposer_lineage_id=derived_proposer_lineage_id,
            split_digest=request.split_digest,
            frozen_artefact_digests=request.frozen_artefact_digests,
            requester_role=request.requester_role,
            requester_holder=request.requester_holder,
            look_kind=request.look_kind,
            status=status,
            fault_attestation_digest=fault_attestation_digest,
            recorded_at_utc=recorded_at_utc,
            programme=request.programme,
            evaluator_name=request.evaluator_name,
            evaluator_digest=request.evaluator_digest,
            study_id=preregistration.study_id,
            preregistration_digest=request.preregistration_digest,
        )
    )


def check_journal_order(
    journal: Sequence[SealedLookJournalEntry], *, programme: str
) -> None:
    """Refuse a journal with a gap, a repeat or another programme's entry.

    Parameters
    ----------
    journal : Sequence[SealedLookJournalEntry]
        The programme's journal, in order.
    programme : str
        The programme of the look being admitted.

    Raises
    ------
    PreregistrationOrderError
        If the journal does not start at ``JOURNAL_ORIGIN_SEQ``, its
        ``journal_seq`` does not rise by exactly one, or an entry belongs to
        another programme.
    EvaluationContractError
        If an entry is malformed.
    """
    if journal and journal[0].journal_seq != JOURNAL_ORIGIN_SEQ:
        raise PreregistrationOrderError(
            f"the journal must start at journal_seq {JOURNAL_ORIGIN_SEQ}"
        )
    for position, entry in enumerate(journal):
        validate_journal_entry(entry)
        if entry.programme != programme:
            raise PreregistrationOrderError(
                "the journal holds another programme's entry"
            )
        if position and entry.journal_seq != journal[position - 1].journal_seq + 1:
            raise PreregistrationOrderError(
                "journal_seq must rise by exactly one, with no gap or repeat"
            )


def select_study_entries(
    journal: Sequence[SealedLookJournalEntry],
    *,
    study_id: str,
    programme: str,
    custody_records: Mapping[str, CustodyIngestRecord],
) -> list[SealedLookJournalEntry]:
    """Return the journal's earlier looks of a study, each tied to custody.

    Parameters
    ----------
    journal : Sequence[SealedLookJournalEntry]
        The programme's checked journal.
    study_id : str
        The study of the look being admitted.
    programme : str
        Its programme.
    custody_records : Mapping[str, CustodyIngestRecord]
        Every custody record, keyed by content digest.

    Returns
    -------
    list[SealedLookJournalEntry]
        Every entry of the study, whatever preregistration admitted it.

    Raises
    ------
    PreregistrationOrderError
        If a study entry names no custody preregistration of the programme.
    """
    entries = [entry for entry in journal if entry.study_id == study_id]
    for entry in entries:
        record = custody_records.get(entry.preregistration_digest)
        if (
            record is None
            or record.kind != "preregistration"
            or record.programme != programme
        ):
            raise PreregistrationOrderError(
                "a study's journal entry names no custody preregistration of the programme"
            )
    return entries
