# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Sealed Look Openings

"""The record of a look's opening, which custody holds before any result.

When a look is admitted, the kernel builds its opening with
:func:`look_opening_for` and custody ingests it, as a record of kind
``sealed_look_opening``, before it releases any data of the split. The
opening names the programme, the look's place in the journal, its study,
preregistration, split and kind, and the custody watermark at opening.

Each journal entry carries the same fields, so :func:`opening_of` rebuilds
the opening an entry claims and the kernel checks its custody record by
digest. A result is used only if its opening was ingested no later than
the look's release (:func:`require_opening_custodied`), so a look hidden
from the journal cannot have produced a usable result. That custody
releases a split only after the opening is ingested is a custody trust.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ._checks import require_choice, require_int, require_sha256, require_text
from .custody_records import CustodyIngestRecord, validate_custody_ingest_record
from .digests import record_digest
from .errors import EvaluationContractError
from .evaluation_contracts import PreregistrationRecord, validate_preregistration
from .sealed_looks import (
    JOURNAL_ORIGIN_SEQ,
    LOOK_KINDS,
    LookKind,
    SealedLookJournalEntry,
    SealedLookRequest,
    validate_journal_entry,
    validate_look_request,
)

E = EvaluationContractError
OPENING_KIND = "sealed_look_opening"
"""The custody kind of a look's opening record."""


@dataclass(frozen=True, slots=True)
class SealedLookOpening:
    """The opening of one admitted look, ingested before any result leaves.

    Attributes
    ----------
    schema : Literal["sealed-look-opening.v1"]
        Record schema.
    programme : str
        The look's programme.
    journal_seq : int
        The look's place in the programme's journal.
    study_id : str
        The study of the preregistration the look was admitted under.
    preregistration_digest : str
        That preregistration's record digest.
    split_digest : str
        The split the look opens.
    look_kind : LookKind
        The look's kind.
    custody_seq_at_open : int
        The custody watermark at admission.
    """

    schema: Literal["sealed-look-opening.v1"]
    programme: str
    journal_seq: int
    study_id: str
    preregistration_digest: str
    split_digest: str
    look_kind: LookKind
    custody_seq_at_open: int


def validate_look_opening(opening: SealedLookOpening) -> SealedLookOpening:
    """Check an opening record's own fields.

    Parameters
    ----------
    opening : SealedLookOpening
        The opening to check.

    Returns
    -------
    SealedLookOpening
        The same opening.

    Raises
    ------
    EvaluationContractError
        If the record or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(opening, SealedLookOpening):
        raise E("a SealedLookOpening is required")
    require_choice(opening.schema, "schema", {"sealed-look-opening.v1"}, E)
    require_text(opening.programme, "programme", E)
    require_int(opening.journal_seq, "journal_seq", E, minimum=JOURNAL_ORIGIN_SEQ)
    require_text(opening.study_id, "study_id", E)
    require_sha256(opening.preregistration_digest, "preregistration_digest")
    require_sha256(opening.split_digest, "split_digest")
    require_choice(opening.look_kind, "look_kind", LOOK_KINDS, E)
    require_int(opening.custody_seq_at_open, "custody_seq_at_open", E, minimum=0)
    return opening


def look_opening_for(
    request: SealedLookRequest,
    preregistration: PreregistrationRecord,
    *,
    prior_journal: Sequence[SealedLookJournalEntry],
    look_custody_seq: int,
) -> SealedLookOpening:
    """Build the opening of an admitted look, at the next journal position.

    Parameters
    ----------
    request : SealedLookRequest
        The admitted request.
    preregistration : PreregistrationRecord
        The preregistration the request names.
    prior_journal : Sequence[SealedLookJournalEntry]
        The programme's journal before this look, as admission checked it.
    look_custody_seq : int
        The custody watermark admission used.

    Returns
    -------
    SealedLookOpening
        The validated opening, for custody to ingest before any release.

    Raises
    ------
    EvaluationContractError
        If the preregistration is not the request's, or a field is malformed.
    """
    validate_look_request(request)
    validate_preregistration(preregistration)
    if record_digest(preregistration) != request.preregistration_digest:
        raise E("the preregistration does not match the request's digest")
    seq = prior_journal[-1].journal_seq + 1 if prior_journal else JOURNAL_ORIGIN_SEQ
    return validate_look_opening(
        SealedLookOpening(
            "sealed-look-opening.v1",
            request.programme,
            seq,
            preregistration.study_id,
            request.preregistration_digest,
            request.split_digest,
            request.look_kind,
            look_custody_seq,
        )
    )


def opening_of(entry: SealedLookJournalEntry) -> SealedLookOpening:
    """Return the opening a journal entry claims, rebuilt from its fields.

    Parameters
    ----------
    entry : SealedLookJournalEntry
        A journal entry.

    Returns
    -------
    SealedLookOpening
        The opening with the entry's programme, position, study, plan, split,
        kind and opening watermark.

    Raises
    ------
    EvaluationContractError
        If the entry is malformed.
    """
    validate_journal_entry(entry)
    return SealedLookOpening(
        "sealed-look-opening.v1",
        entry.programme,
        entry.journal_seq,
        entry.study_id,
        entry.preregistration_digest,
        entry.split_digest,
        entry.look_kind,
        entry.custody_seq_at_open,
    )


def require_opening_custodied(
    entry: SealedLookJournalEntry, record: CustodyIngestRecord
) -> CustodyIngestRecord:
    """Refuse a look whose opening custody did not hold by its release.

    Parameters
    ----------
    entry : SealedLookJournalEntry
        The look's journal entry.
    record : CustodyIngestRecord
        Custody's record of the look's opening.

    Returns
    -------
    CustodyIngestRecord
        The same record.

    Raises
    ------
    EvaluationContractError
        If the record is not of the entry's opening, of kind
        ``sealed_look_opening`` and of the entry's programme, or was ingested
        after the look's release.
    CustodyRecordError
        If the record is malformed.
    """
    validate_custody_ingest_record(record)
    if record.content_digest != record_digest(opening_of(entry)):
        raise E("the custody record is not of the look's opening")
    if record.kind != OPENING_KIND or record.programme != entry.programme:
        raise E("the opening's custody record is of another kind or programme")
    if record.custody_ingest_seq > entry.custody_seq_at_release:
        raise E("the look's opening was ingested after its release")
    return record
