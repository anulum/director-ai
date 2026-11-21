# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Journal Custody Witness

"""Custody witnesses every journal entry and every look's opening.

Custody ingests each look's opening at admission, as a record of kind
``sealed_look_opening``, and each journal entry when the look finishes, as a
record of kind ``sealed_look_journal_entry``, both keyed by the record's
digest. :func:`check_journal_custody` then requires, for the programme up
to the look's watermark:

- every journal entry's own record, so an altered entry has none;
- the set of entry records to equal the journal's entries, so a truncated
  tail or an uncustodied insertion is refused;
- every entry's opening record, ingested no later than its release, and
  the set of opening records to equal the entries' openings, so a look that
  was opened but hidden from the journal is refused.

Two trusts remain: that ``custody_records`` is complete, and that custody
ingests each opening before it releases the look's split. The set equality
fails closed when an opening or an entry is ingested after the next look's
watermark: a liveness requirement on custody, not a gap. Custody ingests
each opening before the release and each entry before the next look opens.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .custody_records import CustodyIngestRecord
from .digests import record_digest
from .errors import PreregistrationOrderError
from .look_openings import OPENING_KIND, opening_of
from .sealed_looks import SealedLookJournalEntry

E = PreregistrationOrderError
ENTRY_KIND = "sealed_look_journal_entry"
"""The custody kind of a journal entry's record."""


def _held(
    records: Mapping[str, CustodyIngestRecord],
    kind: str,
    programme: str,
    watermark: int,
) -> set[str]:
    """Return the digests of the programme's records of ``kind`` up to the watermark."""
    return {
        digest
        for digest, record in records.items()
        if record.kind == kind
        and record.programme == programme
        and record.custody_ingest_seq <= watermark
    }


def _check_studies(
    journal: Sequence[SealedLookJournalEntry],
    *,
    study_id: str,
    preregistration_digest: str,
) -> None:
    """Refuse a preregistration digest named under two studies."""
    studies: dict[str, str] = {preregistration_digest: study_id}
    for entry in journal:
        known = studies.setdefault(entry.preregistration_digest, entry.study_id)
        if known != entry.study_id:
            raise E("one preregistration digest is named under two studies")


def check_journal_custody(
    journal: Sequence[SealedLookJournalEntry],
    *,
    programme: str,
    custody_records: Mapping[str, CustodyIngestRecord],
    look_custody_seq: int,
    study_id: str,
    preregistration_digest: str,
) -> None:
    """Refuse a journal that custody's entry and opening records do not witness.

    Parameters
    ----------
    journal : Sequence[SealedLookJournalEntry]
        The programme's checked journal before this look.
    programme : str
        The programme of the look being admitted.
    custody_records : Mapping[str, CustodyIngestRecord]
        Every validated custody record, keyed by content digest.
    look_custody_seq : int
        The look's custody watermark.
    study_id : str
        The study of the look being admitted.
    preregistration_digest : str
        The record digest of the preregistration it is admitted under.

    Raises
    ------
    PreregistrationOrderError
        If an entry or an opening has no record of its kind and programme,
        an opening was ingested after its look's release, custody holds an
        entry or an opening the journal does not, or a preregistration
        digest is named under two studies, the look's own included.
    """
    entries = {record_digest(entry): entry for entry in journal}
    openings = {record_digest(opening_of(entry)): entry for entry in journal}
    for digest in entries:
        record = custody_records.get(digest)
        if record is None or record.kind != ENTRY_KIND or record.programme != programme:
            raise E("a journal entry has no custody record of the programme")
    for digest, entry in openings.items():
        record = custody_records.get(digest)
        if (
            record is None
            or record.kind != OPENING_KIND
            or record.programme != programme
        ):
            raise E("a journal entry's opening has no custody record")
        if record.custody_ingest_seq > entry.custody_seq_at_release:
            raise E("a look's opening was ingested after its release")
    if _held(custody_records, ENTRY_KIND, programme, look_custody_seq) != set(entries):
        raise E("custody holds journal entries the journal does not, or not yet")
    if _held(custody_records, OPENING_KIND, programme, look_custody_seq) != set(
        openings
    ):
        raise E("custody holds a look's opening that the journal does not")
    _check_studies(
        journal, study_id=study_id, preregistration_digest=preregistration_digest
    )
