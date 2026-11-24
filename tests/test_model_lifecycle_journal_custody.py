# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Journal Custody Witness Tests

"""Custody witnesses every journal entry and opening; the tail cannot hide a look."""

from __future__ import annotations

from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    ENTRY_KIND,
    OPENING_KIND,
    CustodyRecordError,
    EvaluationContractError,
    EvaluatorOutput,
    MalformedDigestError,
    PreregistrationOrderError,
    SealedLookJournalEntry,
    SealedLookOpening,
    SealedLookResult,
    SliceAggregateEstimate,
    canonical_json_bytes,
    from_canonical_json_bytes,
    journal_entry_for,
    look_opening_for,
    opening_of,
    record_digest,
    require_opening_custodied,
    validate_custody_ingest_record,
    validate_journal_entry,
    validate_look_opening,
    validate_look_result,
)
from tests.model_lifecycle_support import (
    CUSTODIAN,
    SPLITS,
    UTC,
    LookScenario,
    custody_record,
    digest_of,
    journal_custody,
    journal_entry,
    opening_record,
    preregistration,
)

OTHER = digest_of("other-artefact")
AMENDED = preregistration(planned_sample_size=4000)


def entry_of(seq: int = 1, **changes: object) -> SealedLookJournalEntry:
    """Return a released entry of another study at ``seq``, opened at 30."""
    lineage = digest_of(f"lineage:{seq}")
    base = journal_entry(
        lineage, (digest_of(f"artefact:{seq}"),), seq=seq, study_id="study-x"
    )
    return replace(base, custody_seq_at_open=30, custody_seq_at_release=40, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def scenario(*journal: SealedLookJournalEntry, **changes: object) -> LookScenario:
    """Return a scenario whose custody the test sets up by hand."""
    return LookScenario(journal=list(journal), ingest_journal=False, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def custodied(look: LookScenario, *entries: SealedLookJournalEntry) -> LookScenario:
    """Ingest each entry's record and opening into ``look``'s custody."""
    for entry in entries:
        look.records.update(journal_custody(entry, 50))
    return look


class TestCustodyWitness:
    """Each entry and opening is in custody, and custody holds nothing else."""

    def test_a_witnessed_journal_is_admitted(self) -> None:
        """Two custodied entries and their openings pass."""
        entries = (entry_of(1), entry_of(2))
        assert custodied(scenario(*entries), *entries).admit()

    def test_an_empty_journal_needs_no_records(self) -> None:
        """A first look has nothing to witness."""
        assert scenario().admit()

    def test_an_altered_entry_is_refused(self) -> None:
        """An entry changed after ingest has no record of its own."""
        entry = entry_of()
        look = custodied(scenario(replace(entry, study_id="study-y")), entry)
        with pytest.raises(PreregistrationOrderError, match="no custody record of"):
            look.admit()

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"kind": "preregistration"}, "no custody record of"),
            ({"programme": "pilot-b"}, "no custody record of"),
        ],
        ids=["kind", "programme"],
    )
    def test_an_entry_record_of_another_kind_or_programme_is_refused(
        self, changes: dict[str, object], fragment: str
    ) -> None:
        """The entry's record is a journal-entry record of the programme."""
        entry = entry_of()
        look = custodied(scenario(entry), entry)
        digest = record_digest(entry)
        look.records[digest] = replace(look.records[digest], **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields
        with pytest.raises(PreregistrationOrderError, match=fragment):
            look.admit()

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            (None, "opening has no custody record"),
            ({"kind": "preregistration"}, "opening has no custody record"),
            ({"programme": "pilot-b"}, "opening has no custody record"),
            ({"custody_ingest_seq": 41}, "ingested after its release"),
        ],
        ids=["missing", "kind", "programme", "after-release"],
    )
    def test_an_entry_needs_its_opening_by_its_release(
        self, changes: dict[str, object] | None, fragment: str
    ) -> None:
        """A-1: the opening's record exists, is of its kind and came first."""
        entry = entry_of()
        look = custodied(scenario(entry), entry)
        digest = record_digest(opening_of(entry))
        if changes is None:
            del look.records[digest]
        else:
            look.records[digest] = replace(look.records[digest], **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields
        with pytest.raises(PreregistrationOrderError, match=fragment):
            look.admit()

    def test_a_truncated_tail_is_refused(self) -> None:
        """Custody's later entry, left out of the journal, is refused."""
        first, second = entry_of(1), entry_of(2)
        look = custodied(scenario(first), first, second)
        with pytest.raises(PreregistrationOrderError, match="journal entries the"):
            look.admit()

    def test_a_hidden_opening_is_refused(self) -> None:
        """A look opened in custody but absent from the journal is refused."""
        entry = entry_of()
        look = custodied(scenario(entry), entry)
        hidden = opening_of(entry_of(2))
        look.add(record_digest(hidden), OPENING_KIND, 35, CUSTODIAN)
        with pytest.raises(PreregistrationOrderError, match="opening that the journal"):
            look.admit()

    def test_records_above_the_watermark_do_not_count(self) -> None:
        """A-2: an entry or an opening ingested after the watermark is not held yet."""
        first, second = entry_of(1), entry_of(2)
        look = custodied(scenario(first, second, watermark=60), first)
        look.records.update(journal_custody(second, 61))
        with pytest.raises(PreregistrationOrderError, match="or not yet"):
            look.admit()
        later = custodied(scenario(first, watermark=60), first)
        later.add(record_digest(opening_of(second)), OPENING_KIND, 61, CUSTODIAN)
        assert later.admit()

    def test_a_held_entry_of_another_programme_is_ignored(self) -> None:
        """Set equality is per programme."""
        entry = entry_of()
        look = custodied(scenario(entry), entry)
        foreign = digest_of("foreign-entry")
        look.add(foreign, ENTRY_KIND, 45, CUSTODIAN, programme="pilot-b")
        assert look.admit()


class TestStudyDefences:
    """One preregistration digest belongs to one study."""

    def test_a_digest_under_two_studies_is_refused(self) -> None:
        """Two custodied entries name one plan under two studies."""
        shared = digest_of("shared-plan")
        first = entry_of(1, preregistration_digest=shared)
        second = entry_of(2, preregistration_digest=shared, study_id="study-y")
        look = custodied(scenario(first, second), first, second)
        with pytest.raises(PreregistrationOrderError, match="under two studies"):
            look.admit()

    def test_the_requests_own_digest_under_another_study_is_refused(self) -> None:
        """An entry naming the request's plan must be of the request's study."""
        own = record_digest(preregistration())
        entry = entry_of(1, preregistration_digest=own)
        look = custodied(scenario(entry), entry)
        with pytest.raises(PreregistrationOrderError, match="under two studies"):
            look.admit()


class TestOpeningAnchor:
    """The amendment rule is anchored at the opening, not the release."""

    def test_an_amendment_while_a_look_ran_is_refused(self) -> None:
        """A plan ingested between an earlier look's opening and release is late."""
        earlier = journal_entry(
            digest_of("lineage:other"), (OTHER,), open_seq=8, release_seq=12
        )
        look = custodied(
            LookScenario(prereg=AMENDED, ingest_journal=False, journal=[earlier]),
            earlier,
        )
        with pytest.raises(PreregistrationOrderError, match="study opened"):
            look.admit()

    def test_an_amendment_before_the_opening_is_admitted(self) -> None:
        """A plan ingested before an earlier look opened is not an amendment after it."""
        earlier = journal_entry(
            digest_of("lineage:other"), (OTHER,), open_seq=12, release_seq=14
        )
        look = LookScenario(prereg=AMENDED, journal=[earlier])
        assert look.admit()


class TestOpeningRecord:
    """The opening record, its builder and its custody check."""

    def opening(self, **changes: object) -> SealedLookOpening:
        """Return the opening of the default look with ``changes`` applied."""
        look = LookScenario()
        base = look_opening_for(
            look.request(), look.prereg, prior_journal=(), look_custody_seq=100
        )
        return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields

    def test_the_builder_takes_the_next_position(self) -> None:
        """A first look is at position 1; a later one follows the journal."""
        look = LookScenario()
        assert self.opening().journal_seq == 1
        after = look_opening_for(
            look.request(),
            look.prereg,
            prior_journal=(entry_of(3),),
            look_custody_seq=70,
        )
        assert (after.journal_seq, after.custody_seq_at_open) == (4, 70)
        assert after.study_id == look.prereg.study_id

    def test_the_builder_refuses_another_preregistration(self) -> None:
        """The plan must be the one the request names."""
        look = LookScenario()
        with pytest.raises(EvaluationContractError, match="does not match"):
            look_opening_for(
                look.request(), AMENDED, prior_journal=(), look_custody_seq=1
            )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"schema": "v0"}, "schema"),
            ({"programme": ""}, "programme"),
            ({"journal_seq": 0}, "journal_seq"),
            ({"study_id": ""}, "study_id"),
            ({"look_kind": "peek"}, "look_kind"),
            ({"custody_seq_at_open": -1}, "custody_seq_at_open"),
        ],
    )
    def test_shape_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """Each field is checked by name."""
        with pytest.raises(EvaluationContractError, match=fragment):
            validate_look_opening(self.opening(**changes))

    @pytest.mark.parametrize("name", ["preregistration_digest", "split_digest"])
    def test_malformed_digests_are_refused(self, name: str) -> None:
        """The plan and split digests are SHA-256 hex."""
        with pytest.raises(MalformedDigestError):
            validate_look_opening(self.opening(**{name: "x"}))

    @pytest.mark.parametrize("kind", [OPENING_KIND, ENTRY_KIND])
    def test_the_new_custody_kinds_hold_metadata_only(self, kind: str) -> None:
        """An opening or entry record is not_applicable data and has no weights."""
        digest = digest_of(kind)
        assert validate_custody_ingest_record(
            custody_record(digest, kind, 1, CUSTODIAN)
        )
        with pytest.raises(CustodyRecordError, match="data_class"):
            validate_custody_ingest_record(
                custody_record(digest, kind, 1, CUSTODIAN, data_class="pilot")
            )

    def test_a_non_opening_is_refused(self) -> None:
        """The validator needs the record type."""
        with pytest.raises(EvaluationContractError, match="SealedLookOpening"):
            validate_look_opening(object())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it

    def test_an_entry_rebuilds_its_opening_and_the_record_round_trips(self) -> None:
        """The entry built from an opening claims exactly that opening."""
        look = LookScenario()
        opening = self.opening()
        entry = journal_entry_for(
            look.request(),
            look.prereg,
            opening=opening,
            custody_seq_at_release=120,
            derived_proposer_lineage_id=look.admit().derived_proposer_lineage_id,
            status="released",
            fault_attestation_digest="",
            recorded_at_utc=UTC,
        )
        assert opening_of(entry) == opening
        raw = canonical_json_bytes(opening)
        assert from_canonical_json_bytes(SealedLookOpening, raw) == opening

    def test_the_builder_refuses_an_opening_of_another_look(self) -> None:
        """The opening's programme, study, plan, split and kind are the request's."""
        look = LookScenario()
        with pytest.raises(EvaluationContractError, match="another look"):
            journal_entry_for(
                look.request(),
                look.prereg,
                opening=self.opening(split_digest=SPLITS[1]),
                custody_seq_at_release=120,
                derived_proposer_lineage_id=digest_of("lineage"),
                status="released",
                fault_attestation_digest="",
                recorded_at_utc=UTC,
            )

    def test_an_entry_released_before_it_opened_is_refused(self) -> None:
        """custody_seq_at_release is at least custody_seq_at_open."""
        with pytest.raises(EvaluationContractError, match="custody_seq_at_release"):
            validate_journal_entry(replace(entry_of(), custody_seq_at_open=41))

    def test_a_v2_entry_is_refused_by_name(self) -> None:
        """v2 recorded no opening, so it is retired."""
        with pytest.raises(EvaluationContractError, match="retired"):
            retired = "sealed-look-journal-entry.v2"
            validate_journal_entry(replace(entry_of(), schema=retired))  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            (
                {"content_digest": digest_of("other-opening")},
                "not of the look's opening",
            ),
            ({"kind": "preregistration"}, "another kind or programme"),
            ({"programme": "pilot-b"}, "another kind or programme"),
            ({"custody_ingest_seq": 41}, "after its release"),
        ],
        ids=["digest", "kind", "programme", "after-release"],
    )
    def test_the_custody_check_refuses_by_name(
        self, changes: dict[str, object], fragment: str
    ) -> None:
        """A-1 at use: the record is the entry's opening and came by its release."""
        entry = entry_of()
        record = replace(opening_record(entry), **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields
        with pytest.raises(EvaluationContractError, match=fragment):
            require_opening_custodied(entry, record)
        assert require_opening_custodied(
            entry, opening_record(entry)
        ) == opening_record(entry)

    def test_a_result_needs_its_opening_by_the_release(self) -> None:
        """A result released before its opening was custodied is never used."""
        entry = journal_entry(digest_of("lineage:r"), (digest_of("A1"),), open_seq=20)
        result = SealedLookResult(
            "sealed-look-result.v1",
            record_digest(entry),
            entry.derived_proposer_lineage_id,
            entry.split_digest,
            EvaluatorOutput(
                "evaluator-output.v1",
                "slice_estimates",
                (
                    SliceAggregateEstimate(
                        entry.frozen_artefact_digests[0],
                        "accuracy",
                        "all",
                        0.8,
                        0.75,
                        0.85,
                        40,
                    ),
                ),
                None,
                None,
            ),
        )
        late = custody_record(
            record_digest(opening_of(entry)), OPENING_KIND, 26, CUSTODIAN
        )
        with pytest.raises(EvaluationContractError, match="after its release"):
            validate_look_result(result, entry, opening_record=late)
