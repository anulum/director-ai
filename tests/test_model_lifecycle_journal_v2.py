# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Journal v2 Tests

"""Admission selects a study's earlier looks from the journal itself (R-5)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    JOURNAL_ORIGIN_SEQ,
    EvaluationContractError,
    LookBudgetExhaustedError,
    MalformedDigestError,
    NonCanonicalJsonError,
    PreregistrationOrderError,
    SealedLookJournalEntry,
    canonical_json_bytes,
    from_canonical_json_bytes,
    journal_entry_for,
    record_digest,
    validate_journal_entry,
)
from tests.model_lifecycle_support import (
    PROGRAMME,
    SPLITS,
    UTC,
    LookScenario,
    digest_of,
    journal_entry,
    preregistration,
)

A1 = digest_of("A1")
OTHER = digest_of("other-artefact")


def _lineage(*members: str) -> str:
    """Return a lineage id that no look in these tests shares."""
    return digest_of("lineage:" + ",".join(members))


AMENDED = preregistration(planned_sample_size=4000)
"""An amendment of the default preregistration of study ``study-1``."""


class TestStudySelection:
    """The amendment rule reads every earlier look of the study."""

    def test_an_earlier_look_under_an_earlier_preregistration_counts(self) -> None:
        """An amendment ingested after a look of the same study is refused."""
        earlier = journal_entry(_lineage(OTHER), (OTHER,), release_seq=8)
        assert earlier.preregistration_digest != record_digest(AMENDED)
        with pytest.raises(PreregistrationOrderError, match="amended after a look"):
            LookScenario(prereg=AMENDED, journal=[earlier]).admit()

    def test_an_earlier_look_under_an_earlier_preregistration_counts_for_the_budget(
        self,
    ) -> None:
        """The same lineage's released look on the split blocks the amended look."""
        scenario = LookScenario(prereg=AMENDED)
        released = journal_entry(
            scenario.admit().derived_proposer_lineage_id, (A1,), release_seq=25
        )
        with pytest.raises(LookBudgetExhaustedError, match="already consumed"):
            LookScenario(prereg=AMENDED, journal=[released]).admit()

    def test_a_look_of_another_study_on_the_same_pool_does_not_count(self) -> None:
        """Another study's earlier look does not make this preregistration an amendment."""
        other = journal_entry(
            _lineage(OTHER), (OTHER,), release_seq=5, study_id="study-other"
        )
        assert LookScenario(journal=[other]).admit()

    @pytest.mark.parametrize(
        ("digest", "programme", "kind"),
        [
            (digest_of("ghost-preregistration"), None, None),
            (digest_of("foreign-preregistration"), "pilot-b", "preregistration"),
            (SPLITS[0], None, None),
        ],
        ids=["missing", "other-programme", "not-a-preregistration"],
    )
    def test_a_study_entry_must_name_a_custody_preregistration_of_the_programme(
        self, digest: str, programme: str | None, kind: str | None
    ) -> None:
        """Each selected entry is tied to a custody preregistration of the programme."""
        scenario = LookScenario(
            journal=[
                journal_entry(_lineage(OTHER), (OTHER,), preregistration_digest=digest)
            ]
        )
        if kind is not None:
            scenario.add(digest, kind, 6, "role/custodian", programme=programme)
        with pytest.raises(
            PreregistrationOrderError, match="no custody preregistration"
        ):
            scenario.admit()


class TestJournalOrder:
    """The journal rises by one and belongs to one programme."""

    def test_a_consecutive_journal_is_admitted(self) -> None:
        """Entries 1, 2 and 3 of other studies pass."""
        journal = [
            journal_entry(
                _lineage(str(seq)), (digest_of(str(seq)),), seq=seq, study_id="study-x"
            )
            for seq in (1, 2, 3)
        ]
        assert LookScenario(journal=journal).admit()

    @pytest.mark.parametrize(
        "seqs", [(1, 3), (1, 1), (1, 2, 1)], ids=["gap", "repeat", "back"]
    )
    def test_a_gap_or_repeat_is_refused(self, seqs: tuple[int, ...]) -> None:
        """journal_seq must rise by exactly one from the origin."""
        journal = [
            journal_entry(
                _lineage(str(i)), (digest_of(str(i)),), seq=seq, study_id="study-x"
            )
            for i, seq in enumerate(seqs)
        ]
        with pytest.raises(PreregistrationOrderError, match="exactly one"):
            LookScenario(journal=journal).admit()

    @pytest.mark.parametrize("first", [2, 7])
    def test_a_journal_missing_its_head_is_refused(self, first: int) -> None:
        """A journal must start at its origin, so no earliest look can be left out."""
        assert JOURNAL_ORIGIN_SEQ == 1
        journal = [
            journal_entry(
                _lineage("a"), (digest_of("a"),), seq=first, study_id="study-x"
            )
        ]
        with pytest.raises(
            PreregistrationOrderError, match="must start at journal_seq 1"
        ):
            LookScenario(journal=journal).admit()

    def test_an_entry_below_the_origin_is_malformed(self) -> None:
        """The validator's minimum is the origin."""
        entry = journal_entry(_lineage("a"), (digest_of("a"),), seq=0)
        with pytest.raises(EvaluationContractError, match="journal_seq"):
            validate_journal_entry(entry)

    def test_an_entry_of_another_programme_is_refused(self) -> None:
        """The journal is per programme."""
        foreign = replace(
            journal_entry(_lineage("a"), (digest_of("a"),), study_id="study-x"),
            programme="pilot-b",
        )
        with pytest.raises(PreregistrationOrderError, match="another programme"):
            LookScenario(journal=[foreign]).admit()


class TestSchema:
    """v1 entries are refused by name; v2 names its study and preregistration."""

    def test_v1_is_refused_by_name(self) -> None:
        """The validator names the retired schema."""
        retired: dict[str, object] = {"schema": "sealed-look-journal-entry.v1"}
        entry = journal_entry(_lineage("a"), (digest_of("a"),))
        old = replace(entry, **retired)  # type: ignore[arg-type]
        with pytest.raises(EvaluationContractError, match="retired"):
            validate_journal_entry(old)

    def test_the_decoder_never_reads_v1_as_v2(self) -> None:
        """A v1 document is refused at decoding."""
        document = json.loads(
            canonical_json_bytes(journal_entry(_lineage("a"), (digest_of("a"),)))
        )
        document["schema"] = "sealed-look-journal-entry.v1"
        raw = json.dumps(
            document, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        with pytest.raises(NonCanonicalJsonError, match="schema"):
            from_canonical_json_bytes(SealedLookJournalEntry, raw)

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"study_id": ""}, EvaluationContractError),
            ({"preregistration_digest": "x"}, MalformedDigestError),
        ],
    )
    def test_v2_fields_are_checked(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """The study and preregistration fields are required."""
        entry = journal_entry(_lineage("a"), (digest_of("a"),))
        with pytest.raises(error):
            validate_journal_entry(replace(entry, **changes))  # type: ignore[arg-type]


class TestBuilder:
    """The kernel builds an entry from the admitted request and its plan."""

    def test_the_builder_takes_the_study_from_the_preregistration(self) -> None:
        """Every request field and the plan's study land in the entry."""
        prereg = preregistration(study_id="study-7")
        scenario = LookScenario(prereg=prereg)
        request = scenario.request()
        entry = journal_entry_for(
            request,
            prereg,
            journal_seq=4,
            custody_seq_at_release=40,
            derived_proposer_lineage_id=scenario.admit().derived_proposer_lineage_id,
            status="released",
            fault_attestation_digest="",
            recorded_at_utc=UTC,
        )
        assert entry.study_id == "study-7"
        assert entry.preregistration_digest == request.preregistration_digest
        assert (entry.split_digest, entry.frozen_artefact_digests, entry.programme) == (
            request.split_digest,
            request.frozen_artefact_digests,
            PROGRAMME,
        )
        assert (
            from_canonical_json_bytes(
                SealedLookJournalEntry, canonical_json_bytes(entry)
            )
            == entry
        )

    def test_the_builder_refuses_another_preregistration(self) -> None:
        """The plan must be the one the request names."""
        scenario = LookScenario()
        with pytest.raises(EvaluationContractError, match="does not match"):
            journal_entry_for(
                scenario.request(),
                AMENDED,
                journal_seq=1,
                custody_seq_at_release=30,
                derived_proposer_lineage_id=digest_of("lineage"),
                status="released",
                fault_attestation_digest="",
                recorded_at_utc=UTC,
            )
