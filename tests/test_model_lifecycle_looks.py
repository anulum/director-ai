# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — model lifecycle sealed look tests

"""Sealed look admission, custody-derived lineage, the budget and promotion.

The scenarios follow the kernel's mutants: a backdated or new-context
artefact stays in its lineage (M-J4-6), a candidate ingested before its
preregistration is refused (M-J4-7), pools and approvals come only from
their roles (M-J4-8), a tainted base merges while a clean base does not
(M-J4-9), and ``candidate_failed`` consumes and taints like ``released``
(M-J4-9b).
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    CustodyRecordError,
    EvaluationContractError,
    EvaluatorOutput,
    LookBudgetExhaustedError,
    MalformedDigestError,
    PreregistrationOrderError,
    PromotionDecisionError,
    PromotionDecisionRecord,
    SealedLookResult,
    SealedPoolCustodyError,
    SliceAggregateEstimate,
    UnverifiedCustodyArtefactError,
    check_look_budget,
    check_promotion_pre_state,
    derive_proposer_lineage,
    record_digest,
    validate_evaluator_output,
    validate_journal_entry,
    validate_look_request,
    validate_look_result,
    validate_promotion_decision,
    validate_slice_estimate,
)
from tests.model_lifecycle_support import (
    APPROVER,
    CUSTODIAN,
    PROPOSER_A,
    PROPOSER_B,
    SPLITS,
    UTC,
    LookScenario,
    custody_record,
    digest_of,
    edge,
    journal_entry,
    preregistration,
)

A1, A2, B1, BASE = digest_of("A1"), digest_of("A2"), digest_of("B1"), digest_of("base")


def _lineage_id(seq: int, digest: str) -> str:
    """Return the root-anchored lineage id of a root record."""
    return hashlib.sha256(f"{seq}:{digest}".encode()).hexdigest()


class TestAdmission:
    """A valid look and each binding refusal before a split is opened."""

    def test_a_valid_look_is_admitted_with_a_root_anchored_id(self) -> None:
        """The lineage is the arm's role class, rooted at its first ingest."""
        component = LookScenario().admit()
        assert component.derived_proposer_lineage_id == _lineage_id(20, A1)
        assert component.members == frozenset({A1})
        assert component.exempt_base_digests == frozenset()

    @pytest.mark.parametrize(
        ("changes", "error", "message"),
        [
            (
                {"preregistration_digest": digest_of("other")},
                EvaluationContractError,
                "does not match",
            ),
            (
                {"split_digest": digest_of("elsewhere")},
                EvaluationContractError,
                "not in the preregistered pool",
            ),
            (
                {"frozen_artefact_digests": (A1, A2)},
                EvaluationContractError,
                "exactly the preregistered",
            ),
            ({"programme": "pilot-b"}, SealedPoolCustodyError, "another programme"),
        ],
    )
    def test_request_binding_refusals(
        self, changes: dict[str, object], error: type[Exception], message: str
    ) -> None:
        """The request must name its preregistration, pool, arms and programme."""
        with pytest.raises(error, match=message):
            LookScenario().admit(**changes)

    def test_a_candidate_ingested_before_its_preregistration_is_refused(self) -> None:
        """M-J4-7: custody order decides, a backdated declared time does not."""
        scenario = LookScenario(
            prereg=preregistration(declared_registered_at_utc="2020-01-01T00:00:00Z")
        )
        scenario.add(A1, "artefact", 5, PROPOSER_A)
        with pytest.raises(
            PreregistrationOrderError, match="before its preregistration"
        ):
            scenario.admit()

    def test_a_candidate_ingested_before_the_approval_is_refused(self) -> None:
        """E10-17: the approval precedes every candidate."""
        scenario = LookScenario(approval_seq=30)
        with pytest.raises(PreregistrationOrderError, match="before the approval"):
            scenario.admit()

    def test_the_requester_role_must_have_ingested_a_candidate(self) -> None:
        """R-3: a request in a role that ingested none of its candidates is refused."""
        with pytest.raises(EvaluationContractError, match="ingested none"):
            LookScenario(requester=PROPOSER_B).admit()

    def test_ingests_after_the_watermark_are_refused(self) -> None:
        """The look cannot reference a record custody has not acknowledged."""
        with pytest.raises(
            PreregistrationOrderError, match="frozen artefact was ingested after"
        ):
            LookScenario(watermark=15).admit()
        with pytest.raises(
            PreregistrationOrderError, match="approval was ingested after"
        ):
            LookScenario(watermark=10).admit()

    def test_a_pool_split_from_the_proposer_is_refused(self) -> None:
        """M-J4-8 (pools): only the custodian role mints a pool's splits."""
        scenario = LookScenario()
        scenario.add(SPLITS[2], "split", 3, PROPOSER_A)
        with pytest.raises(SealedPoolCustodyError, match="custodian role"):
            scenario.admit()

    def test_a_pool_split_of_another_programme_is_refused(self) -> None:
        """Splits belong to the request's programme."""
        scenario = LookScenario()
        scenario.add(SPLITS[1], "split", 2, CUSTODIAN, programme="pilot-b")
        with pytest.raises(SealedPoolCustodyError, match="custodian role"):
            scenario.admit()

    def test_an_approval_from_another_role_is_refused(self) -> None:
        """M-J4-8 (approvals): only the threshold approver role counts."""
        with pytest.raises(SealedPoolCustodyError, match="threshold approver"):
            LookScenario(approval_role=PROPOSER_A).admit()

    def test_a_missing_approval_is_refused(self) -> None:
        """M-J4-8: a preregistration without an approval record cannot look."""
        with pytest.raises(
            UnverifiedCustodyArtefactError, match="the approval has no custody"
        ):
            LookScenario(ingest_approval=False).admit()
        scenario = LookScenario()
        approval_digest = record_digest(scenario.approval())
        scenario.records[approval_digest] = custody_record(
            approval_digest, "corpus", 11, APPROVER
        )
        with pytest.raises(
            EvaluationContractError, match="the approval is custody kind"
        ):
            scenario.admit()

    def test_an_approval_of_another_pool_is_refused(self) -> None:
        """The approval must name this preregistration and its pool."""
        scenario = LookScenario()
        other = replace(scenario.approval(), pool_id="pool-2")
        digest = record_digest(other)
        scenario.records[digest] = custody_record(
            digest, "preregistration_approval", 11, APPROVER
        )
        scenario.approval = lambda: other  # type: ignore[method-assign]
        with pytest.raises(
            SealedPoolCustodyError, match="another preregistration or pool"
        ):
            scenario.admit()

    def test_records_of_another_programme_are_refused(self) -> None:
        """The preregistration and approval belong to the programme."""
        scenario = LookScenario()
        digest = record_digest(scenario.prereg)
        scenario.records[digest] = custody_record(
            digest, "preregistration", 10, programme="pilot-b"
        )
        with pytest.raises(SealedPoolCustodyError, match="belong to the programme"):
            scenario.admit()

    def test_missing_and_mistyped_custody_records_are_refused(self) -> None:
        """Every referenced digest needs a record of the right kind."""
        missing = LookScenario()
        del missing.records[A1]
        with pytest.raises(UnverifiedCustodyArtefactError, match="frozen artefact"):
            missing.admit()
        wrong = LookScenario()
        wrong.add(A1, "corpus", 20, PROPOSER_A)
        with pytest.raises(
            EvaluationContractError, match="frozen artefact is custody kind"
        ):
            wrong.admit()

    def test_records_keyed_under_another_digest_are_refused(self) -> None:
        """The mapping key must be the record's content digest."""
        scenario = LookScenario()
        scenario.records[A2] = scenario.records[A1]
        with pytest.raises(CustodyRecordError, match="keyed by their content_digest"):
            scenario.admit()

    def test_a_preregistration_amended_after_a_study_look_is_refused(self) -> None:
        """An amendment ingested after a study look finished is refused."""
        study_look = journal_entry(
            _lineage_id(1, digest_of("x")), (digest_of("x"),), release_seq=9
        )
        with pytest.raises(PreregistrationOrderError, match="amended after a look"):
            LookScenario(journal=[study_look]).admit()

    def test_a_preregistration_ingested_up_to_the_watermark_is_not_an_amendment(
        self,
    ) -> None:
        """The watermark is inclusive, so the original preregistration passes."""
        study_look = journal_entry(
            _lineage_id(1, digest_of("x")), (digest_of("x"),), release_seq=10
        )
        assert LookScenario(journal=[study_look]).admit()

    @pytest.mark.parametrize("field", ["configured_custodian", "configured_approver"])
    def test_empty_configured_roles_are_refused(self, field: str) -> None:
        """A programme's custodian and approver roles must be named."""
        scenario = LookScenario()
        setattr(scenario, field, "")
        with pytest.raises(SealedPoolCustodyError, match="_role must be"):
            scenario.admit()

    def test_a_negative_watermark_is_refused(self) -> None:
        """The watermark is a custody sequence."""
        with pytest.raises(PreregistrationOrderError, match="look_custody_seq"):
            LookScenario(watermark=-1).admit()


class TestLineageAndBudget:
    """Custody-derived lineage, taint and the per-split budget."""

    def test_a_second_initial_look_on_a_split_is_refused(self) -> None:
        """One confirmatory look per lineage and split."""
        released = journal_entry(_lineage_id(20, A1), (A1,))
        with pytest.raises(LookBudgetExhaustedError, match="already consumed"):
            LookScenario(journal=[released]).admit()

    def test_the_lineage_may_look_once_on_each_split(self) -> None:
        """A release on one split leaves the pool's other splits open."""
        released = journal_entry(_lineage_id(20, A1), (A1,))
        assert LookScenario(journal=[released], split=SPLITS[1]).admit()

    def test_a_backdated_new_context_artefact_stays_in_its_lineage(self) -> None:
        """M-J4-6: same role, a new search run and no edges: still refused."""
        released = journal_entry(_lineage_id(20, A1), (A1,), release_seq=25)
        prereg = preregistration(frozen_look_candidate_digests=(A2,))
        scenario = LookScenario(prereg=prereg, journal=[released])
        scenario.add(A2, "artefact", 30, PROPOSER_A)
        with pytest.raises(LookBudgetExhaustedError, match="already consumed"):
            scenario.admit()

    def test_a_requester_role_artefact_after_a_release_is_tainted(self) -> None:
        """Post-look taint follows the requester role across submitting roles."""
        released = journal_entry(
            _lineage_id(20, B1), (B1,), release_seq=25, role=PROPOSER_A
        )
        prereg = preregistration(frozen_look_candidate_digests=(A2,))
        scenario = LookScenario(prereg=prereg, journal=[released])
        scenario.add(B1, "artefact", 20, PROPOSER_B)
        scenario.add(A2, "artefact", 30, PROPOSER_A)
        with pytest.raises(LookBudgetExhaustedError, match="already consumed"):
            scenario.admit()

    def test_a_requester_role_artefact_before_the_release_is_not_tainted(self) -> None:
        """Taint compares custody sequences: earlier ingests stay apart."""
        released = journal_entry(
            _lineage_id(20, B1), (B1,), release_seq=25, role=PROPOSER_A
        )
        prereg = preregistration(frozen_look_candidate_digests=(A2,))
        scenario = LookScenario(prereg=prereg, journal=[released])
        scenario.add(B1, "artefact", 22, PROPOSER_B)
        scenario.add(A2, "artefact", 21, PROPOSER_A)
        assert scenario.admit().members == frozenset({A1, A2})

    def test_candidate_failed_consumes_like_released(self) -> None:
        """M-J4-9b: a candidate failure spends the look."""
        failed = journal_entry(_lineage_id(20, A1), (A1,), status="candidate_failed")
        with pytest.raises(LookBudgetExhaustedError, match="already consumed"):
            LookScenario(journal=[failed]).admit()

    @pytest.mark.parametrize(
        "status", ["infrastructure_fault", "voided_evaluator_fault"]
    )
    def test_one_shared_fault_relook(self, status: str) -> None:
        """A fault allows one relook; an initial look or a second relook is refused."""
        fault = journal_entry(_lineage_id(20, A1), (A1,), status=status)
        assert LookScenario(journal=[fault], look_kind="fault_relook").admit()
        with pytest.raises(LookBudgetExhaustedError, match="only a fault_relook"):
            LookScenario(journal=[fault]).admit()
        relook = journal_entry(
            _lineage_id(20, A1),
            (A1,),
            status="voided_evaluator_fault",
            look_kind="fault_relook",
            seq=2,
        )
        with pytest.raises(LookBudgetExhaustedError, match="one fault_relook"):
            LookScenario(journal=[fault, relook], look_kind="fault_relook").admit()

    def test_a_relook_needs_an_earlier_fault(self) -> None:
        """A relook without a fault is refused."""
        with pytest.raises(
            LookBudgetExhaustedError, match="needs an earlier attested fault"
        ):
            LookScenario(look_kind="fault_relook").admit()

    def test_a_declared_edge_merges_into_a_released_lineage(self) -> None:
        """An edge from a released artefact joins its lineage, and edges only merge."""
        released = journal_entry(_lineage_id(20, A1), (A1,), release_seq=25)
        prereg = preregistration(frozen_look_candidate_digests=(B1,))
        scenario = LookScenario(
            prereg=prereg,
            journal=[released],
            requester=PROPOSER_B,
            edges=(edge(A1, B1),),
        )
        scenario.add(B1, "artefact", 30, PROPOSER_B)
        with pytest.raises(LookBudgetExhaustedError):
            scenario.admit()
        scenario.edges = ()
        assert scenario.admit().members == frozenset({B1})

    def test_a_tainted_base_merges_its_derivatives(self) -> None:
        """M-J4-9: a base already in a released look on the split merges."""
        released = journal_entry(
            _lineage_id(20, A1), (A1,), release_seq=25, study_id="study-a"
        )
        prereg = preregistration(
            frozen_look_candidate_digests=(B1,), base_checkpoint_digests=(A1,)
        )
        scenario = LookScenario(
            prereg=prereg,
            journal=[released],
            requester=PROPOSER_B,
            edges=(edge(A1, B1),),
            prereg_seq=26,
            approval_seq=27,
        )
        scenario.add(B1, "artefact", 30, PROPOSER_B)
        with pytest.raises(LookBudgetExhaustedError, match="already consumed"):
            scenario.admit()

    def test_a_base_tainted_by_a_candidate_failure_merges(self) -> None:
        """M-J4-9b: candidate_failed taints a base like released."""
        failed = journal_entry(
            _lineage_id(20, A1),
            (A1,),
            status="candidate_failed",
            release_seq=25,
            study_id="study-a",
        )
        prereg = preregistration(
            frozen_look_candidate_digests=(B1,), base_checkpoint_digests=(A1,)
        )
        scenario = LookScenario(
            prereg=prereg,
            journal=[failed],
            requester=PROPOSER_B,
            edges=(edge(A1, B1),),
            prereg_seq=26,
            approval_seq=27,
        )
        scenario.add(B1, "artefact", 30, PROPOSER_B)
        with pytest.raises(LookBudgetExhaustedError):
            scenario.admit()

    def test_independent_proposers_of_one_clean_base_keep_their_looks(self) -> None:
        """M-J4-9 control: an untainted base does not merge unrelated proposers."""
        released = journal_entry(
            _lineage_id(20, A1), (A1,), release_seq=25, study_id="study-a"
        )
        prereg = preregistration(
            frozen_look_candidate_digests=(B1,), base_checkpoint_digests=(BASE,)
        )
        scenario = LookScenario(
            prereg=prereg,
            journal=[released],
            requester=PROPOSER_B,
            edges=(edge(BASE, A1), edge(BASE, B1)),
            prereg_seq=26,
            approval_seq=27,
        )
        scenario.add(BASE, "base_checkpoint", 4, "role/importer")
        scenario.add(B1, "artefact", 30, PROPOSER_B)
        component = scenario.admit()
        assert component.exempt_base_digests == frozenset({BASE})
        assert component.members == frozenset({B1})

    def test_a_base_ingested_after_the_preregistration_is_an_ordinary_arm(self) -> None:
        """A late base is not exempt and must follow the preregistration."""
        prereg = preregistration(
            frozen_look_candidate_digests=(A1, BASE), base_checkpoint_digests=(BASE,)
        )
        scenario = LookScenario(prereg=prereg)
        scenario.add(BASE, "base_checkpoint", 15, "role/importer")
        component = scenario.admit()
        assert component.exempt_base_digests == frozenset()
        assert component.members == frozenset({A1, BASE})

    def test_a_base_in_the_requester_role_after_a_release_is_tainted(self) -> None:
        """A base the requester ingested after its release is not clean."""
        released = journal_entry(
            _lineage_id(20, A1), (A1,), release_seq=25, study_id="study-a"
        )
        prereg = preregistration(
            frozen_look_candidate_digests=(B1,), base_checkpoint_digests=(BASE,)
        )
        scenario = LookScenario(
            prereg=prereg,
            journal=[released],
            requester=PROPOSER_B,
            edges=(edge(BASE, B1),),
            prereg_seq=40,
            approval_seq=41,
        )
        scenario.add(BASE, "base_checkpoint", 30, PROPOSER_A)
        scenario.add(B1, "artefact", 50, PROPOSER_B)
        with pytest.raises(LookBudgetExhaustedError):
            scenario.admit()

    def test_a_look_of_exempt_bases_only_is_refused(self) -> None:
        """A look needs a candidate besides the clean bases."""
        prereg = preregistration(
            frozen_look_candidate_digests=(BASE,), base_checkpoint_digests=(BASE,)
        )
        scenario = LookScenario(prereg=prereg)
        scenario.add(BASE, "base_checkpoint", 4, "role/importer")
        with pytest.raises(PreregistrationOrderError, match="not an exempt base"):
            scenario.admit()

    def test_looks_outside_custody_do_not_taint(self) -> None:
        """A consuming look of unknown artefacts by an unknown role merges nothing."""
        stranger = journal_entry(
            _lineage_id(1, A2),
            (A2,),
            role="role/unknown",
            release_seq=5,
            study_id="study-other",
        )
        assert LookScenario(journal=[stranger]).admit().members == frozenset({A1})

    def test_derivation_refuses_unknown_candidates(self) -> None:
        """Direct callers get the same custody refusal."""
        with pytest.raises(UnverifiedCustodyArtefactError):
            derive_proposer_lineage(
                (A1,), exempt_bases=frozenset(), records={}, edges=(), prior_journal=()
            )

    def test_the_budget_check_alone_refuses_a_consumed_split(self) -> None:
        """check_look_budget works on any component."""
        component = LookScenario().admit()
        released = journal_entry(component.derived_proposer_lineage_id, (A1,))
        with pytest.raises(LookBudgetExhaustedError):
            check_look_budget(
                component,
                split_digest=SPLITS[0],
                look_kind="initial_confirmatory",
                prior_journal=[released],
            )


def _estimate(**changes: object) -> SliceAggregateEstimate:
    """Return a valid slice estimate for ``A1``."""
    base = SliceAggregateEstimate(A1, "accuracy", "all", 0.8, 0.75, 0.85, 40)
    return replace(base, **changes)  # type: ignore[arg-type]


def _output(**changes: object) -> EvaluatorOutput:
    """Return a valid slice-estimate evaluator output."""
    base = EvaluatorOutput(
        "evaluator-output.v1", "slice_estimates", (_estimate(),), None, None
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestLookRecords:
    """Request, journal entry, evaluator output and result validators."""

    def test_request_and_entry_validators_accept_valid_records(self) -> None:
        """The builders' records are valid."""
        request = LookScenario().request()
        assert validate_look_request(request) == request
        entry = journal_entry(_lineage_id(20, A1), (A1,))
        assert validate_journal_entry(entry) == entry

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "request"}, EvaluationContractError),
            ({"split_digest": ""}, MalformedDigestError),
            ({"frozen_artefact_digests": ()}, EvaluationContractError),
            ({"declared_lineage_edges": ("edge",)}, EvaluationContractError),
            ({"declared_lineage_edges": [edge(A1, A2)]}, EvaluationContractError),
            ({"requester_role": ""}, EvaluationContractError),
            ({"requester_holder": ""}, EvaluationContractError),
            ({"look_kind": "second_look"}, EvaluationContractError),
            ({"evaluator_name": ""}, EvaluationContractError),
            ({"evaluator_digest": ""}, MalformedDigestError),
        ],
    )
    def test_request_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Each request field is exact."""
        with pytest.raises(error):
            validate_look_request(LookScenario().request(**changes))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"journal_seq": -1}, EvaluationContractError),
            ({"status": "passed"}, EvaluationContractError),
            ({"fault_attestation_digest": digest_of("a")}, EvaluationContractError),
            ({"status": "infrastructure_fault"}, MalformedDigestError),
            ({"recorded_at_utc": ""}, EvaluationContractError),
            ({"programme": ""}, EvaluationContractError),
        ],
    )
    def test_entry_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """The attestation digest is present exactly for the two faults."""
        entry = journal_entry(_lineage_id(20, A1), (A1,))
        with pytest.raises(error):
            validate_journal_entry(replace(entry, **changes))  # type: ignore[arg-type]

    def test_output_forms(self) -> None:
        """Exactly one aggregate form is filled."""
        assert validate_evaluator_output(_output()) == _output()
        counts = _output(
            output_mode="test_suite_counts",
            slice_estimates=(),
            passed_count=9,
            run_count=10,
        )
        assert validate_evaluator_output(counts) == counts
        bad = [
            _output(passed_count=1),
            _output(slice_estimates=()),
            _output(slice_estimates=("x",)),
            _output(output_mode="raw_items"),
            replace(counts, slice_estimates=(_estimate(),)),
            replace(counts, passed_count=11),
            replace(counts, run_count=0),
            _output(schema="output"),
        ]
        for output in bad:
            with pytest.raises(EvaluationContractError):
                validate_evaluator_output(output)

    @pytest.mark.parametrize(
        "changes",
        [
            {"metric_id": ""},
            {"slice_id": ""},
            {"point_estimate": 0.9},
            {"ci_lower": float("nan")},
            {"cluster_count": 0},
        ],
    )
    def test_slice_estimate_refusals(self, changes: dict[str, object]) -> None:
        """Values are finite, the point inside its interval, clusters counted."""
        with pytest.raises(EvaluationContractError):
            validate_slice_estimate(_estimate(**changes))

    def test_a_result_binds_to_its_released_entry(self) -> None:
        """A result names its entry's digest, lineage, split and artefacts."""
        entry = journal_entry(_lineage_id(20, A1), (A1,))
        result = SealedLookResult(
            "sealed-look-result.v1",
            record_digest(entry),
            entry.derived_proposer_lineage_id,
            entry.split_digest,
            _output(),
        )
        assert validate_look_result(result, entry) == result
        bad = [
            (replace(result, schema="result"), entry),  # type: ignore[arg-type]
            (result, replace(entry, status="candidate_failed")),
            (replace(result, look_journal_digest=digest_of("other")), entry),
            (replace(result, split_digest=SPLITS[1]), entry),
            (
                replace(
                    result,
                    evaluator_output=_output(
                        slice_estimates=(_estimate(artefact_digest=A2),)
                    ),
                ),
                entry,
            ),
        ]
        for candidate, claimed in bad:
            with pytest.raises(EvaluationContractError):
                validate_look_result(candidate, claimed)

    def test_wrong_record_types_are_refused(self) -> None:
        """Each look validator needs its own record type."""
        for validator in (
            validate_look_request,
            validate_journal_entry,
            validate_evaluator_output,
            validate_slice_estimate,
        ):
            with pytest.raises(EvaluationContractError):
                validator("record")  # type: ignore[arg-type]


def _decision(**changes: object) -> PromotionDecisionRecord:
    """Return a valid generation-3 promotion decision for ``A1``."""
    base = PromotionDecisionRecord(
        schema="model-promotion-decision.v1",
        intent_sha256=digest_of("intent"),
        model_slot="guard-default",
        expected_active_generation=3,
        expected_production_digest=digest_of("production"),
        candidate_content_digest=A1,
        preregistration_digest=digest_of("prereg"),
        sealed_look_result_digest=digest_of("result"),
        rigor_admission_digest=digest_of("rigor"),
        remanentia_control_packet_digest=digest_of("packet"),
        remanentia_candidate_blob_digest=digest_of("candidate-blob"),
        remanentia_rollback_blob_digest=digest_of("rollback-blob"),
        decided_at_utc=UTC,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


class TestPromotion:
    """Promotion decisions and the production pre-state."""

    def test_a_valid_decision_and_its_pre_state(self) -> None:
        """A decision applies only while the slot shows its expected state."""
        decision = validate_promotion_decision(_decision())
        check_promotion_pre_state(
            decision, active_generation=3, production_digest=digest_of("production")
        )
        with pytest.raises(PromotionDecisionError, match="active generation"):
            check_promotion_pre_state(
                decision, active_generation=4, production_digest=digest_of("production")
            )
        with pytest.raises(PromotionDecisionError, match="production digest"):
            check_promotion_pre_state(
                decision, active_generation=3, production_digest=digest_of("stale")
            )

    def test_generation_zero_expects_the_empty_digest(self) -> None:
        """Generation 0, and only it, has 64 zeros as its production digest."""
        assert validate_promotion_decision(
            _decision(expected_active_generation=0, expected_production_digest="0" * 64)
        )
        with pytest.raises(PromotionDecisionError, match="generation 0"):
            validate_promotion_decision(_decision(expected_active_generation=0))
        with pytest.raises(PromotionDecisionError, match="generation 0"):
            validate_promotion_decision(_decision(expected_production_digest="0" * 64))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"schema": "decision"}, PromotionDecisionError),
            ({"intent_sha256": ""}, MalformedDigestError),
            ({"model_slot": ""}, PromotionDecisionError),
            ({"expected_active_generation": -1}, PromotionDecisionError),
            (
                {"candidate_content_digest": digest_of("production")},
                PromotionDecisionError,
            ),
            ({"rigor_admission_digest": ""}, MalformedDigestError),
            ({"decided_at_utc": ""}, PromotionDecisionError),
        ],
    )
    def test_decision_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Every decision field is exact."""
        with pytest.raises(error):
            validate_promotion_decision(_decision(**changes))

    def test_the_look_must_belong_to_the_decision(self) -> None:
        """The result binds by digest to its entry, which froze the candidate."""
        entry = journal_entry(_lineage_id(20, A1), (A1, A2))

        def result_of(
            output: EvaluatorOutput, claimed: object = entry
        ) -> SealedLookResult:
            """Return a result for ``output`` bound to ``claimed``."""
            return SealedLookResult(
                "sealed-look-result.v1",
                record_digest(claimed),
                entry.derived_proposer_lineage_id,
                entry.split_digest,
                output,
            )

        def bound(
            result: SealedLookResult, **changes: object
        ) -> PromotionDecisionRecord:
            """Return a decision that names ``result``."""
            return _decision(sealed_look_result_digest=record_digest(result), **changes)

        result = result_of(_output())
        assert validate_promotion_decision(
            bound(result), look_result=result, journal_entry=entry
        )
        with pytest.raises(PromotionDecisionError, match="checked together"):
            validate_promotion_decision(bound(result), look_result=result)
        with pytest.raises(PromotionDecisionError, match="checked together"):
            validate_promotion_decision(bound(result), journal_entry=entry)
        with pytest.raises(PromotionDecisionError, match="does not match"):
            validate_promotion_decision(
                _decision(), look_result=result, journal_entry=entry
            )
        stray = result_of(_output(), claimed=replace(entry, journal_seq=9))
        with pytest.raises(PromotionDecisionError, match="not bound"):
            validate_promotion_decision(
                bound(stray), look_result=stray, journal_entry=entry
            )
        counts = result_of(
            EvaluatorOutput("evaluator-output.v1", "test_suite_counts", (), 10, 10)
        )
        assert validate_promotion_decision(
            bound(counts), look_result=counts, journal_entry=entry
        )
        with pytest.raises(
            PromotionDecisionError, match="did not freeze the candidate"
        ):
            validate_promotion_decision(
                bound(counts, candidate_content_digest=digest_of("other")),
                look_result=counts,
                journal_entry=entry,
            )
        other = result_of(_output(slice_estimates=(_estimate(artefact_digest=A2),)))
        with pytest.raises(
            PromotionDecisionError, match="does not report the candidate"
        ):
            validate_promotion_decision(
                bound(other), look_result=other, journal_entry=entry
            )

    def test_the_pre_state_inputs_are_typed(self) -> None:
        """A boolean generation or a malformed digest is refused."""
        decision = _decision(expected_active_generation=1)
        with pytest.raises(PromotionDecisionError, match="must be an integer"):
            check_promotion_pre_state(
                decision,
                active_generation=True,
                production_digest=digest_of("production"),
            )
        with pytest.raises(MalformedDigestError):
            check_promotion_pre_state(
                decision, active_generation=1, production_digest="stale"
            )

    def test_a_non_decision_is_refused(self) -> None:
        """The validator needs a decision."""
        with pytest.raises(PromotionDecisionError):
            validate_promotion_decision("decision")  # type: ignore[arg-type]
