# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Promotion-or-Decline Gate Tests

"""One promotion or one complete decline, with E-4, E-5, E-8, M-J4-12 and Q-9."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from director_ai.model_lifecycle import (
    EvaluatorOutput,
    ExportParityError,
    FamilyWitnesses,
    MalformedDigestError,
    ModelArtefactEnvelope,
    PromotionDecisionError,
    PromotionDecisionRecord,
    PromotionDependencyPins,
    PromotionEvidence,
    SealedLookResult,
    ServingMemoryBudget,
    SliceAggregateEstimate,
    StudyDeclineRecord,
    UnpinnedPromotionDependencyError,
    canonical_json_bytes,
    evidence_arm_digests,
    from_canonical_json_bytes,
    record_digest,
    spans_noise_floors,
    validate_promotion_evidence,
    validate_promotion_or_decline,
)
from tests.model_lifecycle_support import (
    UTC,
    digest_of,
    edge,
    journal_entry,
    opening_record,
    pool,
    preregistration,
)
from tests.model_lifecycle_support import envelope as model_envelope
from tests.test_model_lifecycle_quantum import barren, parity, receipt, spec
from tests.test_model_lifecycle_recipes import budget
from tests.test_model_lifecycle_study_terms import (
    dynamics,
    envelope,
    report,
    slice_check,
    terms,
)
from tests.test_model_lifecycle_tier_pins import ROLES, journal

A_BASE, A_CAND = digest_of("arm-baseline"), digest_of("arm-candidate")
LINEAGE, FAMILY, WITNESS = (
    digest_of("lineage"),
    digest_of("family"),
    digest_of("witness"),
)
ARMS = (("baseline", A_BASE), ("candidate", A_CAND))
BUDGET = ServingMemoryBudget("serving-memory-budget.v1", "host_ram", 6, 3, 1, 10)


def estimate(artefact: str, slice_id: str) -> SliceAggregateEstimate:
    """Return one accuracy estimate of ``artefact`` on ``slice_id``."""
    return SliceAggregateEstimate(artefact, "accuracy", slice_id, 0.9, 0.88, 0.92, 40)


SLICES = EvaluatorOutput(
    "evaluator-output.v1",
    "slice_estimates",
    (
        estimate(A_CAND, "all"),
        estimate(A_CAND, "long_context"),
        estimate(A_BASE, "all"),
        estimate(A_BASE, "long_context"),
    ),
    None,
    None,
)


FAMILY_SCHEMAS = {
    "language": "language-family-contract.v1",
    "dynamics": "dynamics-family-contract.v1",
    "spiking": "sc-neurocore.spiking-artefact-contract.v1",
    "quantum_variational": "scpn-qc.variational-artefact-contract.v1",
}
QSPEC = spec()
NO_WITNESSES: dict[str, Any] = dict.fromkeys(
    (
        "spiking_parity_report",
        "barren_plateau_receipt",
        "acceptance_spec",
        "candidate_receipt",
        "baseline_receipt",
    )
)


def candidate_envelope(kind: str) -> ModelArtefactEnvelope:
    """Return the candidate's envelope for a ``kind`` study."""
    return model_envelope(
        family=kind,
        content_digest=A_CAND,
        lineage_edges=(edge(digest_of("base"), A_CAND),),
        family_contract_schema=FAMILY_SCHEMAS[kind],
        family_contract_digest=FAMILY,
    )


def family_setup(
    kind: str,
) -> tuple[dict[str, object], dict[str, Any], tuple[str, ...]]:
    """Return a ``kind`` study's term changes, witness fields and companion contracts."""
    if kind == "dynamics":
        return (
            {
                "family": "dynamics",
                "dynamics_eval_contract_digest": record_digest(dynamics()),
            },
            NO_WITNESSES,
            (),
        )
    base: dict[str, object] = {"family": kind, "dynamics_eval_contract_digest": None}
    if kind == "spiking":
        report_witness = {
            **NO_WITNESSES,
            "spiking_parity_report": parity(checkpoint_content_digest=A_CAND),
        }
        return base, report_witness, (digest_of("numeric"),)
    if kind == "quantum_variational":
        quantum = {
            **NO_WITNESSES,
            "acceptance_spec": QSPEC,
            "candidate_receipt": receipt("finite_shot_simulator", "arm-candidate"),
            "baseline_receipt": receipt("finite_shot_simulator", "arm-baseline"),
            "barren_plateau_receipt": barren(
                companion_digest=FAMILY, split_pool_digest=record_digest(pool())
            ),
        }
        return (
            {**base, "quantum_acceptance_spec_digest": record_digest(QSPEC)},
            quantum,
            (),
        )
    return base, NO_WITNESSES, ()


def inputs(
    *,
    promote: bool = True,
    terms_changes: dict[str, object] | None = None,
    plan_changes: dict[str, object] | None = None,
    entry_changes: dict[str, Any] | None = None,
    report_changes: dict[str, object] | None = None,
    output: EvaluatorOutput = SLICES,
    frozen: tuple[str, ...] = (A_BASE, A_CAND),
    kind: str = "dynamics",
    witness_changes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete, admissible promotion or decline of a ``kind`` study."""
    family_terms, witness_fields, secondaries = family_setup(kind)
    study = terms(**{**family_terms, **(terms_changes or {})})
    plan = preregistration(
        **{
            "declared_arms": ("baseline", "candidate"),
            "frozen_look_candidate_digests": (A_BASE, A_CAND),
            "non_inferiority_margins_bps": (("accuracy", 150),),
            "search_spaces_and_budgets_digest": record_digest(study),
            **(plan_changes or {}),
        }
    )
    entry = journal_entry(
        LINEAGE,
        frozen,
        **{"preregistration_digest": record_digest(plan), **(entry_changes or {})},
    )
    result = SealedLookResult(
        "sealed-look-result.v1",
        record_digest(entry),
        LINEAGE,
        entry.split_digest,
        output,
    )
    export = report(
        **{
            "parent_content_digest": A_CAND,
            "dynamics_rollout_horizon_h": 100,
            **(report_changes or {}),
        }
    )
    witnesses = replace(
        FamilyWitnesses(
            "promotion-family-witnesses.v1", candidate_envelope(kind), **witness_fields
        ),
        **(witness_changes or {}),
    )
    extra = [record_digest(witnesses.candidate_envelope)]
    extra += [
        record_digest(getattr(witnesses, name))
        for name in witness_fields
        if getattr(witnesses, name) is not None
    ]
    if witnesses.acceptance_spec is not None:
        extra.append(witnesses.acceptance_spec.classical_baseline_digest)
    pins = PromotionDependencyPins(
        "promotion-dependency-pins.v1",
        "slot-a",
        FAMILY,
        secondaries,
        (record_digest(export), WITNESS, *extra),
    )
    pinned = (FAMILY, *secondaries, *pins.witness_and_parity_digests)
    tier = journal(*(("tier_pinned", digest, "slot-a") for digest in pinned))
    decision = PromotionDecisionRecord(
        "model-promotion-decision.v1",
        digest_of("intent"),
        "slot-a",
        0,
        "0" * 64,
        A_CAND,
        record_digest(plan),
        record_digest(result),
        digest_of("rigor"),
        digest_of("control"),
        digest_of("blob-c"),
        digest_of("blob-r"),
        UTC,
    )
    decline = StudyDeclineRecord(
        "model-study-decline.v1",
        plan.study_id,
        record_digest(plan),
        record_digest(result),
        (("candidate", ("gain interval below margin",)),),
        True,
        True,
        UTC,
    )
    return {
        "decision": decision if promote else None,
        "decline": None if promote else decline,
        "evidence": PromotionEvidence(
            "promotion-evidence.v1", "candidate" if promote else None, ARMS
        ),
        "terms": study,
        "preregistration": plan,
        "look_result": result,
        "journal_entry": entry,
        "opening_record": opening_record(entry),
        "memory_budget": BUDGET,
        "pins": pins,
        "export_reports": (export,),
        "dynamics_contract": dynamics() if kind == "dynamics" else None,
        "family_witnesses": witnesses,
        "tier_journal": tier,
        "tier_seq_at_decision": len(tier),
        "roles": ROLES,
    }


def refused(
    kwargs: dict[str, Any],
    fragment: str,
    error: type[Exception] = PromotionDecisionError,
) -> None:
    """Assert that the gate refuses ``kwargs`` with ``fragment``."""
    with pytest.raises(error, match=fragment):
        validate_promotion_or_decline(**kwargs)


class TestAdmitted:
    """Complete promotions and declines are admitted."""

    def test_a_complete_promotion_is_admitted(self) -> None:
        """Every gate passes exactly at its floors: 3 x 50 = 150 bps, 0.006 = 3 x 0.002."""
        kwargs = inputs()
        assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]

    def test_a_complete_decline_is_admitted(self) -> None:
        """A decline naming every arm but the baseline is admitted."""
        kwargs = inputs(promote=False)
        assert validate_promotion_or_decline(**kwargs) == kwargs["decline"]

    def test_a_study_without_dynamics_needs_no_horizon(self) -> None:
        """Without a dynamics contract, no export horizon is required."""
        kwargs = inputs(
            kind="language", report_changes={"dynamics_rollout_horizon_h": None}
        )
        assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]

    def test_a_test_suite_look_has_no_slices_to_cover(self) -> None:
        """A test-suite look's coverage is vacuous."""
        counts = EvaluatorOutput("evaluator-output.v1", "test_suite_counts", (), 12, 12)
        kwargs = inputs(
            output=counts, report_changes={"slice_checks": (slice_check(),)}
        )
        assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]

    def test_records_round_trip(self) -> None:
        """The evidence record decodes through the codec."""
        item = inputs()["evidence"]
        assert (
            from_canonical_json_bytes(PromotionEvidence, canonical_json_bytes(item))
            == item
        )


class TestExactlyOneOutcome:
    """E-8: a promotion or a complete decline, never both or neither."""

    @pytest.mark.parametrize("both", [True, False])
    def test_both_or_neither_is_refused(self, both: bool) -> None:
        """Exactly one of a decision and a decline is given."""
        promotion, decline = inputs(), inputs(promote=False)
        kwargs = {
            **promotion,
            "decline": decline["decline"] if both else None,
            "decision": promotion["decision"] if both else None,
        }
        refused(kwargs, "exactly one")

    def test_a_decline_promotes_no_arm(self) -> None:
        """A decline's evidence names no promoted arm."""
        kwargs = inputs(promote=False)
        refused({**kwargs, "evidence": inputs()["evidence"]}, "promotes no arm")

    def test_a_promotion_names_its_arm(self) -> None:
        """A promotion's evidence names the promoted arm."""
        refused(
            {**inputs(), "evidence": inputs(promote=False)["evidence"]},
            "names its promoted arm",
        )

    def test_a_decline_leaving_an_arm_out_is_refused(self) -> None:
        """M-B2-DECLINE-1 and completeness: every arm but the baseline is named."""
        kwargs = inputs(
            promote=False,
            plan_changes={
                "declared_arms": ("baseline", "candidate", "wide"),
                "frozen_look_candidate_digests": (A_BASE, A_CAND, LINEAGE),
            },
            terms_changes={
                "arm_matched_budgets": (
                    ("baseline", budget()),
                    ("candidate", budget()),
                    ("wide", budget()),
                ),
                "arm_inference_envelopes": (
                    ("baseline", envelope()),
                    ("candidate", envelope()),
                    ("wide", envelope()),
                ),
            },
        )
        evidence = PromotionEvidence(
            "promotion-evidence.v1", None, (*ARMS, ("wide", LINEAGE))
        )
        refused({**kwargs, "evidence": evidence}, "every declared arm but the baseline")

    def test_a_decline_naming_the_baseline_is_refused(self) -> None:
        """The baseline is not a failed candidate."""
        kwargs = inputs(promote=False)
        both = replace(
            kwargs["decline"],
            failed_arm_reasons=(("candidate", ("x",)), ("baseline", ("y",))),
        )
        refused({**kwargs, "decline": both}, "every declared arm but the baseline")

    def test_a_decline_cannot_rename_the_baseline(self) -> None:
        """R-1 P1: declining only the terms' baseline leaves the candidate undeclined."""
        kwargs = inputs(promote=False)
        buried = replace(kwargs["decline"], failed_arm_reasons=(("baseline", ("x",)),))
        refused({**kwargs, "decline": buried}, "every declared arm but the baseline")

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"study_id": "study-2"}, "another study or preregistration"),
            (
                {"preregistration_digest": digest_of("other")},
                "another study or preregistration",
            ),
            ({"sealed_look_result_digest": digest_of("other")}, "another look result"),
            ({"failed_arm_reasons": ()}, "every failed arm"),
        ],
    )
    def test_decline_binding_refusals(
        self, changes: dict[str, object], fragment: str
    ) -> None:
        """A decline binds to the study, its preregistration and its look."""
        kwargs = inputs(promote=False)
        refused({**kwargs, "decline": replace(kwargs["decline"], **changes)}, fragment)


class TestStudyBinding:
    """The terms, the look and the decision bind to the preregistration."""

    def test_terms_must_be_the_preregistrations(self) -> None:
        """B-1: the terms' record digest is search_spaces_and_budgets_digest."""
        refused(
            inputs(plan_changes={"search_spaces_and_budgets_digest": digest_of("x")}),
            "search_spaces_and_budgets",
        )

    def test_the_terms_arms_are_the_declared_arms(self) -> None:
        """The terms and the preregistration name the same arms."""
        refused(
            inputs(plan_changes={"declared_arms": ("baseline", "other")}),
            "the terms' arms are not",
        )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"study_id": "study-2"}, "another study"),
            ({"status": "infrastructure_fault"}, "not bound"),
            ({"preregistration_digest": digest_of("other")}, "another preregistration"),
        ],
    )
    def test_look_binding_refusals(
        self, changes: dict[str, Any], fragment: str
    ) -> None:
        """The look is the study's, released, under this preregistration."""
        refused(inputs(entry_changes=changes), fragment)

    def test_a_look_of_another_programme_is_refused(self) -> None:
        """The look belongs to the configured programme."""
        kwargs = inputs()
        refused(
            {**kwargs, "roles": replace(kwargs["roles"], programme="pilot-b")},
            "another programme",
        )

    def test_the_decision_names_the_preregistration(self) -> None:
        """The decision's preregistration digest is the study's."""
        kwargs = inputs()
        refused(
            {
                **kwargs,
                "decision": replace(
                    kwargs["decision"], preregistration_digest=digest_of("x")
                ),
            },
            "another preregistration",
        )

    def test_the_promoted_arm_is_the_candidate(self) -> None:
        """B-2: the promoted arm's artefact is the decision's candidate."""
        swapped = PromotionEvidence(
            "promotion-evidence.v1",
            "candidate",
            (("baseline", A_CAND), ("candidate", A_BASE)),
        )
        refused({**inputs(), "evidence": swapped}, "not the decision's candidate")

    def test_the_baseline_is_never_promoted(self) -> None:
        """R-1: the terms' baseline cannot be the promoted arm."""
        promoted_baseline = PromotionEvidence("promotion-evidence.v1", "baseline", ARMS)
        refused({**inputs(), "evidence": promoted_baseline}, "is the study's baseline")

    def test_s4_the_baseline_is_measured_in_the_same_look(self) -> None:
        """A look that skipped the baseline on a slice is refused for promotion."""
        partial = EvaluatorOutput(
            "evaluator-output.v1",
            "slice_estimates",
            (
                estimate(A_CAND, "all"),
                estimate(A_CAND, "long_context"),
                estimate(A_BASE, "all"),
            ),
            None,
            None,
        )
        refused(inputs(output=partial), "did not measure the baseline")


class TestEvidence:
    """B-2: the arms-to-artefacts join, a named trust."""

    @pytest.mark.parametrize(
        ("evidence", "fragment"),
        [
            (("candidate", ARMS[:1]), "exactly the declared arms"),
            (("candidate", (*ARMS[:1], ("candidate", WITNESS))), "frozen candidates"),
            (("other", ARMS), "promoted arm is not a declared arm"),
        ],
    )
    def test_join_refusals(self, evidence: tuple[Any, ...], fragment: str) -> None:
        """The join names exactly the declared arms, at frozen artefacts."""
        with pytest.raises(PromotionDecisionError, match=fragment):
            evidence_arm_digests(
                PromotionEvidence("promotion-evidence.v1", *evidence),
                inputs()["preregistration"],
            )

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"promoted_arm": ""}, PromotionDecisionError),
            ({"schema": "evidence.v2"}, PromotionDecisionError),
            (
                {"arm_artefact_digests": (("baseline", A_BASE), ("baseline", A_CAND))},
                PromotionDecisionError,
            ),
            (
                {"arm_artefact_digests": (("baseline", A_BASE), ("candidate", A_BASE))},
                PromotionDecisionError,
            ),
            ({"arm_artefact_digests": (("baseline",),)}, PromotionDecisionError),
            ({"arm_artefact_digests": (("baseline", "x"),)}, MalformedDigestError),
        ],
    )
    def test_shape_refusals(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Arms and digests are distinct, and every field is checked."""
        with pytest.raises(error):
            validate_promotion_evidence(replace(inputs()["evidence"], **changes))
        with pytest.raises(PromotionDecisionError):
            validate_promotion_evidence(BUDGET)  # type: ignore[arg-type]


class TestStudyGates:
    """E-4, E-5 and M-J4-12 for the promoted arm's study."""

    @pytest.mark.parametrize(
        ("terms_changes", "plan_changes", "fragment"),
        [
            ({"min_training_seeds": 2}, {}, "at least 3 training seeds"),
            (
                {},
                {"non_inferiority_margins_bps": (("accuracy", 149),)},
                "3 flip-rate ceilings",
            ),
            (
                {},
                {
                    "non_inferiority_margins_bps": (
                        ("accuracy", 150),
                        ("rollout_rmse_m", 600),
                    )
                },
                "basis-point margin",
            ),
            (
                {"non_rate_metric_margins": (("rollout_rmse_m", 0.0059),)},
                {},
                "3 noise floors",
            ),
        ],
        ids=["E-4-seeds", "E-4-rate", "E-4-mixing", "M-B2-NOISE-1"],
    )
    def test_e4_refusals(
        self,
        terms_changes: dict[str, object],
        plan_changes: dict[str, object],
        fragment: str,
    ) -> None:
        """Three seeds, and every margin spans three floors or ceilings."""
        refused(
            inputs(terms_changes=terms_changes, plan_changes=plan_changes), fragment
        )

    def test_noise_floors_compare_declared_decimals(self) -> None:
        """0.3 spans three floors of 0.1, though 3 * 0.1 > 0.3 in binary floats."""
        assert spans_noise_floors(0.3, 0.1) and not spans_noise_floors(0.29999, 0.1)
        assert spans_noise_floors(0.0, 0.0)

    @pytest.mark.parametrize(
        "arms",
        [
            (
                ("baseline", budget()),
                ("candidate", budget(held_equal_axes=("training_tokens",))),
            ),
            (("baseline", budget()), ("candidate", budget(gradient_steps=5001))),
        ],
        ids=["different-axes", "different-values"],
    )
    def test_e5_refusals(self, arms: tuple[tuple[str, object], ...]) -> None:
        """Every arm holds the same axes equal, at equal declared values."""
        refused(inputs(terms_changes={"arm_matched_budgets": arms}), "E-5")

    def test_e5_allows_unheld_axes_to_differ(self) -> None:
        """An axis no arm holds equal may differ."""
        arms = (
            ("baseline", budget()),
            ("candidate", budget(accelerator_seconds=7200.0)),
        )
        kwargs = inputs(terms_changes={"arm_matched_budgets": arms})
        assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]

    @pytest.mark.parametrize(
        "cap",
        [
            "max_generated_tokens_per_item",
            "max_rollouts_n",
            "max_verifier_calls_per_item",
            "max_wall_ms_per_item",
        ],
    )
    def test_m_j4_12_refuses_more_test_time_compute(self, cap: str) -> None:
        """Unless Pareto, the promoted arm's caps stay within the baseline's."""
        bigger = envelope(**{cap: getattr(envelope(), cap) + 1})
        changes: dict[str, object] = {
            "arm_inference_envelopes": (("baseline", envelope()), ("candidate", bigger))
        }
        refused(inputs(terms_changes=changes), f"M-J4-12: the promoted arm's {cap}")
        kwargs = inputs(terms_changes={**changes, "pareto_multi_budget_study": True})
        assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]


class TestFrozenBaselineAndMatchedBudget:
    """R-1 and S-3: the terms fix the comparator and a training-compute axis."""

    def three_arms(
        self, baseline: str, output: EvaluatorOutput = SLICES
    ) -> dict[str, Any]:
        """Return a promotion over baseline, candidate and wide arms (+0, +5, +10 rollouts)."""
        caps = {
            "baseline": envelope(),
            "candidate": envelope(max_rollouts_n=13),
            "wide": envelope(max_rollouts_n=18),
        }
        kwargs = inputs(
            plan_changes={
                "declared_arms": ("baseline", "candidate", "wide"),
                "frozen_look_candidate_digests": (A_BASE, A_CAND, LINEAGE),
            },
            terms_changes={
                "arm_matched_budgets": tuple((arm, budget()) for arm in caps),
                "arm_inference_envelopes": tuple(caps.items()),
                "baseline_arm": baseline,
            },
            output=output,
            frozen=(A_BASE, A_CAND, LINEAGE),
        )
        evidence = PromotionEvidence(
            "promotion-evidence.v1", "candidate", (*ARMS, ("wide", LINEAGE))
        )
        return {**kwargs, "evidence": evidence}

    def test_r1_p2_m_j4_12_reads_the_terms_baseline(self) -> None:
        """Against the frozen baseline the wider candidate is refused; no caller can swap it."""
        refused(
            self.three_arms("baseline"), "M-J4-12: the promoted arm's max_rollouts_n"
        )

    def test_a_preregistered_wide_baseline_is_honoured(self) -> None:
        """A baseline the terms froze as the wide arm admits the narrower candidate."""
        output = EvaluatorOutput(
            "evaluator-output.v1",
            "slice_estimates",
            (
                estimate(A_CAND, "all"),
                estimate(A_CAND, "long_context"),
                estimate(LINEAGE, "all"),
                estimate(LINEAGE, "long_context"),
            ),
            None,
            None,
        )
        kwargs = self.three_arms("wide", output)
        assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]

    @pytest.mark.parametrize(
        ("held", "pareto", "admitted"),
        [
            ((), False, False),
            ((), True, True),
            (("human_review_seconds",), False, False),
            (("accelerator_seconds",), False, True),
        ],
        ids=["empty-matched", "empty-pareto", "review-only", "accelerator"],
    )
    def test_s3_a_matched_study_holds_training_compute(
        self, held: tuple[str, ...], pareto: bool, admitted: bool
    ) -> None:
        """Unless Pareto, at least one training-compute axis is held equal."""
        arms = (
            ("baseline", budget(held_equal_axes=held)),
            ("candidate", budget(held_equal_axes=held)),
        )
        kwargs = inputs(
            terms_changes={
                "arm_matched_budgets": arms,
                "pareto_multi_budget_study": pareto,
            }
        )
        if admitted:
            assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]
        else:
            refused(kwargs, "training-compute axis")


class TestServingAndPins:
    """The memory budget and Q-9."""

    @pytest.mark.parametrize("missing", ["memory_budget", "pins"])
    def test_a_promotion_needs_its_budget_and_pins(self, missing: str) -> None:
        """Both are required for a promotion."""
        refused(
            {**inputs(), missing: None}, "serving memory budget and dependency pins"
        )

    def test_the_memory_budget_must_fit(self) -> None:
        """Weights, cache and overhead fit the ceiling."""
        refused(
            {**inputs(), "memory_budget": replace(BUDGET, runtime_overhead_bytes=2)},
            "exceed",
        )

    def test_an_unpinned_witness_is_refused(self) -> None:
        """M-B2-PIN-1 through the gate: a witness pinned after the decision."""
        kwargs = inputs()
        refused(
            {**kwargs, "tier_seq_at_decision": 2},
            "not pinned",
            UnpinnedPromotionDependencyError,
        )


class TestExports:
    """B-4 and the cross-record rules on the candidate's export reports."""

    def test_at_least_one_report(self) -> None:
        """A promotion needs a serving export."""
        refused({**inputs(), "export_reports": ()}, "at least one export")

    def test_a_report_of_another_parent_is_refused(self) -> None:
        """Every report is the candidate's."""
        refused(
            inputs(report_changes={"parent_content_digest": A_BASE}), "another parent"
        )

    def test_an_unpinned_report_is_refused(self) -> None:
        """Every report is a pinned witness."""
        kwargs = inputs()
        other = replace(kwargs["export_reports"][0], exported_byte_length=39_999_999)
        refused({**kwargs, "export_reports": (other,)}, "pinned witnesses")

    def test_a_report_missing_a_slice_is_refused(self) -> None:
        """Every look slice of the candidate is covered."""
        refused(
            inputs(report_changes={"slice_checks": (slice_check(),)}), "does not cover"
        )

    @pytest.mark.parametrize(("margin", "ok"), [(0.006, True), (0.0059, False)])
    def test_non_rate_export_margins_span_three_floors(
        self, margin: float, ok: bool
    ) -> None:
        """A non-rate slice's allowed margin spans three noise floors."""
        check = slice_check(
            metric_id="rollout_rmse_m", allowed_margin=margin, paired_diff_lcb=-0.001
        )
        kwargs = inputs(
            report_changes={
                "slice_checks": (
                    slice_check(),
                    slice_check(slice_id="long_context"),
                    check,
                )
            }
        )
        if ok:
            assert validate_promotion_or_decline(**kwargs) == kwargs["decision"]
        else:
            refused(kwargs, "allowed margin spans < 3 noise floors")

    @pytest.mark.parametrize("horizon", [99, None])
    def test_a_short_dynamics_horizon_is_refused(self, horizon: int | None) -> None:
        """A dynamics export's horizon reaches the contract's."""
        refused(
            inputs(report_changes={"dynamics_rollout_horizon_h": horizon}),
            "rollout horizon",
        )

    def test_the_dynamics_contract_is_the_terms(self) -> None:
        """Required when named, refused when not, and bound by digest."""
        kwargs = inputs()
        refused({**kwargs, "dynamics_contract": None}, "is required")
        refused(
            {**kwargs, "dynamics_contract": dynamics(min_initial_states_m=60)},
            "not the terms'",
        )
        stray = inputs(kind="language")
        refused({**stray, "dynamics_contract": dynamics()}, "name no dynamics")

    def test_a_malformed_report_is_refused(self) -> None:
        """Each report is validated."""
        refused(
            inputs(report_changes={"familywise_alpha": 0.02}),
            "familywise_alpha",
            ExportParityError,
        )
