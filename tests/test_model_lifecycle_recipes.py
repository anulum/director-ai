# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Recipe and RLVR Tests

"""Training recipes, verifiable-reward outcomes and batch masking rates."""

from __future__ import annotations

from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    ArchitectureSpecError,
    CurriculumStageSpec,
    EnvelopeError,
    HardwareCapabilityProfile,
    MalformedDigestError,
    MatchedBudgetAccountingSpec,
    PretrainingCheckpointState,
    RecipeContractError,
    RlvrAlgorithmSpec,
    RlvrContractError,
    SeedQuadruple,
    TrainingRecipeManifest,
    UnattestedVerifierFaultMaskError,
    VerifiableRewardOutcome,
    VerifierFaultAttestation,
    VerifierMaskRateDriftError,
    VerifierMaskRateExceededError,
    canonical_json_bytes,
    from_canonical_json_bytes,
    record_digest,
    validate_curriculum_stage,
    validate_hardware_profile,
    validate_matched_budget,
    validate_pretraining_checkpoint_state,
    validate_recipe_and_curriculum,
    validate_rlvr_batch_mask_rate,
    validate_rlvr_spec,
    validate_seed_quadruple,
    validate_verifiable_reward_outcome,
    validate_verifier_fault_attestation,
    wilson_interval,
)
from tests.model_lifecycle_support import UTC, digest_of
from tests.test_model_lifecycle_architecture import mup
from tests.test_model_lifecycle_architecture import spec as architecture

BASE = digest_of("base")


def seeds(**changes: object) -> SeedQuadruple:
    """Return four distinct valid seeds with ``changes`` applied."""
    return replace(SeedQuadruple("seed-quadruple.v1", 1, 2, 3, 4), **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def hardware(**changes: object) -> HardwareCapabilityProfile:
    """Return a valid consumer-GPU profile with ``changes`` applied."""
    base = HardwareCapabilityProfile(
        "hardware-capability-profile.v1", "rocm_consumer_gpu", "bf16", "x86-64-v3", None
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def stage(index: int = 0, **changes: object) -> CurriculumStageSpec:
    """Return a valid curriculum stage at ``index`` with ``changes`` applied."""
    base = CurriculumStageSpec(
        "curriculum-stage.v1",
        index,
        "stable",
        2048,
        10**9,
        3e-4,
        (("shard-a", 0.1), ("shard-b", 0.2), ("shard-c", 0.7)),
        None,
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def budget(**changes: object) -> MatchedBudgetAccountingSpec:
    """Return a valid matched budget with ``changes`` applied."""
    base = MatchedBudgetAccountingSpec(
        "matched-budget-accounting.v1",
        10**9,
        5000,
        3600.0,
        12,
        0,
        0,
        ("training_tokens", "gradient_steps"),
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def rlvr(**changes: object) -> RlvrAlgorithmSpec:
    """Return a valid group-relative spec with ``changes`` applied."""
    base = RlvrAlgorithmSpec(
        schema="rlvr-algorithm-spec.v1",
        algorithm="grpo_unbiased",
        group_size_g=8,
        min_valid_group_rollouts=4,
        loss_length_normaliser="fixed_max_response_tokens",
        clip_eps_low=0.2,
        clip_eps_high=0.28,
        kl_anchor_beta=0.0,
        reference_checkpoint_digest=BASE,
        drop_zero_variance_groups=True,
        conciseness_bonus_on_pass_only=0.1,
        max_masked_rollout_fraction_per_batch=0.05,
        max_masked_fraction_drift_vs_reference=0.0,
        masked_drift_alpha=0.05,
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def recipe(**changes: object) -> TrainingRecipeManifest:
    """Return a valid from-scratch language recipe with ``changes`` applied."""
    base = TrainingRecipeManifest(
        schema="training-recipe-manifest.v1",
        family="language",
        adaptation_mode="pretrained_from_scratch",
        funding_kind="local_owned_hardware",
        reproducibility_mode="statistical_ci_train",
        seeds=seeds(),
        hardware_profile=hardware(),
        dedup_report_digest=digest_of("dedup"),
        family_contract_digest=digest_of("family"),
        base_checkpoint_digests=(),
        architecture_spec=architecture(),
        curriculum_stages=(stage(0, phase="warmup"), stage(1), stage(2, phase="decay")),
        rlvr_spec=None,
        matched_budget=budget(),
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


RLVR_RECIPE = {
    "adaptation_mode": "rlvr_policy_optimisation",
    "base_checkpoint_digests": (BASE,),
    "architecture_spec": None,
    "rlvr_spec": rlvr(),
}


def attestation(**changes: object) -> VerifierFaultAttestation:
    """Return a valid outer-harness attestation with ``changes`` applied."""
    base = VerifierFaultAttestation(
        "verifier-fault-attestation.v1", "host_io_error", 4242, True, UTC
    )
    return replace(base, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def outcome(
    status: str = "verified_pass", reward: float | None = 1.0, **changes: object
) -> VerifiableRewardOutcome:
    """Return an outcome with ``status`` and ``reward`` and ``changes`` applied."""
    base = VerifiableRewardOutcome(
        "verifiable-reward-outcome.v1",
        "prompt-1",
        0,
        "verified_pass",
        1.0,
        None,
        512,
        900,
    )
    return replace(base, status=status, reward=reward, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


def fault(**changes: object) -> VerifiableRewardOutcome:
    """Return an attested verifier-fault outcome with ``changes`` applied."""
    attested = outcome(
        "verifier_infrastructure_fault",
        None,
        fault_attestation_digest=record_digest(attestation()),
    )
    return replace(attested, **changes)  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields


class TestRecipeParts:
    """Seeds, hardware classes, curriculum stages and matched budgets."""

    @pytest.mark.parametrize(
        "changes",
        [
            {"dropout_seed": 1},
            {"init_seed": 2**63},
            {"init_seed": -1},
            {"dropout_seed": True},
            {"schema": "seeds.v2"},
        ],
        ids=["M-B2-REC-1-duplicate", "range", "negative", "bool", "schema"],
    )
    def test_seed_refusals(self, changes: dict[str, object]) -> None:
        """M-B2-REC-1: four distinct seeds in ``[0, 2**63)``."""
        with pytest.raises(RecipeContractError):
            validate_seed_quadruple(seeds(**changes))

    def test_the_largest_seed_passes(self) -> None:
        """``2**63 - 1`` is a valid seed."""
        validate_seed_quadruple(seeds(init_seed=2**63 - 1))

    @pytest.mark.parametrize("tier", ["cuda_sm61_gpu", "cpu_x86_64_v2_avx"])
    def test_m_b2_rec_2_bf16_on_classes_without_it(self, tier: str) -> None:
        """M-B2-REC-2 (B2-2): bf16 is refused where the class lacks it; fp32 passes."""
        with pytest.raises(RecipeContractError, match="no native bf16"):
            validate_hardware_profile(hardware(target_tier=tier))
        validate_hardware_profile(hardware(target_tier=tier, compute_dtype="fp32"))

    @pytest.mark.parametrize(
        "changes",
        [
            {"target_tier": "ml_box"},
            {"compute_dtype": "fp16"},
            {"host_cpu_isa_ceiling": "x86-64-v4"},
            {"diloco_inner_steps_h": 0},
            {"schema": "hw.v2"},
        ],
    )
    def test_hardware_refusals(self, changes: dict[str, object]) -> None:
        """Each malformed hardware field is refused."""
        with pytest.raises(RecipeContractError):
            validate_hardware_profile(hardware(**changes))

    def test_cloud_bf16_and_diloco_pass(self) -> None:
        """A cloud class takes bf16 and a DiLoCo inner step count."""
        validate_hardware_profile(
            hardware(target_tier="cloud_accelerator", diloco_inner_steps_h=500)
        )

    def test_decimal_mixture_sum(self) -> None:
        """0.7 + 0.1 + 0.1 + 0.1 is exactly one in decimals, though not in binary floats."""
        weights = (("a", 0.7), ("b", 0.1), ("c", 0.1), ("d", 0.1))
        assert 0.7 + 0.1 + 0.1 + 0.1 != 1.0
        validate_curriculum_stage(stage(shard_mixture_weights=weights))
        validate_curriculum_stage(
            stage(shard_mixture_weights=(("only", 1.0),), rope_theta_override=5e5)
        )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"shard_mixture_weights": (("a", 0.5), ("b", 0.499))}, "sum to 0.999"),
            (
                {"shard_mixture_weights": (("a", 0.5), ("b", 0.5), ("c", 0.0))},
                "above zero",
            ),
            ({"shard_mixture_weights": (("a", 0.5), ("a", 0.5))}, "repeat"),
            ({"shard_mixture_weights": (("a", 0.5, 0.5),)}, "pairs"),
            ({"shard_mixture_weights": (("a", 1),)}, "mixture weight"),
            ({"shard_mixture_weights": ()}, "sum to 0"),
            ({"peak_or_target_lr": 0.0}, "peak_or_target_lr"),
            ({"rope_theta_override": 0.0}, "rope_theta_override"),
            ({"sequence_length": 0}, "sequence_length"),
            ({"token_budget": 0}, "token_budget"),
            ({"phase": "cooldown"}, "phase"),
            ({"stage_index": -1}, "stage_index"),
            ({"schema": "stage.v2"}, "schema"),
        ],
        ids=[
            "M-B2-REC-3",
            "zero",
            "repeat",
            "pairs",
            "int",
            "empty",
            "lr",
            "rope",
            "seq",
            "tokens",
            "phase",
            "index",
            "schema",
        ],
    )
    def test_stage_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """M-B2-REC-3 and every other stage rule, by name."""
        with pytest.raises(RecipeContractError, match=fragment):
            validate_curriculum_stage(replace(stage(), **changes))  # type: ignore[arg-type]  # **changes takes any value so refusal tests can set ill-typed fields

    @pytest.mark.parametrize(
        "changes",
        [
            {"training_tokens": -1},
            {"accelerator_seconds": -1.0},
            {"accelerator_seconds": 3600},
            {"held_equal_axes": ("wall_clock",)},
            {"held_equal_axes": ("gradient_steps", "gradient_steps")},
            {"held_equal_axes": ["gradient_steps"]},
            {"schema": "budget.v2"},
        ],
    )
    def test_budget_refusals(self, changes: dict[str, object]) -> None:
        """Each malformed budget field is refused."""
        with pytest.raises(RecipeContractError):
            validate_matched_budget(budget(**changes))

    def test_records_of_other_types_are_refused(self) -> None:
        """Each part validator checks its own record type."""
        for check in (
            validate_seed_quadruple,
            validate_hardware_profile,
            validate_curriculum_stage,
            validate_matched_budget,
            validate_recipe_and_curriculum,
            validate_pretraining_checkpoint_state,
        ):
            with pytest.raises(RecipeContractError):
                check(rlvr())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


class TestRecipe:
    """The recipe's mode, architecture and verifiable-reward rules."""

    @pytest.mark.parametrize(
        "changes",
        [
            {},
            RLVR_RECIPE,
            {
                "adaptation_mode": "peft_safetensors_adapter",
                "base_checkpoint_digests": (BASE,),
                "architecture_spec": None,
                "curriculum_stages": (),
            },
        ],
        ids=["scratch", "rlvr", "adapter"],
    )
    def test_valid_recipes(self, changes: dict[str, object]) -> None:
        """A scratch, an RLVR and an adapter recipe validate and round-trip."""
        item = recipe(**changes)
        assert validate_recipe_and_curriculum(item) == item
        assert (
            from_canonical_json_bytes(
                TrainingRecipeManifest, canonical_json_bytes(item)
            )
            == item
        )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"base_checkpoint_digests": (BASE,)}, "no base checkpoint"),
            ({"architecture_spec": None}, "needs an architecture"),
            ({**RLVR_RECIPE, "base_checkpoint_digests": ()}, "at least one base"),
            ({**RLVR_RECIPE, "rlvr_spec": None}, "RLVR spec belongs"),
            ({"rlvr_spec": rlvr()}, "RLVR spec belongs"),
            ({"family": "dynamics"}, "another family"),
            ({"curriculum_stages": (stage(0), stage(2))}, "0..N-1"),
            ({"curriculum_stages": (stage(1), stage(0))}, "0..N-1"),
            ({**RLVR_RECIPE, "base_checkpoint_digests": (BASE, BASE)}, "repeat"),
            ({"adaptation_mode": "distillation"}, "adaptation_mode"),
            ({"funding_kind": "donated"}, "funding_kind"),
            ({"reproducibility_mode": "best_effort"}, "reproducibility_mode"),
            ({"schema": "recipe.v2"}, "schema"),
        ],
    )
    def test_recipe_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """Each recipe rule is refused by name."""
        with pytest.raises(RecipeContractError, match=fragment):
            validate_recipe_and_curriculum(recipe(**changes))

    @pytest.mark.parametrize(
        ("changes", "error"),
        [
            ({"family": "vision"}, EnvelopeError),
            ({"dedup_report_digest": "x"}, MalformedDigestError),
            ({"family_contract_digest": "x"}, MalformedDigestError),
            ({**RLVR_RECIPE, "base_checkpoint_digests": ("x",)}, MalformedDigestError),
            ({**RLVR_RECIPE, "rlvr_spec": rlvr(group_size_g=2)}, RlvrContractError),
            ({"seeds": seeds(dropout_seed=1)}, RecipeContractError),
            (
                {"hardware_profile": hardware(target_tier="cuda_sm61_gpu")},
                RecipeContractError,
            ),
            ({"matched_budget": budget(gradient_steps=-1)}, RecipeContractError),
            (
                {
                    "funding_kind": "monthly_cloud_credit",
                    "architecture_spec": architecture(
                        mup_spec=mup(proxy_validation_receipt_digest=None)
                    ),
                },
                ArchitectureSpecError,
            ),
        ],
    )
    def test_part_refusals_surface(
        self, changes: dict[str, object], error: type[Exception]
    ) -> None:
        """Every part's refusal surfaces, the architecture's funding rule included."""
        with pytest.raises(error):
            validate_recipe_and_curriculum(recipe(**changes))

    def test_checkpoint_state(self) -> None:
        """A resumption state validates; a negative loss average is allowed."""
        state = PretrainingCheckpointState(
            "pretraining-checkpoint-state.v1",
            record_digest(recipe()),
            1000,
            10**8,
            digest_of("w"),
            digest_of("o"),
            digest_of("shard"),
            4096,
            digest_of("rng"),
            -0.5,
            1.2,
        )
        assert validate_pretraining_checkpoint_state(state) == state
        assert (
            from_canonical_json_bytes(
                PretrainingCheckpointState, canonical_json_bytes(state)
            )
            == state
        )
        for changes, error in (
            ({"grad_norm_ema": -0.1}, RecipeContractError),
            ({"rolling_loss_ema": float("nan")}, RecipeContractError),
            ({"step": -1}, RecipeContractError),
            ({"schema": "state.v2"}, RecipeContractError),
            ({"rng_state_sha256": "x"}, MalformedDigestError),
        ):
            with pytest.raises(error):
                validate_pretraining_checkpoint_state(replace(state, **changes))


class TestRlvrSpec:
    """The algorithm spec's bounds."""

    @pytest.mark.parametrize(
        "changes",
        [
            {"group_size_g": 3},
            {"min_valid_group_rollouts": 1},
            {"min_valid_group_rollouts": 9},
            {"clip_eps_low": 0.0},
            {"clip_eps_high": 0.1},
            {"kl_anchor_beta": -0.1},
            {"conciseness_bonus_on_pass_only": -0.1},
            {"max_masked_rollout_fraction_per_batch": 0.06},
            {"max_masked_rollout_fraction_per_batch": -0.01},
            {"max_masked_fraction_drift_vs_reference": -0.01},
            {"masked_drift_alpha": 0.0},
            {"masked_drift_alpha": 0.2},
            {"algorithm": "ppo"},
            {"loss_length_normaliser": "per_sequence"},
            {"drop_zero_variance_groups": 1},
            {"schema": "rlvr.v2"},
        ],
    )
    def test_spec_refusals(self, changes: dict[str, object]) -> None:
        """Each spec bound is refused."""
        with pytest.raises(RlvrContractError):
            validate_rlvr_spec(rlvr(**changes))

    def test_a_group_below_four_is_refused_on_its_own(self) -> None:
        """A group of three with a consistent minimum is refused by the group bound."""
        with pytest.raises(
            RlvrContractError, match="group_size_g must be an integer >= 4"
        ):
            validate_rlvr_spec(rlvr(group_size_g=3, min_valid_group_rollouts=2))

    def test_spec_bounds_are_inclusive(self) -> None:
        """A group of 4, a minimum equal to it, a 0.05 ceiling and alpha 0.10 are valid."""
        validate_rlvr_spec(
            rlvr(
                group_size_g=4,
                min_valid_group_rollouts=4,
                masked_drift_alpha=0.10,
                clip_eps_high=0.2,
            )
        )
        with pytest.raises(MalformedDigestError):
            validate_rlvr_spec(rlvr(reference_checkpoint_digest="x"))

    @pytest.mark.parametrize(
        "changes",
        [
            {"candidate_caused_excluded": False},
            {"harness_pid": 0},
            {"fault_kind": "oom"},
            {"observed_at_utc": "now"},
            {"schema": "attest.v2"},
        ],
    )
    def test_attestation_refusals(self, changes: dict[str, object]) -> None:
        """An attestation excludes the candidate as the cause; each field is checked."""
        assert validate_verifier_fault_attestation(attestation()) == attestation()
        with pytest.raises(RlvrContractError):
            validate_verifier_fault_attestation(attestation(**changes))

    def test_records_of_other_types_are_refused(self) -> None:
        """The spec and attestation validators check their own record types."""
        with pytest.raises(RlvrContractError):
            validate_rlvr_spec(attestation())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it
        with pytest.raises(RlvrContractError):
            validate_verifier_fault_attestation(rlvr())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it

    def test_records_round_trip(self) -> None:
        """The spec and the attestation decode through the codec."""
        for item in (rlvr(), attestation()):
            assert (
                from_canonical_json_bytes(type(item), canonical_json_bytes(item))
                == item
            )


class TestOutcomes:
    """TR-1: the reward a status may carry."""

    @pytest.mark.parametrize(
        "item",
        [
            outcome(reward=1.0),
            outcome(reward=1.1),
            outcome(reward=0.01),
            outcome("verified_fail", 0.0),
            outcome("candidate_resource_exceeded", 0.0),
            fault(),
        ],
        ids=["pass", "pass-with-bonus", "pass-small", "fail", "resource", "fault"],
    )
    def test_valid_outcomes(self, item: VerifiableRewardOutcome) -> None:
        """Each status with its allowed reward validates."""
        assert validate_verifiable_reward_outcome(item, spec=rlvr()) == item

    @pytest.mark.parametrize(
        "item",
        [
            outcome(reward=0.0),
            outcome(reward=1.2),
            outcome(reward=None),
            outcome("verified_fail", None),
            outcome("candidate_resource_exceeded", 0.5),
            outcome("verified_fail", 0.1),
            outcome(
                "verified_fail",
                0.0,
                fault_attestation_digest=record_digest(attestation()),
            ),
            outcome(rollout_index=-1),
            outcome(prompt_id=""),
            outcome(status="skipped"),
            outcome(schema="outcome.v2"),
        ],
        ids=[
            "pass-zero",
            "pass-over-bonus",
            "pass-none",
            "M-TR-1-fail-none",
            "resource-half",
            "fail-bonus",
            "fail-attested",
            "index",
            "prompt",
            "status",
            "schema",
        ],
    )
    def test_outcome_refusals(self, item: VerifiableRewardOutcome) -> None:
        """M-TR-1: a candidate's failure earns exactly zero, never a mask."""
        with pytest.raises(RlvrContractError):
            validate_verifiable_reward_outcome(item, spec=rlvr())

    @pytest.mark.parametrize(
        "item",
        [fault(fault_attestation_digest=None), fault(reward=0.0)],
        ids=["M-TR-1b-unattested", "fault-with-reward"],
    )
    def test_m_tr_1b_unattested_faults(self, item: VerifiableRewardOutcome) -> None:
        """M-TR-1b: a fault carries no reward and names its attestation."""
        with pytest.raises(UnattestedVerifierFaultMaskError):
            validate_verifiable_reward_outcome(item, spec=rlvr())

    def test_the_named_attestation_must_match(self) -> None:
        """A given attestation must be the one the fault names and must be valid."""
        validate_verifiable_reward_outcome(
            fault(), spec=rlvr(), attestation=attestation()
        )
        with pytest.raises(UnattestedVerifierFaultMaskError, match="not the one named"):
            validate_verifiable_reward_outcome(
                fault(), spec=rlvr(), attestation=attestation(harness_pid=7)
            )
        with pytest.raises(RlvrContractError):
            validate_verifiable_reward_outcome(
                fault(),
                spec=rlvr(),
                attestation=attestation(candidate_caused_excluded=False),
            )
        with pytest.raises(MalformedDigestError):
            validate_verifiable_reward_outcome(
                fault(fault_attestation_digest="x"), spec=rlvr()
            )

    def test_a_malformed_spec_or_record_is_refused(self) -> None:
        """The spec is checked for every outcome, and the outcome type too."""
        with pytest.raises(RlvrContractError):
            validate_verifiable_reward_outcome(
                outcome("verified_fail", 0.0), spec=rlvr(group_size_g=2)
            )
        with pytest.raises(RlvrContractError):
            validate_verifiable_reward_outcome(attestation(), spec=rlvr())  # type: ignore[arg-type]  # an ill-typed argument on purpose: the call must refuse it


def batch(masked: int, total: int) -> list[VerifiableRewardOutcome]:
    """Return ``total`` outcomes of which ``masked`` are attested faults."""
    return [
        fault() if i < masked else outcome("verified_fail", 0.0) for i in range(total)
    ]


class TestBatchMasking:
    """TR-2: the per-batch ceiling and the drift test."""

    def test_the_ceiling_is_exact(self) -> None:
        """One fault in 20 is exactly 5 % and passes; three in 50 is 6 % and halts."""
        bound = validate_rlvr_batch_mask_rate(
            batch(1, 20),
            spec=rlvr(max_masked_fraction_drift_vs_reference=0.05),
            reference_masked_count=0,
            reference_total_count=1000,
        )
        # 0.05 - sqrt((0.05 - 0.0088814)**2 + 0.0038268**2) at z = 1.95996
        assert bound == pytest.approx(0.0087038, abs=1e-6)
        with pytest.raises(VerifierMaskRateExceededError):
            validate_rlvr_batch_mask_rate(
                batch(3, 50),
                spec=rlvr(),
                reference_masked_count=0,
                reference_total_count=1000,
            )

    def test_m_tr_2b_drift_above_the_reference_halts(self) -> None:
        """M-TR-2b: 5 % masking against a near-zero reference exceeds a zero drift."""
        with pytest.raises(VerifierMaskRateDriftError):
            validate_rlvr_batch_mask_rate(
                batch(1, 20),
                spec=rlvr(),
                reference_masked_count=0,
                reference_total_count=1000,
            )
        assert (
            validate_rlvr_batch_mask_rate(
                batch(1, 20),
                spec=rlvr(),
                reference_masked_count=50,
                reference_total_count=1000,
            )
            <= 0.0
        )

    def test_wilson_interval_values(self) -> None:
        """The interval matches the closed form at z = 1.96."""
        low, high = wilson_interval(1, 20, 1.96)
        assert low == pytest.approx(0.00888, abs=1e-4)
        assert high == pytest.approx(0.23613, abs=1e-4)
        assert wilson_interval(0, 1000, 1.96)[0] == pytest.approx(0.0, abs=1e-12)
        assert wilson_interval(1000, 1000, 1.96)[1] == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize(
        ("outcomes", "masked", "total"),
        [
            ([], 0, 10),
            (batch(0, 10), 0, 0),
            (batch(0, 10), 11, 10),
            (batch(0, 10), -1, 10),
            ([outcome(reward=None)], 0, 10),
        ],
        ids=[
            "empty",
            "no-reference",
            "reference-over",
            "reference-negative",
            "bad-outcome",
        ],
    )
    def test_batch_refusals(
        self, outcomes: list[VerifiableRewardOutcome], masked: int, total: int
    ) -> None:
        """An empty batch, a malformed outcome or impossible reference counts are refused."""
        with pytest.raises(RlvrContractError):
            validate_rlvr_batch_mask_rate(
                outcomes,
                spec=rlvr(),
                reference_masked_count=masked,
                reference_total_count=total,
            )
