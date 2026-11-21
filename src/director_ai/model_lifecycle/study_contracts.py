# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Study Term Validators

"""Validators of build-2 study terms and of the dynamics evaluation contract.

The dynamics contract fixes how much evidence a physical-dynamics model must
face: at least 50 initial states and 8 action sequences, a conformal coverage
level in ``(0.5, 1)``, an in-distribution false-alarm rate of at most 5 %,
state-region bins and subtle-shift sweeps, at least 100 planning initial
states per seed, a rollout horizon of at least 100 steps for export parity,
and a fitted physics baseline whenever the model claims no unmodelled
physics.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .errors import EvaluationContractError
from .recipe_parts import validate_matched_budget
from .study_records import (
    DynamicsEvaluationContract,
    InferenceComputeEnvelope,
    PreregistrationBuild2Terms,
)

E = EvaluationContractError
BPS_SCALE = 10_000
"""Basis points in a whole."""


def validate_inference_envelope(
    envelope: InferenceComputeEnvelope,
) -> InferenceComputeEnvelope:
    """Check an arm's per-item test-time compute caps.

    Parameters
    ----------
    envelope : InferenceComputeEnvelope
        The caps to check.

    Returns
    -------
    InferenceComputeEnvelope
        The same caps.

    Raises
    ------
    EvaluationContractError
        If a cap is below its minimum or a field is malformed.
    """
    if not isinstance(envelope, InferenceComputeEnvelope):
        raise E("an InferenceComputeEnvelope is required")
    require_choice(envelope.schema, "schema", {"inference-compute-envelope.v1"}, E)
    require_int(
        envelope.max_generated_tokens_per_item,
        "max_generated_tokens_per_item",
        E,
        minimum=1,
    )
    require_int(envelope.max_rollouts_n, "max_rollouts_n", E, minimum=1)
    require_int(
        envelope.max_verifier_calls_per_item,
        "max_verifier_calls_per_item",
        E,
        minimum=0,
    )
    require_int(envelope.max_wall_ms_per_item, "max_wall_ms_per_item", E, minimum=1)
    return envelope


def _named_table(value: object, field: str) -> list[tuple[str, object]]:
    """Return ``(name, entry)`` pairs with distinct, non-empty names."""
    pairs: list[tuple[str, object]] = []
    for pair in require_tuple(value, field, E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E(f"{field} must hold (name, value) pairs")
        pairs.append((require_text(pair[0], f"{field} name", E), pair[1]))
    require_unique([name for name, _ in pairs], f"{field} names", E)
    return pairs


def validate_build2_terms(
    terms: PreregistrationBuild2Terms,
) -> PreregistrationBuild2Terms:
    """Check a build-2 study's terms.

    Parameters
    ----------
    terms : PreregistrationBuild2Terms
        The terms to check.

    Returns
    -------
    PreregistrationBuild2Terms
        The same terms.

    Raises
    ------
    EvaluationContractError
        If the flip-rate ceiling is outside ``[0, 10000]`` basis points, a
        noise floor is negative, the two arm tables name different arms, or a
        field is malformed.
    RecipeContractError
        If an arm's matched budget is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(terms, PreregistrationBuild2Terms):
        raise E("a PreregistrationBuild2Terms is required")
    require_choice(terms.schema, "schema", {"preregistration-build2-terms.v1"}, E)
    require_int(terms.min_training_seeds, "min_training_seeds", E, minimum=1)
    require_choice(
        terms.seed_aggregation_rule,
        "seed_aggregation_rule",
        {"every_seed_must_pass", "median_seed"},
        E,
    )
    ceiling = require_int(
        terms.test_retest_flip_rate_ceiling_bps,
        "test_retest_flip_rate_ceiling_bps",
        E,
        minimum=0,
    )
    if ceiling > BPS_SCALE:
        raise E("test_retest_flip_rate_ceiling_bps must not exceed 10000")
    for _, floor in _named_table(
        terms.non_rate_metric_noise_floors, "non_rate_metric_noise_floors"
    ):
        if require_finite(floor, "noise floor", E) < 0.0:
            raise E("a non-rate noise floor must not be negative")
    budgets = _named_table(terms.arm_matched_budgets, "arm_matched_budgets")
    envelopes = _named_table(terms.arm_inference_envelopes, "arm_inference_envelopes")
    if not budgets:
        raise E("a study needs at least one arm")
    if {arm for arm, _ in budgets} != {arm for arm, _ in envelopes}:
        raise E("every arm needs both a matched budget and an inference envelope")
    for _, budget in terms.arm_matched_budgets:
        validate_matched_budget(budget)
    for _, envelope in terms.arm_inference_envelopes:
        validate_inference_envelope(envelope)
    require_bool(terms.pareto_multi_budget_study, "pareto_multi_budget_study", E)
    require_text(
        terms.predeclared_promotion_eligible_arm_rule,
        "predeclared_promotion_eligible_arm_rule",
        E,
    )
    for name in ("dynamics_eval_contract_digest", "quantum_acceptance_spec_digest"):
        if getattr(terms, name) is not None:
            require_sha256(getattr(terms, name), name)
    return terms


def _check_dynamics_fields(contract: DynamicsEvaluationContract) -> None:
    """Check the dynamics contract's names, kinds, digests and planner sizes."""
    require_choice(contract.schema, "schema", {"dynamics-evaluation-contract.v1"}, E)
    require_text(contract.behaviour_policy_id, "behaviour_policy_id", E)
    require_text(contract.detectability_condition, "detectability_condition", E)
    for split, behaviour in _named_table(
        contract.split_expected_behaviours, "split_expected_behaviours"
    ):
        require_choice(
            behaviour,
            f"behaviour of {split}",
            {"predict_within_tolerance", "defer_to_fallback"},
            E,
        )
    require_choice(
        contract.calibration_score_kind,
        "calibration_score_kind",
        {"per_trajectory_per_horizon", "max_over_horizon_simultaneous"},
        E,
    )
    require_choice(
        contract.interval_proper_score, "interval_proper_score", {"winkler", "crps"}, E
    )
    require_choice(
        contract.shift_detector_score_kind,
        "shift_detector_score_kind",
        {"one_step_residual", "cusum", "ensemble_disagreement"},
        E,
    )
    for name in ("state_region_bins", "subtle_shift_sweeps"):
        items = require_tuple(getattr(contract, name), name, E)
        for item in items:
            require_text(item, name, E)
        require_unique(items, name, E)
    for name in (
        "planner_population",
        "planner_iterations",
        "planning_horizon_steps",
        "episode_length_steps",
    ):
        require_int(getattr(contract, name), name, E, minimum=1)
    for name in (
        "oracle_planner_digest",
        "classical_controller_digest",
        "energy_reference_params_digest",
    ):
        require_sha256(getattr(contract, name), name)
    if contract.fitted_physics_parametric_baseline_digest is not None:
        require_sha256(
            contract.fitted_physics_parametric_baseline_digest,
            "fitted_physics_parametric_baseline_digest",
        )
    require_bool(
        contract.unmodelled_physics_terms_present, "unmodelled_physics_terms_present", E
    )


def validate_dynamics_evaluation_contract(
    contract: DynamicsEvaluationContract,
) -> DynamicsEvaluationContract:
    """Check a dynamics evaluation contract against the evidence floors W-1 to W-7.

    Parameters
    ----------
    contract : DynamicsEvaluationContract
        The contract to check.

    Returns
    -------
    DynamicsEvaluationContract
        The same contract.

    Raises
    ------
    EvaluationContractError
        If an evidence floor is not met, no split is named, a model without
        unmodelled physics names no fitted physics baseline, or a field is
        malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(contract, DynamicsEvaluationContract):
        raise E("a DynamicsEvaluationContract is required")
    _check_dynamics_fields(contract)
    require_int(contract.min_initial_states_m, "min_initial_states_m", E, minimum=50)
    require_int(contract.min_action_sequences_k, "min_action_sequences_k", E, minimum=8)
    coverage = require_finite(
        contract.conformal_coverage_level, "conformal_coverage_level", E
    )
    if not 0.5 < coverage < 1.0:
        raise E("conformal_coverage_level must lie in (0.5, 1)")
    alarm = require_finite(
        contract.max_id_false_alarm_rate, "max_id_false_alarm_rate", E
    )
    if not 0.0 < alarm <= 0.05:
        raise E("max_id_false_alarm_rate must lie in (0, 0.05]")
    unsafe = require_finite(
        contract.max_unsafe_before_deferral_rate, "max_unsafe_before_deferral_rate", E
    )
    if not 0.0 <= unsafe <= 1.0:
        raise E("max_unsafe_before_deferral_rate must lie in [0, 1]")
    if not contract.split_expected_behaviours:
        raise E("a dynamics evaluation names at least one split")
    if not contract.state_region_bins or not contract.subtle_shift_sweeps:
        raise E("a dynamics evaluation needs state-region bins and subtle-shift sweeps")
    require_int(
        contract.min_planning_initial_states_per_seed,
        "min_planning_initial_states_per_seed",
        E,
        minimum=100,
    )
    require_int(
        contract.onnx_parity_rollout_horizon_h,
        "onnx_parity_rollout_horizon_h",
        E,
        minimum=100,
    )
    if (
        contract.unmodelled_physics_terms_present is False
        and contract.fitted_physics_parametric_baseline_digest is None
    ):
        raise E(
            "without unmodelled physics terms, a fitted physics baseline is required"
        )
    return contract
