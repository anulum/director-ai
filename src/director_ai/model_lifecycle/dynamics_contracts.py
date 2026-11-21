# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Dynamics Evaluation Validator

"""The validator of the dynamics evaluation contract.

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
from .study_contracts import named_table
from .study_records import DynamicsEvaluationContract

E = EvaluationContractError


def _check_dynamics_fields(contract: DynamicsEvaluationContract) -> None:
    """Check the dynamics contract's names, kinds, digests and planner sizes."""
    require_choice(contract.schema, "schema", {"dynamics-evaluation-contract.v1"}, E)
    require_text(contract.behaviour_policy_id, "behaviour_policy_id", E)
    require_text(contract.detectability_condition, "detectability_condition", E)
    for split, behaviour in named_table(
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
