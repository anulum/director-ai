# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Study and Export Records

"""Records of a build-2 study's terms, dynamics evaluation and export parity.

:class:`PreregistrationBuild2Terms` holds what a preregistration fixes for a
training study: its model family, seeds, noise floors and native-unit
margins, its baseline arm, and the matched budget and test-time compute of
each arm. A dynamics study, and only a dynamics study, names its dynamics
evaluation contract; a variational quantum study, and only one, names its
acceptance spec. Its
record digest is the preregistration's ``search_spaces_and_budgets_digest``,
and it binds the search spaces through its own ``search_spaces_digest``.
:class:`DynamicsEvaluationContract` fixes how a physical dynamics model is
evaluated. :class:`ExportParityReport` ties an exported serving binary to
its parent by per-slice non-inferiority checks. The rules are in
:mod:`.study_contracts` and :mod:`.export_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .envelopes import ActiveModelFamily
from .recipe_records import MatchedBudgetAccountingSpec

WASM_AUDIOWORKLET_MAX_BYTES = 2_097_152
"""Exported AudioWorklet models must stay below this many bytes."""
MAX_FAMILYWISE_ALPHA = 0.01
"""The largest family-wise error rate an export parity report may use."""
INT8_QUANTISATION_MODES: frozenset[str] = frozenset(
    {"dynamic_qint8", "static_qoperator_int8"}
)


@dataclass(frozen=True, slots=True)
class InferenceComputeEnvelope:
    """Per-item test-time token, rollout, verifier, and wall-clock caps."""

    schema: Literal["inference-compute-envelope.v1"]
    max_generated_tokens_per_item: int
    max_rollouts_n: int
    max_verifier_calls_per_item: int
    max_wall_ms_per_item: int


@dataclass(frozen=True, slots=True)
class PreregistrationBuild2Terms:
    """Build-2 study terms with unit-consistent rate and non-rate noise floors."""

    schema: Literal["preregistration-build2-terms.v1"]
    search_spaces_digest: str
    family: ActiveModelFamily
    min_training_seeds: int
    seed_aggregation_rule: Literal["every_seed_must_pass", "median_seed"]
    test_retest_flip_rate_ceiling_bps: int
    non_rate_metric_noise_floors: tuple[tuple[str, float], ...]
    non_rate_metric_margins: tuple[tuple[str, float], ...]
    arm_matched_budgets: tuple[tuple[str, MatchedBudgetAccountingSpec], ...]
    arm_inference_envelopes: tuple[tuple[str, InferenceComputeEnvelope], ...]
    baseline_arm: str
    pareto_multi_budget_study: bool
    predeclared_promotion_eligible_arm_rule: str
    dynamics_eval_contract_digest: str | None
    quantum_acceptance_spec_digest: str | None


@dataclass(frozen=True, slots=True)
class DynamicsEvaluationContract:
    """Physical dynamics evaluation, conformal calibration, and shift gates."""

    schema: Literal["dynamics-evaluation-contract.v1"]
    behaviour_policy_id: str
    min_initial_states_m: int
    min_action_sequences_k: int
    split_expected_behaviours: tuple[
        tuple[str, Literal["predict_within_tolerance", "defer_to_fallback"]],
        ...,
    ]
    conformal_coverage_level: float
    calibration_score_kind: Literal[
        "per_trajectory_per_horizon",
        "max_over_horizon_simultaneous",
    ]
    interval_proper_score: Literal["winkler", "crps"]
    state_region_bins: tuple[str, ...]
    shift_detector_score_kind: Literal[
        "one_step_residual",
        "cusum",
        "ensemble_disagreement",
    ]
    max_id_false_alarm_rate: float
    max_unsafe_before_deferral_rate: float
    detectability_condition: str
    subtle_shift_sweeps: tuple[str, ...]
    planner_population: int
    planner_iterations: int
    planning_horizon_steps: int
    episode_length_steps: int
    min_planning_initial_states_per_seed: int
    oracle_planner_digest: str
    classical_controller_digest: str
    fitted_physics_parametric_baseline_digest: str | None
    unmodelled_physics_terms_present: bool
    energy_reference_params_digest: str
    onnx_parity_rollout_horizon_h: int


@dataclass(frozen=True, slots=True)
class OnnxExportProfileSpec:
    """ONNX export profile covering x86-64-v2 CPU and WASM AudioWorklet targets."""

    schema: Literal["onnx-export-profile.v1"]
    onnx_opset_version: Literal[17, 18]
    quantisation_mode: Literal[
        "fp32",
        "fp16",
        "dynamic_qint8",
        "static_qoperator_int8",
    ]
    target_isa: Literal["x86-64-v2", "wasm32_simd128"]
    int8_reduce_range: bool
    input_tensor_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    wasm_audioworklet_target: bool
    max_exported_bytes: int
    zero_per_quantum_heap_alloc: bool


@dataclass(frozen=True, slots=True)
class ExportParityMetricSlice:
    """Per-slice Holm-adjusted non-inferiority result for an exported binary."""

    slice_id: str
    metric_id: str
    parent_estimate: float
    export_estimate: float
    paired_diff_lcb: float
    allowed_margin: float
    holm_adjusted_alpha: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ExportParityReport:
    """Serving-format parity report tying exported binary digest to parent."""

    schema: Literal["export-parity-report.v1"]
    parent_content_digest: str
    exported_file_sha256: str
    exported_byte_length: int
    export_format: Literal["gguf", "onnx"]
    serving_runtime_id_and_version: str
    onnx_profile: OnnxExportProfileSpec | None
    tokeniser_and_template_parity_exact: bool
    greedy_dev_identity_rate: float
    structured_json_validity_rate: float | None
    gguf_stop_token_and_zero_unused_passed: bool | None
    dynamics_rollout_horizon_h: int | None
    familywise_alpha: float
    slice_checks: tuple[ExportParityMetricSlice, ...]
