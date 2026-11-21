# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Study Term and Export Tests

"""Build-2 study terms, the dynamics evaluation contract and export parity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from director_ai.model_lifecycle import (
    DynamicsEvaluationContract,
    EvaluationContractError,
    ExportParityError,
    ExportParityMetricSlice,
    ExportParityReport,
    InferenceComputeEnvelope,
    MalformedDigestError,
    OnnxExportProfileSpec,
    PreregistrationBuild2Terms,
    RecipeContractError,
    canonical_json_bytes,
    from_canonical_json_bytes,
    validate_build2_terms,
    validate_dynamics_evaluation_contract,
    validate_export_parity_report,
    validate_inference_envelope,
    validate_onnx_export_profile,
)
from tests.model_lifecycle_support import digest_of
from tests.test_model_lifecycle_recipes import budget


def envelope(**changes: object) -> InferenceComputeEnvelope:
    """Return valid per-item compute caps with ``changes`` applied."""
    base = InferenceComputeEnvelope("inference-compute-envelope.v1", 2048, 8, 8, 60_000)
    return replace(base, **changes)  # type: ignore[arg-type]


def terms(**changes: object) -> PreregistrationBuild2Terms:
    """Return valid two-arm study terms with ``changes`` applied."""
    base = PreregistrationBuild2Terms(
        schema="preregistration-build2-terms.v1",
        min_training_seeds=3,
        seed_aggregation_rule="every_seed_must_pass",
        test_retest_flip_rate_ceiling_bps=50,
        non_rate_metric_noise_floors=(("rollout_rmse_m", 0.002),),
        arm_matched_budgets=(("baseline", budget()), ("candidate", budget())),
        arm_inference_envelopes=(("baseline", envelope()), ("candidate", envelope())),
        pareto_multi_budget_study=False,
        predeclared_promotion_eligible_arm_rule="candidate only",
        dynamics_eval_contract_digest=digest_of("dynamics"),
        quantum_acceptance_spec_digest=None,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def dynamics(**changes: object) -> DynamicsEvaluationContract:
    """Return a dynamics contract exactly at the W-1 to W-7 floors."""
    base = DynamicsEvaluationContract(
        schema="dynamics-evaluation-contract.v1",
        behaviour_policy_id="pid-baseline-v3",
        min_initial_states_m=50,
        min_action_sequences_k=8,
        split_expected_behaviours=(
            ("in_distribution", "predict_within_tolerance"),
            ("unseen_regime", "defer_to_fallback"),
        ),
        conformal_coverage_level=0.9,
        calibration_score_kind="max_over_horizon_simultaneous",
        interval_proper_score="winkler",
        state_region_bins=("low_density", "high_density"),
        shift_detector_score_kind="cusum",
        max_id_false_alarm_rate=0.05,
        max_unsafe_before_deferral_rate=0.0,
        detectability_condition="shift above 3 noise sigma within 20 steps",
        subtle_shift_sweeps=("actuator_gain_drift",),
        planner_population=64,
        planner_iterations=10,
        planning_horizon_steps=50,
        episode_length_steps=500,
        min_planning_initial_states_per_seed=100,
        oracle_planner_digest=digest_of("oracle"),
        classical_controller_digest=digest_of("pid"),
        fitted_physics_parametric_baseline_digest=None,
        unmodelled_physics_terms_present=True,
        energy_reference_params_digest=digest_of("energy"),
        onnx_parity_rollout_horizon_h=100,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def profile(**changes: object) -> OnnxExportProfileSpec:
    """Return a valid x86-64-v2 INT8 ONNX profile with ``changes`` applied."""
    base = OnnxExportProfileSpec(
        "onnx-export-profile.v1",
        17,
        "dynamic_qint8",
        "x86-64-v2",
        True,
        (("input_ids", (1, 512)), ("attention_mask", (1, 512))),
        False,
        50_000_000,
        False,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


WASM = {
    "target_isa": "wasm32_simd128",
    "quantisation_mode": "fp32",
    "wasm_audioworklet_target": True,
    "zero_per_quantum_heap_alloc": True,
    "max_exported_bytes": 2_000_000,
    "input_tensor_shapes": (("frame", (1, 128)),),
}


def slice_check(**changes: object) -> ExportParityMetricSlice:
    """Return a passed non-inferiority slice with ``changes`` applied."""
    base = ExportParityMetricSlice(
        "all", "accuracy", 0.91, 0.905, -0.008, 0.01, 0.005, True
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def report(**changes: object) -> ExportParityReport:
    """Return a valid ONNX parity report with ``changes`` applied."""
    base = ExportParityReport(
        schema="export-parity-report.v1",
        parent_content_digest=digest_of("parent"),
        exported_file_sha256=digest_of("onnx"),
        exported_byte_length=40_000_000,
        export_format="onnx",
        serving_runtime_id_and_version="onnxruntime 1.23.2",
        onnx_profile=profile(),
        tokeniser_and_template_parity_exact=True,
        greedy_dev_identity_rate=0.998,
        structured_json_validity_rate=None,
        gguf_stop_token_and_zero_unused_passed=None,
        dynamics_rollout_horizon_h=None,
        familywise_alpha=0.01,
        slice_checks=(slice_check(), slice_check(slice_id="long_context")),
    )
    return replace(base, **changes)  # type: ignore[arg-type]


GGUF = {
    "export_format": "gguf",
    "onnx_profile": None,
    "gguf_stop_token_and_zero_unused_passed": True,
    "serving_runtime_id_and_version": "llama.cpp 8887a48f",
}


class TestStudyTerms:
    """Build-2 terms and per-item compute caps."""

    def test_valid_terms_round_trip(self) -> None:
        """Valid terms and caps validate and decode through the codec."""
        for item in (terms(), terms(quantum_acceptance_spec_digest=digest_of("q"))):
            assert validate_build2_terms(item) == item
            assert (
                from_canonical_json_bytes(
                    PreregistrationBuild2Terms, canonical_json_bytes(item)
                )
                == item
            )
        caps = envelope()
        assert (
            from_canonical_json_bytes(
                InferenceComputeEnvelope, canonical_json_bytes(caps)
            )
            == caps
        )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"test_retest_flip_rate_ceiling_bps": 10_001}, "10000"),
            ({"test_retest_flip_rate_ceiling_bps": -1}, "flip_rate"),
            ({"non_rate_metric_noise_floors": (("rmse", -0.1),)}, "negative"),
            ({"non_rate_metric_noise_floors": (("rmse", 0.1), ("rmse", 0.2))}, "names"),
            (
                {"arm_inference_envelopes": (("baseline", envelope()),)},
                "both a matched budget",
            ),
            (
                {"arm_matched_budgets": (), "arm_inference_envelopes": ()},
                "at least one arm",
            ),
            ({"arm_matched_budgets": (("x", budget(), 1),)}, "pairs"),
            ({"min_training_seeds": 0}, "min_training_seeds"),
            ({"seed_aggregation_rule": "mean_seed"}, "seed_aggregation_rule"),
            ({"pareto_multi_budget_study": 0}, "pareto"),
            ({"predeclared_promotion_eligible_arm_rule": ""}, "predeclared"),
            ({"schema": "terms.v2"}, "schema"),
        ],
    )
    def test_term_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """Each malformed term is refused by name."""
        with pytest.raises(EvaluationContractError, match=fragment):
            validate_build2_terms(terms(**changes))

    def test_parts_surface_their_own_errors(self) -> None:
        """An arm's budget, caps and the digests are checked by their own rules."""
        with pytest.raises(RecipeContractError):
            validate_build2_terms(
                terms(
                    arm_matched_budgets=(
                        ("baseline", budget(gradient_steps=-1)),
                        ("candidate", budget()),
                    )
                )
            )
        with pytest.raises(EvaluationContractError, match="max_rollouts_n"):
            validate_build2_terms(
                terms(
                    arm_inference_envelopes=(
                        ("baseline", envelope()),
                        ("candidate", envelope(max_rollouts_n=0)),
                    )
                )
            )
        with pytest.raises(MalformedDigestError):
            validate_build2_terms(terms(dynamics_eval_contract_digest="x"))

    @pytest.mark.parametrize(
        "changes",
        [
            {"max_generated_tokens_per_item": 0},
            {"max_verifier_calls_per_item": -1},
            {"max_wall_ms_per_item": 0},
            {"schema": "caps.v2"},
        ],
    )
    def test_envelope_refusals(self, changes: dict[str, object]) -> None:
        """Each cap has its floor; zero verifier calls are allowed."""
        validate_inference_envelope(envelope(max_verifier_calls_per_item=0))
        with pytest.raises(EvaluationContractError):
            validate_inference_envelope(envelope(**changes))

    def test_records_of_other_types_are_refused(self) -> None:
        """Each validator checks its own record type."""
        for check in (
            validate_build2_terms,
            validate_inference_envelope,
            validate_dynamics_evaluation_contract,
        ):
            with pytest.raises(EvaluationContractError):
                check(profile())  # type: ignore[arg-type]
        for check_export in (
            validate_onnx_export_profile,
            validate_export_parity_report,
        ):
            with pytest.raises(ExportParityError):
                check_export(terms())  # type: ignore[arg-type]


class TestDynamicsContract:
    """W-1 to W-7."""

    def test_the_floors_are_inclusive(self) -> None:
        """A contract exactly at every floor validates and round-trips."""
        item = dynamics()
        assert validate_dynamics_evaluation_contract(item) == item
        assert (
            from_canonical_json_bytes(
                DynamicsEvaluationContract, canonical_json_bytes(item)
            )
            == item
        )
        validate_dynamics_evaluation_contract(
            dynamics(
                unmodelled_physics_terms_present=False,
                fitted_physics_parametric_baseline_digest=digest_of("fit"),
            )
        )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"unmodelled_physics_terms_present": False}, "fitted physics baseline"),
            ({"onnx_parity_rollout_horizon_h": 99}, "onnx_parity_rollout_horizon_h"),
            ({"min_initial_states_m": 49}, "min_initial_states_m"),
            ({"min_action_sequences_k": 7}, "min_action_sequences_k"),
            ({"conformal_coverage_level": 0.5}, "conformal_coverage_level"),
            ({"conformal_coverage_level": 1.0}, "conformal_coverage_level"),
            ({"max_id_false_alarm_rate": 0.051}, "max_id_false_alarm_rate"),
            ({"max_id_false_alarm_rate": 0.0}, "max_id_false_alarm_rate"),
            (
                {"max_unsafe_before_deferral_rate": 1.5},
                "max_unsafe_before_deferral_rate",
            ),
            ({"state_region_bins": ()}, "state-region bins"),
            ({"subtle_shift_sweeps": ()}, "state-region bins"),
            (
                {"min_planning_initial_states_per_seed": 99},
                "min_planning_initial_states",
            ),
            ({"state_region_bins": ("a", "a")}, "repeat"),
            ({"split_expected_behaviours": (("id", "guess"),)}, "behaviour of id"),
            ({"split_expected_behaviours": ()}, "at least one split"),
            ({"calibration_score_kind": "pointwise"}, "calibration_score_kind"),
            ({"interval_proper_score": "brier"}, "interval_proper_score"),
            ({"shift_detector_score_kind": "kl"}, "shift_detector_score_kind"),
            ({"planner_population": 0}, "planner_population"),
            ({"behaviour_policy_id": ""}, "behaviour_policy_id"),
            (
                {"unmodelled_physics_terms_present": 1},
                "unmodelled_physics_terms_present",
            ),
            ({"schema": "dynamics.v2"}, "schema"),
        ],
        ids=[
            "M-B2-DYN-1",
            "M-B2-DYN-2",
            "W-1-m",
            "W-1-k",
            "coverage-half",
            "coverage-one",
            "alarm-over",
            "alarm-zero",
            "unsafe",
            "bins",
            "sweeps",
            "planning",
            "bins-repeat",
            "behaviour",
            "no-splits",
            "calibration",
            "score",
            "detector",
            "population",
            "policy",
            "bool",
            "schema",
        ],
    )
    def test_dynamics_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """M-B2-DYN-1, M-B2-DYN-2 and every other floor, by name."""
        with pytest.raises(EvaluationContractError, match=fragment):
            validate_dynamics_evaluation_contract(dynamics(**changes))

    def test_dynamics_digests(self) -> None:
        """Every dynamics digest is 64-hex."""
        for name in (
            "oracle_planner_digest",
            "classical_controller_digest",
            "energy_reference_params_digest",
            "fitted_physics_parametric_baseline_digest",
        ):
            with pytest.raises(MalformedDigestError):
                validate_dynamics_evaluation_contract(dynamics(**{name: "x"}))


class TestExportParity:
    """Row 5: non-inferiority, ONNX profiles and GGUF checks."""

    @pytest.mark.parametrize(
        "changes",
        [
            {},
            GGUF,
            {"onnx_profile": profile(**WASM), "exported_byte_length": 2_000_000},
            {
                "onnx_profile": profile(
                    quantisation_mode="fp32", int8_reduce_range=False
                )
            },
            {"structured_json_validity_rate": 1.0, "dynamics_rollout_horizon_h": 100},
        ],
        ids=["onnx-int8", "gguf", "wasm-at-cap", "fp32-full-range", "optional-rates"],
    )
    def test_valid_reports(self, changes: dict[str, object]) -> None:
        """Each export shape validates and round-trips."""
        item = report(**changes)
        assert validate_export_parity_report(item) == item
        assert (
            from_canonical_json_bytes(ExportParityReport, canonical_json_bytes(item))
            == item
        )

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            (
                {
                    "onnx_profile": profile(
                        **{**WASM, "max_exported_bytes": 2_097_152}
                    ),
                    "exported_byte_length": 2_000_000,
                },
                "below 2 MiB",
            ),
            ({"onnx_profile": profile(int8_reduce_range=False)}, "reduced range"),
            (
                {
                    "onnx_profile": profile(
                        quantisation_mode="static_qoperator_int8",
                        int8_reduce_range=False,
                    )
                },
                "reduced range",
            ),
            (
                {"onnx_profile": profile(input_tensor_shapes=(("input_ids", (1, 0)),))},
                "dimension",
            ),
            ({"onnx_profile": profile(input_tensor_shapes=())}, "static input shapes"),
            (
                {"onnx_profile": profile(input_tensor_shapes=(("input_ids", ()),))},
                "static shape",
            ),
            (
                {
                    "onnx_profile": profile(
                        input_tensor_shapes=(("a", (1,)), ("a", (2,)))
                    )
                },
                "repeat",
            ),
            ({"onnx_profile": profile(input_tensor_shapes=(("a", (1,), 3),))}, "pairs"),
            ({"onnx_profile": profile(onnx_opset_version=16)}, "opset"),
            ({"onnx_profile": profile(onnx_opset_version=True)}, "opset"),
            ({"onnx_profile": profile(max_exported_bytes=0)}, "max_exported_bytes"),
            (
                {
                    "onnx_profile": profile(**{**WASM, "target_isa": "x86-64-v2"}),
                    "exported_byte_length": 2_000_000,
                },
                "WASM SIMD",
            ),
            (
                {
                    "onnx_profile": profile(
                        **{**WASM, "zero_per_quantum_heap_alloc": False}
                    ),
                    "exported_byte_length": 2_000_000,
                },
                "WASM SIMD",
            ),
            (
                {"onnx_profile": profile(**WASM), "exported_byte_length": 2_000_001},
                "fit its profile",
            ),
            ({"onnx_profile": None}, "needs its profile"),
            ({"gguf_stop_token_and_zero_unused_passed": True}, "no GGUF check"),
            ({**GGUF, "gguf_stop_token_and_zero_unused_passed": False}, "stop-token"),
            ({**GGUF, "onnx_profile": profile()}, "stop-token"),
            ({"familywise_alpha": 0.02}, "familywise_alpha"),
            ({"familywise_alpha": 0.0}, "familywise_alpha"),
            ({"slice_checks": (slice_check(passed=False),)}, "not non-inferior"),
            (
                {"slice_checks": (slice_check(paired_diff_lcb=-0.011),)},
                "not non-inferior",
            ),
            ({"slice_checks": ()}, "at least one slice"),
            ({"slice_checks": (slice_check(), slice_check())}, "repeat"),
            (
                {"slice_checks": (slice_check(holm_adjusted_alpha=0.02),)},
                "holm_adjusted_alpha",
            ),
            ({"slice_checks": (slice_check(allowed_margin=-0.01),)}, "allowed_margin"),
            ({"slice_checks": ("all",)}, "ExportParityMetricSlice"),
            ({"tokeniser_and_template_parity_exact": False}, "match the parent"),
            ({"greedy_dev_identity_rate": 1.01}, "greedy_dev_identity_rate"),
            ({"structured_json_validity_rate": -0.1}, "structured_json_validity_rate"),
            ({"dynamics_rollout_horizon_h": 0}, "dynamics_rollout_horizon_h"),
            ({"exported_byte_length": 0}, "exported_byte_length"),
            ({"export_format": "tflite"}, "export_format"),
            ({"serving_runtime_id_and_version": ""}, "serving_runtime"),
            ({"schema": "parity.v2"}, "schema"),
        ],
        ids=[
            "M-B2-EXP-1",
            "M-B2-EXP-2",
            "static-int8",
            "M-B2-EXP-3",
            "no-shapes",
            "dynamic",
            "repeat-input",
            "pairs",
            "opset",
            "opset-bool",
            "cap",
            "wasm-isa",
            "wasm-alloc",
            "over-cap",
            "no-profile",
            "onnx-gguf-flag",
            "gguf-failed",
            "gguf-profile",
            "alpha-high",
            "alpha-zero",
            "slice-failed",
            "slice-lcb",
            "no-slices",
            "slice-repeat",
            "holm",
            "margin",
            "slice-type",
            "tokeniser",
            "greedy",
            "json",
            "horizon",
            "length",
            "format",
            "runtime",
            "schema",
        ],
    )
    def test_report_refusals(self, changes: dict[str, object], fragment: str) -> None:
        """M-B2-EXP-1 to -3 and every other parity rule, by name."""
        with pytest.raises(ExportParityError, match=fragment):
            validate_export_parity_report(report(**changes))

    def test_every_onnx_export_fits_its_cap(self) -> None:
        """An x86-64-v2 export at its cap passes; one byte more is refused."""
        at_cap = report(exported_byte_length=50_000_000)
        assert validate_export_parity_report(at_cap) == at_cap
        with pytest.raises(ExportParityError, match="fit its profile"):
            validate_export_parity_report(report(exported_byte_length=50_000_001))
        with pytest.raises(ExportParityError, match="fit its profile"):
            validate_export_parity_report(report(exported_byte_length=500_000_000))

    def test_report_digests(self) -> None:
        """The parent and export digests are 64-hex."""
        for name in ("parent_content_digest", "exported_file_sha256"):
            with pytest.raises(MalformedDigestError):
                validate_export_parity_report(report(**{name: "x"}))
        assert (
            from_canonical_json_bytes(
                OnnxExportProfileSpec, canonical_json_bytes(profile())
            )
            == profile()
        )
