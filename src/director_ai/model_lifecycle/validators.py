# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Validator Facade

"""Re-export of every validator, digest function and check of the kernel.

Consumers import from :mod:`director_ai.model_lifecycle`; this facade only
groups the functions for the package itself.
"""

from __future__ import annotations

from .architecture_contracts import (
    check_transformer_architecture_shape,
    validate_transformer_architecture_spec,
)
from .base_checkpoints import (
    validate_base_checkpoint_record,
)
from .cost_and_promotion_contracts import (
    check_promotion_pre_state,
    validate_promotion_decision,
)
from .custody_records import (
    is_sealed_kind,
    validate_custody_ingest_record,
)
from .dataset_contracts import (
    validate_dedup_and_distillation,
    validate_dedup_report,
    validate_distillation_manifest,
)
from .deployment_records import (
    validate_post_swap_evidence,
    validate_post_swap_verification,
)
from .digests import (
    canonical_json_bytes,
    decode_record,
    record_digest,
    strict_json_loads,
)
from .dynamics_contracts import (
    validate_dynamics_evaluation_contract,
)
from .encoding_contracts import (
    validate_embedding_space_identity,
    validate_normaliser_envelope,
    validate_tokeniser_and_normaliser_disjointness,
    validate_tokeniser_envelope,
)
from .envelopes import (
    require_active_family,
    validate_envelope,
    validate_lineage_edge,
)
from .evaluation_contracts import (
    validate_preregistration,
    validate_sealed_split_pool,
)
from .evaluator_outputs import (
    validate_evaluator_output,
    validate_look_result,
    validate_slice_estimate,
)
from .export_contracts import (
    validate_export_parity_report,
    validate_onnx_export_profile,
)
from .family_companions import (
    validate_dynamics_family_contract,
    validate_language_family_contract,
)
from .family_contracts import (
    compute_family_model_identity_digest,
    compute_family_tensor_set_identity,
    validate_family_tensor_layout,
)
from .family_registry import (
    family_rule,
)
from .family_witnesses import (
    require_spiking_parity_passed,
    validate_barren_plateau_receipt,
    validate_spiking_lowering_parity,
)
from .float_scan import (
    any_lane_matches,
    scan_float_chunk,
)
from .graders import (
    validate_grader,
)
from .job_contracts import (
    validate_job_cancel,
    validate_job_status,
    validate_job_submission,
    validate_sandbox_confinement,
)
from .journal_entries import (
    check_journal_order,
    journal_entry_for,
    select_study_entries,
)
from .licences import (
    require_output_training_permitted,
    validate_compute_budget,
    validate_licence_record,
)
from .lineage import (
    derive_proposer_lineage,
    exempt_base_digests,
)
from .look_admission import (
    admit_sealed_look,
)
from .look_budget import (
    check_look_budget,
)
from .merge_contracts import (
    merge_identity,
    validate_adapter_merge,
    validate_merge_result,
)
from .power import (
    required_paired_sample_size,
)
from .preregistration_approvals import (
    validate_preregistration_approval,
)
from .promotion_evidence import (
    check_held_equal_axes,
    check_inference_envelope,
    check_noise_floors,
    evidence_arm_digests,
    spans_noise_floors,
    validate_promotion_evidence,
)
from .promotion_exports import (
    check_promotion_exports,
)
from .promotion_gate import (
    validate_promotion_or_decline,
)
from .promotion_records import (
    validate_dependency_pins,
    validate_serving_memory_budget,
    validate_study_decline,
)
from .promotion_witnesses import (
    check_family_witnesses,
    validate_family_witnesses,
)
from .quantum_acceptance import (
    admits_quantum_arm,
    composite_lower_bound,
    paired_improvement,
)
from .quantum_contracts import (
    validate_execution_receipt,
    validate_paired_receipts,
    validate_receipt_shape,
    validate_shot_noise_spec,
)
from .quantum_paired_noise import (
    paired_shot_radius,
    paired_shot_standard_error,
)
from .quantum_seeds import (
    validate_quantum_seeds,
    validate_seed_tuple,
)
from .quantum_shot_noise import (
    hoeffding_radius,
    shot_variance,
)
from .recipe_contracts import (
    validate_curriculum_stage,
    validate_pretraining_checkpoint_state,
    validate_recipe_and_curriculum,
)
from .recipe_parts import (
    validate_hardware_profile,
    validate_matched_budget,
    validate_seed_quadruple,
)
from .record_codec import (
    from_canonical_json_bytes,
)
from .rlvr_contracts import (
    validate_rlvr_spec,
    validate_verifiable_reward_outcome,
    validate_verifier_fault_attestation,
)
from .rlvr_mask_rate import (
    validate_rlvr_batch_mask_rate,
    wilson_interval,
)
from .safetensors_parser import (
    compute_tensor_set_digest,
    compute_tensor_set_identity,
    read_safetensors_descriptors,
    read_tensor_set_descriptors,
)
from .sealed_looks import (
    validate_journal_entry,
    validate_look_request,
)
from .spend_contracts import (
    amount,
    validate_cost_estimate,
    validate_paid_run_approval,
    validate_spend_binding,
    validate_spend_reservation_live,
)
from .study_contracts import (
    validate_build2_terms,
    validate_inference_envelope,
)
from .submission_guard import (
    guard_job_submission,
)
from .tensors import (
    tensor_set_identity,
)
from .tier_journal import (
    active_pins,
    check_tier_journal_order,
    require_dependencies_pinned,
    validate_tier_journal_entry,
)
from .weight_archives import (
    compute_archive_tensor_set_identity,
    list_archive_members,
    read_archive_tensor_descriptors,
)
from .width_and_position_contracts import (
    validate_mup_spec,
    validate_mup_validation_receipt,
    validate_yarn_rope_scaling,
)

__all__ = [
    "active_pins",
    "admit_sealed_look",
    "admits_quantum_arm",
    "amount",
    "any_lane_matches",
    "canonical_json_bytes",
    "check_family_witnesses",
    "check_held_equal_axes",
    "check_inference_envelope",
    "check_journal_order",
    "check_look_budget",
    "check_noise_floors",
    "check_promotion_exports",
    "check_promotion_pre_state",
    "check_tier_journal_order",
    "check_transformer_architecture_shape",
    "composite_lower_bound",
    "compute_archive_tensor_set_identity",
    "compute_family_model_identity_digest",
    "compute_family_tensor_set_identity",
    "compute_tensor_set_digest",
    "compute_tensor_set_identity",
    "decode_record",
    "derive_proposer_lineage",
    "evidence_arm_digests",
    "exempt_base_digests",
    "family_rule",
    "from_canonical_json_bytes",
    "guard_job_submission",
    "hoeffding_radius",
    "is_sealed_kind",
    "journal_entry_for",
    "list_archive_members",
    "paired_improvement",
    "paired_shot_radius",
    "paired_shot_standard_error",
    "merge_identity",
    "read_archive_tensor_descriptors",
    "read_safetensors_descriptors",
    "read_tensor_set_descriptors",
    "record_digest",
    "require_active_family",
    "require_dependencies_pinned",
    "require_output_training_permitted",
    "require_spiking_parity_passed",
    "required_paired_sample_size",
    "scan_float_chunk",
    "select_study_entries",
    "shot_variance",
    "spans_noise_floors",
    "strict_json_loads",
    "tensor_set_identity",
    "validate_barren_plateau_receipt",
    "validate_adapter_merge",
    "validate_base_checkpoint_record",
    "validate_build2_terms",
    "validate_compute_budget",
    "validate_cost_estimate",
    "validate_curriculum_stage",
    "validate_custody_ingest_record",
    "validate_dedup_and_distillation",
    "validate_dedup_report",
    "validate_dependency_pins",
    "validate_distillation_manifest",
    "validate_dynamics_evaluation_contract",
    "validate_dynamics_family_contract",
    "validate_embedding_space_identity",
    "validate_envelope",
    "validate_evaluator_output",
    "validate_execution_receipt",
    "validate_export_parity_report",
    "validate_family_tensor_layout",
    "validate_family_witnesses",
    "validate_grader",
    "validate_hardware_profile",
    "validate_inference_envelope",
    "validate_job_cancel",
    "validate_job_status",
    "validate_job_submission",
    "validate_journal_entry",
    "validate_language_family_contract",
    "validate_licence_record",
    "validate_lineage_edge",
    "validate_look_request",
    "validate_look_result",
    "validate_matched_budget",
    "validate_merge_result",
    "validate_mup_spec",
    "validate_mup_validation_receipt",
    "validate_normaliser_envelope",
    "validate_onnx_export_profile",
    "validate_paid_run_approval",
    "validate_paired_receipts",
    "validate_post_swap_evidence",
    "validate_post_swap_verification",
    "validate_preregistration",
    "validate_preregistration_approval",
    "validate_pretraining_checkpoint_state",
    "validate_promotion_decision",
    "validate_promotion_evidence",
    "validate_promotion_or_decline",
    "validate_quantum_seeds",
    "validate_receipt_shape",
    "validate_recipe_and_curriculum",
    "validate_rlvr_batch_mask_rate",
    "validate_rlvr_spec",
    "validate_sandbox_confinement",
    "validate_sealed_split_pool",
    "validate_seed_quadruple",
    "validate_seed_tuple",
    "validate_serving_memory_budget",
    "validate_shot_noise_spec",
    "validate_slice_estimate",
    "validate_spend_binding",
    "validate_spend_reservation_live",
    "validate_spiking_lowering_parity",
    "validate_study_decline",
    "validate_tier_journal_entry",
    "validate_tokeniser_and_normaliser_disjointness",
    "validate_tokeniser_envelope",
    "validate_transformer_architecture_spec",
    "validate_verifiable_reward_outcome",
    "validate_verifier_fault_attestation",
    "validate_yarn_rope_scaling",
    "wilson_interval",
]
