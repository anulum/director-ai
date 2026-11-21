# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Record Codec

"""Decode any kernel record from canonical JSON and validate it.

:func:`from_canonical_json_bytes` looks up the record type's validator, so a
consumer such as blob custody cannot decode a record without checking it.
A type without a registered validator is refused. Two checks need more
than the record and are left to their callers: a look result's binding to
its journal entry (:func:`~.validators.validate_look_result`) and the
width-transfer receipt an architecture spec needs on its funding pool
(:func:`~.architecture_contracts.validate_transformer_architecture_spec`).

Only a record's canonical bytes decode. Custody keys a record blob by the
SHA-256 of its bytes, while admission looks a record up by the digest of
its canonical encoding; the two agree only for canonical bytes, so any other
spelling (whitespace, key order, ``1.50``, an exponent that underflows to
zero) is refused here rather than missed later.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from ._checks import require_choice, require_int, require_sha256
from .architecture_contracts import check_transformer_architecture_shape
from .architecture_specs import MupValidationReceipt, TransformerArchitectureSpec
from .base_checkpoints import BaseCheckpointRecord, validate_base_checkpoint_record
from .cost_and_promotion_contracts import (
    PromotionDecisionRecord,
    validate_promotion_decision,
)
from .custody_records import CustodyIngestRecord, validate_custody_ingest_record
from .dataset_contracts import validate_dedup_report, validate_distillation_manifest
from .dataset_records import DatasetDeduplicationReport, SyntheticDistillationManifest
from .digests import canonical_json_bytes, decode_record
from .encoding_contracts import (
    validate_embedding_space_identity,
    validate_normaliser_envelope,
    validate_tokeniser_envelope,
)
from .encoding_records import (
    DynamicsNormaliserEnvelope,
    EmbeddingSpaceIdentity,
    TokeniserArtefactEnvelope,
)
from .envelopes import (
    LineageEdge,
    ModelArtefactEnvelope,
    validate_envelope,
    validate_lineage_edge,
)
from .errors import (
    EvaluationContractError,
    MalformedWeightsError,
    NonCanonicalJsonError,
)
from .evaluation_contracts import (
    PreregistrationRecord,
    SealedSplitPool,
    validate_preregistration,
    validate_sealed_split_pool,
)
from .evaluator_outputs import (
    EvaluatorOutput,
    SealedLookResult,
    SliceAggregateEstimate,
    validate_evaluator_output,
    validate_slice_estimate,
)
from .export_contracts import (
    validate_export_parity_report,
    validate_onnx_export_profile,
)
from .family_companions import (
    DynamicsFamilyContract,
    LanguageFamilyContract,
    validate_dynamics_family_contract,
    validate_language_family_contract,
)
from .graders import GraderSpec, validate_grader
from .job_contracts import (
    validate_job_cancel,
    validate_job_status,
    validate_job_submission,
    validate_sandbox_confinement,
)
from .licences import (
    ComputeAndTuningBudget,
    DualAxisLicenceRecord,
    validate_compute_budget,
    validate_licence_record,
)
from .preregistration_approvals import (
    PreregistrationApproval,
    validate_preregistration_approval,
)
from .promotion_records import (
    PromotionDependencyPins,
    ServingMemoryBudget,
    StudyDeclineRecord,
    validate_dependency_pins,
    validate_serving_memory_budget,
    validate_study_decline,
)
from .recipe_contracts import (
    validate_pretraining_checkpoint_state,
    validate_recipe_and_curriculum,
)
from .recipe_records import PretrainingCheckpointState, TrainingRecipeManifest
from .rlvr_contracts import validate_rlvr_spec, validate_verifier_fault_attestation
from .rlvr_records import RlvrAlgorithmSpec, VerifierFaultAttestation
from .sealed_looks import (
    SealedLookJournalEntry,
    SealedLookRequest,
    validate_journal_entry,
    validate_look_request,
)
from .spend_contracts import (
    validate_cost_estimate,
    validate_paid_run_approval,
    validate_spend_binding,
)
from .spend_records import (
    CloudJobCancelReceipt,
    CloudJobStatusReport,
    CloudJobSubmissionReceipt,
    CostEstimateRecord,
    PaidRunApprovalRecord,
    SandboxConfinementSpec,
    SpendReservationBinding,
)
from .study_contracts import (
    validate_build2_terms,
    validate_dynamics_evaluation_contract,
    validate_inference_envelope,
)
from .study_records import (
    DynamicsEvaluationContract,
    ExportParityReport,
    InferenceComputeEnvelope,
    OnnxExportProfileSpec,
    PreregistrationBuild2Terms,
)
from .tensors import TensorDescriptor, TensorSetIdentity, tensor_set_identity
from .width_and_position_contracts import validate_mup_validation_receipt

RecordT = TypeVar("RecordT")


def _validate_descriptor(descriptor: TensorDescriptor) -> None:
    """Check one descriptor as a one-tensor set."""
    tensor_set_identity([descriptor])


def _validate_identity(identity: TensorSetIdentity) -> None:
    """Check a tensor-set identity's fields."""
    require_choice(
        identity.schema, "schema", {"tensor-set-identity.v1"}, MalformedWeightsError
    )
    require_int(identity.tensor_count, "tensor_count", MalformedWeightsError, minimum=1)
    require_int(
        identity.total_byte_length,
        "total_byte_length",
        MalformedWeightsError,
        minimum=0,
    )
    require_sha256(identity.tensor_set_digest, "tensor_set_digest")


def _validate_result(result: SealedLookResult) -> None:
    """Check a look result's own fields; its binding to an entry needs the entry."""
    require_choice(
        result.schema, "schema", {"sealed-look-result.v1"}, EvaluationContractError
    )
    require_sha256(result.look_journal_digest, "look_journal_digest")
    require_sha256(result.derived_proposer_lineage_id, "derived_proposer_lineage_id")
    require_sha256(result.split_digest, "split_digest")
    validate_evaluator_output(result.evaluator_output)


VALIDATORS: dict[type, Callable[[object], object]] = {
    cast(type, record): cast(Callable[[object], object], validator)
    for record, validator in (
        (DualAxisLicenceRecord, validate_licence_record),
        (ComputeAndTuningBudget, validate_compute_budget),
        (LineageEdge, validate_lineage_edge),
        (ModelArtefactEnvelope, validate_envelope),
        (BaseCheckpointRecord, validate_base_checkpoint_record),
        (CustodyIngestRecord, validate_custody_ingest_record),
        (PreregistrationApproval, validate_preregistration_approval),
        (GraderSpec, validate_grader),
        (SealedSplitPool, validate_sealed_split_pool),
        (PreregistrationRecord, validate_preregistration),
        (SealedLookRequest, validate_look_request),
        (SealedLookJournalEntry, validate_journal_entry),
        (SliceAggregateEstimate, validate_slice_estimate),
        (EvaluatorOutput, validate_evaluator_output),
        (SealedLookResult, _validate_result),
        (PromotionDecisionRecord, validate_promotion_decision),
        (TensorDescriptor, _validate_descriptor),
        (TensorSetIdentity, _validate_identity),
        (LanguageFamilyContract, validate_language_family_contract),
        (DynamicsFamilyContract, validate_dynamics_family_contract),
        (TransformerArchitectureSpec, check_transformer_architecture_shape),
        (MupValidationReceipt, validate_mup_validation_receipt),
        (DatasetDeduplicationReport, validate_dedup_report),
        (SyntheticDistillationManifest, validate_distillation_manifest),
        (TokeniserArtefactEnvelope, validate_tokeniser_envelope),
        (DynamicsNormaliserEnvelope, validate_normaliser_envelope),
        (EmbeddingSpaceIdentity, validate_embedding_space_identity),
        (TrainingRecipeManifest, validate_recipe_and_curriculum),
        (PretrainingCheckpointState, validate_pretraining_checkpoint_state),
        (RlvrAlgorithmSpec, validate_rlvr_spec),
        (VerifierFaultAttestation, validate_verifier_fault_attestation),
        (InferenceComputeEnvelope, validate_inference_envelope),
        (PreregistrationBuild2Terms, validate_build2_terms),
        (DynamicsEvaluationContract, validate_dynamics_evaluation_contract),
        (OnnxExportProfileSpec, validate_onnx_export_profile),
        (ExportParityReport, validate_export_parity_report),
        (SpendReservationBinding, validate_spend_binding),
        (CostEstimateRecord, validate_cost_estimate),
        (PaidRunApprovalRecord, validate_paid_run_approval),
        (SandboxConfinementSpec, validate_sandbox_confinement),
        (CloudJobSubmissionReceipt, validate_job_submission),
        (CloudJobStatusReport, validate_job_status),
        (CloudJobCancelReceipt, validate_job_cancel),
        (ServingMemoryBudget, validate_serving_memory_budget),
        (PromotionDependencyPins, validate_dependency_pins),
        (StudyDeclineRecord, validate_study_decline),
    )
}
"""Every record type the codec decodes, with the validator it runs."""


def from_canonical_json_bytes(cls: type[RecordT], raw: bytes) -> RecordT:
    """Decode a kernel record from JSON bytes and run its validator.

    Parameters
    ----------
    cls : type
        A kernel record type registered in :data:`VALIDATORS`.
    raw : bytes
        UTF-8 JSON text of one record.

    Returns
    -------
    RecordT
        The decoded and validated record.

    Raises
    ------
    NonCanonicalJsonError
        If ``cls`` has no registered validator, the bytes break the strict
        decoding rules, or they are not the record's canonical encoding.
    ModelLifecycleError
        The validator's named refusal when the record is malformed.
    """
    validator = VALIDATORS.get(cls)
    if validator is None:
        raise NonCanonicalJsonError(
            f"{getattr(cls, '__name__', cls)!s} has no registered validator"
        )
    record = decode_record(cls, raw, validator)
    if canonical_json_bytes(record) != raw:
        raise NonCanonicalJsonError(
            f"{cls.__name__} bytes are not the record's canonical encoding"
        )
    return record
