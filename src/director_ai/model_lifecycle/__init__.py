# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Contract Kernel

"""Contract kernel for the model lifecycle: records and validators.

This package is the public import surface. Consumers, blob custody among
them, import only from ``director_ai.model_lifecycle``. It uses the Python
standard library alone: no ``torch``, ``transformers`` or ``safetensors``,
and no module under another licence, so an AGPL or BUSL host can import it
without taking in either licence.

It defines:

* the identity of weights: :class:`TensorSetIdentity` from a hardened
  safetensors and ``safetensors_tar`` reader;
* licences and budgets: :class:`DualAxisLicenceRecord`, whose
  ``output_training_permitted`` is the one source for whether outputs may
  train another model, and :class:`ComputeAndTuningBudget`;
* artefact envelopes: :class:`ModelArtefactEnvelope`, with the reserved
  families ``spiking`` and ``quantum_variational`` refused;
* custody records: :class:`CustodyIngestRecord` with its 17 keys, the fixed
  sealed kinds, :class:`BaseCheckpointRecord` and
  :class:`PreregistrationApproval`;
* evaluation: :class:`PreregistrationRecord`, the four look outcomes, the
  custody-derived, root-anchored lineage and the look budget, enforced by
  :func:`admit_sealed_look` before any split is opened;
* promotion: :class:`PromotionDecisionRecord` and its production pre-state.

Nothing here trains, evaluates or deploys a model. The package makes no
capability claim; it is the contract other components build on.

The lineage rule is exactly as strong as the authentication of the
submitting role. On hosts where every seat runs as one OS user that
authentication is procedural; see :mod:`.custody_records`.
"""

from __future__ import annotations

from .contracts import (
    CONSUMING_OUTCOMES,
    CUSTODY_DATA_CLASSES,
    CUSTODY_OBJECT_KINDS,
    EMPTY_PRODUCTION_DIGEST,
    FAULT_OUTCOMES,
    FLOAT_DTYPES,
    MODEL_CUSTODY_KINDS,
    RESERVED_MODEL_FAMILIES,
    SAFE_TENSORS_DTYPES,
    SEALED_CUSTODY_KINDS,
    ActiveModelFamily,
    ArchiveMember,
    ArtefactKind,
    BaseCheckpointRecord,
    CiMethod,
    ComputeAndTuningBudget,
    CustodyDataClass,
    CustodyIngestRecord,
    CustodyObjectKind,
    CustodyProgrammeRoles,
    DualAxisLicenceRecord,
    EvaluatorOutput,
    GraderRole,
    GraderSpec,
    LineageComponent,
    LineageEdge,
    LineageRelation,
    LookKind,
    ModelArtefactEnvelope,
    PreregistrationApproval,
    PreregistrationRecord,
    PromotionDecisionRecord,
    ReservedModelFamily,
    RoleAuthenticationMode,
    SealedLookJournalEntry,
    SealedLookOutcomeStatus,
    SealedLookRequest,
    SealedLookResult,
    SealedSplitPool,
    SliceAggregateEstimate,
    TensorDescriptor,
    TensorDtype,
    TensorSetIdentity,
    WeightFormat,
)
from .errors import (
    CustodyRecordError,
    EnvelopeError,
    EvaluationContractError,
    LicenceRecordError,
    LookBudgetExhaustedError,
    MalformedDigestError,
    MalformedWeightsError,
    ModelLifecycleError,
    NegativeZeroTensorError,
    NonCanonicalFloatError,
    NonCanonicalJsonError,
    NonFiniteTensorError,
    OutputTrainingNotPermittedError,
    PreregistrationOrderError,
    PromotionDecisionError,
    ReservedModelFamilyError,
    SampleSizeError,
    SealedPoolCustodyError,
    UnsupportedSafetensorsDtypeError,
    UnverifiedCustodyArtefactError,
)
from .validators import (
    admit_sealed_look,
    any_lane_matches,
    canonical_json_bytes,
    check_look_budget,
    check_promotion_pre_state,
    compute_archive_tensor_set_identity,
    compute_tensor_set_digest,
    compute_tensor_set_identity,
    decode_record,
    derive_proposer_lineage,
    exempt_base_digests,
    from_canonical_json_bytes,
    is_sealed_kind,
    list_archive_members,
    read_safetensors_descriptors,
    record_digest,
    require_active_family,
    require_output_training_permitted,
    required_paired_sample_size,
    scan_float_chunk,
    strict_json_loads,
    tensor_set_identity,
    validate_base_checkpoint_record,
    validate_compute_budget,
    validate_custody_ingest_record,
    validate_envelope,
    validate_evaluator_output,
    validate_grader,
    validate_journal_entry,
    validate_licence_record,
    validate_lineage_edge,
    validate_look_request,
    validate_look_result,
    validate_preregistration,
    validate_preregistration_approval,
    validate_promotion_decision,
    validate_sealed_split_pool,
    validate_slice_estimate,
)

__all__ = [
    "ActiveModelFamily",
    "ArchiveMember",
    "ArtefactKind",
    "BaseCheckpointRecord",
    "CONSUMING_OUTCOMES",
    "CUSTODY_DATA_CLASSES",
    "CUSTODY_OBJECT_KINDS",
    "CiMethod",
    "ComputeAndTuningBudget",
    "CustodyDataClass",
    "CustodyIngestRecord",
    "CustodyObjectKind",
    "CustodyProgrammeRoles",
    "CustodyRecordError",
    "DualAxisLicenceRecord",
    "EMPTY_PRODUCTION_DIGEST",
    "EnvelopeError",
    "EvaluationContractError",
    "EvaluatorOutput",
    "FAULT_OUTCOMES",
    "FLOAT_DTYPES",
    "GraderRole",
    "GraderSpec",
    "LicenceRecordError",
    "LineageComponent",
    "LineageEdge",
    "LineageRelation",
    "LookBudgetExhaustedError",
    "LookKind",
    "MODEL_CUSTODY_KINDS",
    "MalformedDigestError",
    "MalformedWeightsError",
    "ModelArtefactEnvelope",
    "ModelLifecycleError",
    "NegativeZeroTensorError",
    "NonCanonicalFloatError",
    "NonCanonicalJsonError",
    "NonFiniteTensorError",
    "OutputTrainingNotPermittedError",
    "PreregistrationApproval",
    "PreregistrationOrderError",
    "PreregistrationRecord",
    "PromotionDecisionError",
    "PromotionDecisionRecord",
    "RESERVED_MODEL_FAMILIES",
    "ReservedModelFamily",
    "ReservedModelFamilyError",
    "RoleAuthenticationMode",
    "SAFE_TENSORS_DTYPES",
    "SEALED_CUSTODY_KINDS",
    "SampleSizeError",
    "SealedLookJournalEntry",
    "SealedLookOutcomeStatus",
    "SealedLookRequest",
    "SealedLookResult",
    "SealedPoolCustodyError",
    "SealedSplitPool",
    "SliceAggregateEstimate",
    "TensorDescriptor",
    "TensorDtype",
    "TensorSetIdentity",
    "UnsupportedSafetensorsDtypeError",
    "UnverifiedCustodyArtefactError",
    "WeightFormat",
    "admit_sealed_look",
    "any_lane_matches",
    "canonical_json_bytes",
    "check_look_budget",
    "check_promotion_pre_state",
    "compute_archive_tensor_set_identity",
    "compute_tensor_set_digest",
    "compute_tensor_set_identity",
    "decode_record",
    "derive_proposer_lineage",
    "exempt_base_digests",
    "from_canonical_json_bytes",
    "is_sealed_kind",
    "list_archive_members",
    "read_safetensors_descriptors",
    "record_digest",
    "require_active_family",
    "require_output_training_permitted",
    "required_paired_sample_size",
    "scan_float_chunk",
    "strict_json_loads",
    "tensor_set_identity",
    "validate_base_checkpoint_record",
    "validate_compute_budget",
    "validate_custody_ingest_record",
    "validate_envelope",
    "validate_evaluator_output",
    "validate_grader",
    "validate_journal_entry",
    "validate_licence_record",
    "validate_lineage_edge",
    "validate_look_request",
    "validate_look_result",
    "validate_preregistration",
    "validate_preregistration_approval",
    "validate_promotion_decision",
    "validate_sealed_split_pool",
    "validate_slice_estimate",
]
