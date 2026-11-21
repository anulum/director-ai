# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Record Facade

"""Re-export of every record type and constant of the kernel.

Consumers import from :mod:`director_ai.model_lifecycle`; this facade only
groups the records by kind for the package itself.
"""

from __future__ import annotations

from .architecture_contracts import (
    MUP_INV_D_PROFILES,
    PROFILE_LOCKS,
)
from .architecture_specs import (
    ARCHITECTURE_KINDS,
    DOCUMENT_MASK_MODES,
    GGUF_WIRE_PROFILES,
    POOL_FUNDING_KINDS,
    ArchitectureKind,
    DocumentMaskMode,
    GgufWireProfile,
    MaximalUpdateParametrisationSpec,
    MupValidationReceipt,
    PoolFundingKind,
    TransformerArchitectureSpec,
    YarnRopeScalingSpec,
)
from .base_checkpoints import (
    BaseCheckpointRecord,
)
from .cost_and_promotion_contracts import (
    EMPTY_PRODUCTION_DIGEST,
    PromotionDecisionRecord,
)
from .custody_records import (
    CUSTODY_OBJECT_KINDS,
    MODEL_CUSTODY_KINDS,
    SEALED_CUSTODY_KINDS,
    CustodyIngestRecord,
    CustodyObjectKind,
    RoleAuthenticationMode,
    WeightFormat,
)
from .dataset_records import (
    CLUSTER_KEY_KINDS,
    DOMAIN_KINDS,
    MIN_REAL_TRAIN_CLUSTER_FRACTION_WITH_SYNTHETIC,
    SPLIT_PARTITIONS,
    DatasetDeduplicationReport,
    SplitPartition,
    SyntheticDistillationManifest,
)
from .deployment_records import (
    OBSERVATION_OUTCOMES,
    POST_SWAP_EFFECTS,
    POST_SWAP_OBSERVATIONS,
    POST_SWAP_RESULTS,
    PostSwapEvidence,
    PostSwapVerification,
)
from .encoding_records import (
    POOLING_MODES,
    SCALER_KINDS,
    TOKENISER_ALGORITHMS,
    UNICODE_NORMALISATIONS,
    DynamicsNormaliserEnvelope,
    EmbeddingSpaceIdentity,
    TokeniserArtefactEnvelope,
)
from .envelopes import (
    ActiveModelFamily,
    ArtefactKind,
    LineageEdge,
    LineageRelation,
    ModelArtefactEnvelope,
)
from .evaluation_contracts import (
    CiMethod,
    PreregistrationRecord,
    SealedSplitPool,
)
from .evaluator_outputs import (
    EvaluatorOutput,
    SealedLookResult,
    SliceAggregateEstimate,
)
from .family_companions import (
    ACTUATION_MODES,
    DynamicsFamilyContract,
    LanguageFamilyContract,
    SecondaryFamilyContractRef,
)
from .family_contracts import (
    HDF5_MAGIC,
)
from .family_registry import (
    DYNAMICS_FAMILY_RULE,
    FAMILY_CONTRACT_REGISTRY,
    FIXED_POINT_PREFIX,
    INTEGER_BITS,
    LANGUAGE_FAMILY_RULE,
    QUANTUM_VARIATIONAL_FAMILY_RULE,
    SPIKING_FAMILY_RULE,
    FixedPointFormatDecl,
    RegisteredFamilyContractRule,
)
from .float_scan import (
    FLOAT_DTYPES,
)
from .graders import (
    GraderRole,
    GraderSpec,
)
from .licences import (
    CUSTODY_DATA_CLASSES,
    ComputeAndTuningBudget,
    CustodyDataClass,
    DualAxisLicenceRecord,
)
from .lineage import (
    LineageComponent,
)
from .look_admission import (
    CustodyProgrammeRoles,
)
from .preregistration_approvals import (
    PreregistrationApproval,
)
from .promotion_evidence import (
    ENVELOPE_CAPS,
    MIN_TRAINING_SEEDS,
    NOISE_FLOOR_MULTIPLE,
    PromotionEvidence,
)
from .promotion_records import (
    PromotionDependencyPins,
    ServingMemoryBudget,
    StudyDeclineRecord,
)
from .quantum_acceptance import (
    EXACT_TOLERANCE_MULTIPLE,
)
from .quantum_contracts import (
    MIN_NOISE_FLOOR_MULTIPLIER,
    MIN_SHOTS_PER_PAULI_GROUP,
)
from .quantum_paired_noise import (
    PairedShotInstance,
)
from .quantum_records import (
    OPTIMISATION_SENSES,
    QUANTUM_EXECUTION_MODES,
    SPIKING_RESET_SEMANTICS,
    BarrenPlateauWitnessReceipt,
    QuantumCircuitExecutionReceipt,
    QuantumShotNoiseAcceptanceSpec,
    QuantumVariationalSeedTuple,
    SpikingLoweringParityReport,
)
from .quantum_seeds import (
    SEED_LIMIT,
)
from .quantum_shot_noise import (
    MAX_SHOTS_PER_PAULI_GROUP,
    PauliGroupShots,
)
from .recipe_records import (
    ADAPTATION_MODES,
    BUDGET_AXES,
    CURRICULUM_PHASES,
    HARDWARE_TIERS,
    NO_BF16_TIERS,
    REPRODUCIBILITY_MODES,
    CurriculumStageSpec,
    HardwareCapabilityProfile,
    MatchedBudgetAccountingSpec,
    PretrainingCheckpointState,
    SeedQuadruple,
    TrainingRecipeManifest,
)
from .rlvr_records import (
    FAULT_KINDS,
    MAX_DRIFT_ALPHA,
    MAX_MASKED_FRACTION_CEILING,
    OUTCOME_STATUSES,
    RLVR_ALGORITHMS,
    RlvrAlgorithmSpec,
    VerifiableOutcomeStatus,
    VerifiableRewardOutcome,
    VerifierFaultAttestation,
)
from .sealed_looks import (
    CONSUMING_OUTCOMES,
    FAULT_OUTCOMES,
    JOURNAL_ORIGIN_SEQ,
    RETIRED_JOURNAL_SCHEMAS,
    LookKind,
    SealedLookJournalEntry,
    SealedLookOutcomeStatus,
    SealedLookRequest,
)
from .spend_contracts import (
    MONEY_CONTEXT,
)
from .spend_records import (
    JOB_STATES,
    NETWORK_ISOLATION_LAYERS,
    RESERVATION_STATES,
    SCHEDULING_STRATEGIES,
    CloudJobCancelReceipt,
    CloudJobStatusReport,
    CloudJobSubmissionReceipt,
    CostEstimateRecord,
    LiveReservationProjection,
    PaidRunApprovalRecord,
    SandboxConfinementSpec,
    SpendReservationBinding,
)
from .study_records import (
    INT8_QUANTISATION_MODES,
    MAX_FAMILYWISE_ALPHA,
    WASM_AUDIOWORKLET_MAX_BYTES,
    DynamicsEvaluationContract,
    ExportParityMetricSlice,
    ExportParityReport,
    InferenceComputeEnvelope,
    OnnxExportProfileSpec,
    PreregistrationBuild2Terms,
)
from .submission_guard import (
    LOCAL_FUNDING,
    PAID_FUNDING,
)
from .tensors import (
    SAFE_TENSORS_DTYPES,
    TensorDescriptor,
    TensorDtype,
    TensorSetIdentity,
)
from .tier_journal import (
    PIN_ACTIONS,
    TIER_ACTIONS,
    TIER_JOURNAL_ORIGIN_SEQ,
    CustodyTierJournalEntry,
)
from .weight_archives import (
    ArchiveMember,
)

__all__ = [
    "ACTUATION_MODES",
    "ADAPTATION_MODES",
    "ARCHITECTURE_KINDS",
    "BUDGET_AXES",
    "CLUSTER_KEY_KINDS",
    "CONSUMING_OUTCOMES",
    "CURRICULUM_PHASES",
    "CUSTODY_DATA_CLASSES",
    "CUSTODY_OBJECT_KINDS",
    "DOCUMENT_MASK_MODES",
    "DOMAIN_KINDS",
    "DYNAMICS_FAMILY_RULE",
    "EMPTY_PRODUCTION_DIGEST",
    "ENVELOPE_CAPS",
    "EXACT_TOLERANCE_MULTIPLE",
    "FAMILY_CONTRACT_REGISTRY",
    "FAULT_KINDS",
    "FAULT_OUTCOMES",
    "FIXED_POINT_PREFIX",
    "FLOAT_DTYPES",
    "GGUF_WIRE_PROFILES",
    "HARDWARE_TIERS",
    "HDF5_MAGIC",
    "INT8_QUANTISATION_MODES",
    "INTEGER_BITS",
    "JOB_STATES",
    "JOURNAL_ORIGIN_SEQ",
    "LANGUAGE_FAMILY_RULE",
    "LOCAL_FUNDING",
    "MAX_DRIFT_ALPHA",
    "MAX_FAMILYWISE_ALPHA",
    "MAX_MASKED_FRACTION_CEILING",
    "MAX_SHOTS_PER_PAULI_GROUP",
    "MIN_NOISE_FLOOR_MULTIPLIER",
    "MIN_REAL_TRAIN_CLUSTER_FRACTION_WITH_SYNTHETIC",
    "MIN_TRAINING_SEEDS",
    "MIN_SHOTS_PER_PAULI_GROUP",
    "MODEL_CUSTODY_KINDS",
    "MONEY_CONTEXT",
    "MUP_INV_D_PROFILES",
    "NETWORK_ISOLATION_LAYERS",
    "NOISE_FLOOR_MULTIPLE",
    "NO_BF16_TIERS",
    "OBSERVATION_OUTCOMES",
    "OPTIMISATION_SENSES",
    "OUTCOME_STATUSES",
    "PIN_ACTIONS",
    "PAID_FUNDING",
    "POOLING_MODES",
    "POOL_FUNDING_KINDS",
    "POST_SWAP_EFFECTS",
    "POST_SWAP_OBSERVATIONS",
    "POST_SWAP_RESULTS",
    "PROFILE_LOCKS",
    "QUANTUM_EXECUTION_MODES",
    "QUANTUM_VARIATIONAL_FAMILY_RULE",
    "REPRODUCIBILITY_MODES",
    "RESERVATION_STATES",
    "RETIRED_JOURNAL_SCHEMAS",
    "RLVR_ALGORITHMS",
    "SAFE_TENSORS_DTYPES",
    "SCALER_KINDS",
    "SCHEDULING_STRATEGIES",
    "SEALED_CUSTODY_KINDS",
    "SEED_LIMIT",
    "SPIKING_FAMILY_RULE",
    "SPIKING_RESET_SEMANTICS",
    "SPLIT_PARTITIONS",
    "TIER_ACTIONS",
    "TIER_JOURNAL_ORIGIN_SEQ",
    "TOKENISER_ALGORITHMS",
    "UNICODE_NORMALISATIONS",
    "WASM_AUDIOWORKLET_MAX_BYTES",
    "ActiveModelFamily",
    "ArchitectureKind",
    "ArchiveMember",
    "ArtefactKind",
    "BarrenPlateauWitnessReceipt",
    "BaseCheckpointRecord",
    "CiMethod",
    "CloudJobCancelReceipt",
    "CloudJobStatusReport",
    "CloudJobSubmissionReceipt",
    "ComputeAndTuningBudget",
    "CostEstimateRecord",
    "CurriculumStageSpec",
    "CustodyDataClass",
    "CustodyIngestRecord",
    "CustodyObjectKind",
    "CustodyProgrammeRoles",
    "CustodyTierJournalEntry",
    "DatasetDeduplicationReport",
    "DocumentMaskMode",
    "DualAxisLicenceRecord",
    "DynamicsEvaluationContract",
    "DynamicsFamilyContract",
    "DynamicsNormaliserEnvelope",
    "EmbeddingSpaceIdentity",
    "EvaluatorOutput",
    "ExportParityMetricSlice",
    "ExportParityReport",
    "FixedPointFormatDecl",
    "GgufWireProfile",
    "GraderRole",
    "GraderSpec",
    "HardwareCapabilityProfile",
    "InferenceComputeEnvelope",
    "LanguageFamilyContract",
    "LineageComponent",
    "LineageEdge",
    "LineageRelation",
    "LiveReservationProjection",
    "LookKind",
    "MatchedBudgetAccountingSpec",
    "MaximalUpdateParametrisationSpec",
    "ModelArtefactEnvelope",
    "MupValidationReceipt",
    "OnnxExportProfileSpec",
    "PaidRunApprovalRecord",
    "PairedShotInstance",
    "PauliGroupShots",
    "PoolFundingKind",
    "PostSwapEvidence",
    "PostSwapVerification",
    "PreregistrationApproval",
    "PreregistrationBuild2Terms",
    "PreregistrationRecord",
    "PretrainingCheckpointState",
    "PromotionDecisionRecord",
    "PromotionDependencyPins",
    "PromotionEvidence",
    "QuantumCircuitExecutionReceipt",
    "QuantumShotNoiseAcceptanceSpec",
    "QuantumVariationalSeedTuple",
    "RegisteredFamilyContractRule",
    "RlvrAlgorithmSpec",
    "RoleAuthenticationMode",
    "SandboxConfinementSpec",
    "SealedLookJournalEntry",
    "SealedLookOutcomeStatus",
    "SealedLookRequest",
    "SealedLookResult",
    "SealedSplitPool",
    "SecondaryFamilyContractRef",
    "SeedQuadruple",
    "ServingMemoryBudget",
    "SliceAggregateEstimate",
    "SpendReservationBinding",
    "SpikingLoweringParityReport",
    "SplitPartition",
    "StudyDeclineRecord",
    "SyntheticDistillationManifest",
    "TensorDescriptor",
    "TensorDtype",
    "TensorSetIdentity",
    "TokeniserArtefactEnvelope",
    "TrainingRecipeManifest",
    "TransformerArchitectureSpec",
    "VerifiableOutcomeStatus",
    "VerifiableRewardOutcome",
    "VerifierFaultAttestation",
    "WeightFormat",
    "YarnRopeScalingSpec",
]
