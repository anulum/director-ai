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
from .sealed_looks import (
    CONSUMING_OUTCOMES,
    FAULT_OUTCOMES,
    LookKind,
    SealedLookJournalEntry,
    SealedLookOutcomeStatus,
    SealedLookRequest,
)
from .tensors import (
    SAFE_TENSORS_DTYPES,
    TensorDescriptor,
    TensorDtype,
    TensorSetIdentity,
)
from .weight_archives import (
    ArchiveMember,
)

__all__ = [
    "ACTUATION_MODES",
    "CONSUMING_OUTCOMES",
    "CUSTODY_DATA_CLASSES",
    "CUSTODY_OBJECT_KINDS",
    "DYNAMICS_FAMILY_RULE",
    "EMPTY_PRODUCTION_DIGEST",
    "FAMILY_CONTRACT_REGISTRY",
    "FAULT_OUTCOMES",
    "FIXED_POINT_PREFIX",
    "FLOAT_DTYPES",
    "HDF5_MAGIC",
    "INTEGER_BITS",
    "LANGUAGE_FAMILY_RULE",
    "MODEL_CUSTODY_KINDS",
    "QUANTUM_VARIATIONAL_FAMILY_RULE",
    "SAFE_TENSORS_DTYPES",
    "SEALED_CUSTODY_KINDS",
    "SPIKING_FAMILY_RULE",
    "ActiveModelFamily",
    "ArchiveMember",
    "ArtefactKind",
    "BaseCheckpointRecord",
    "CiMethod",
    "ComputeAndTuningBudget",
    "CustodyDataClass",
    "CustodyIngestRecord",
    "CustodyObjectKind",
    "CustodyProgrammeRoles",
    "DualAxisLicenceRecord",
    "DynamicsFamilyContract",
    "EvaluatorOutput",
    "FixedPointFormatDecl",
    "GraderRole",
    "GraderSpec",
    "LanguageFamilyContract",
    "LineageComponent",
    "LineageEdge",
    "LineageRelation",
    "LookKind",
    "ModelArtefactEnvelope",
    "PreregistrationApproval",
    "PreregistrationRecord",
    "PromotionDecisionRecord",
    "RegisteredFamilyContractRule",
    "RoleAuthenticationMode",
    "SealedLookJournalEntry",
    "SealedLookOutcomeStatus",
    "SealedLookRequest",
    "SealedLookResult",
    "SealedSplitPool",
    "SecondaryFamilyContractRef",
    "SliceAggregateEstimate",
    "TensorDescriptor",
    "TensorDtype",
    "TensorSetIdentity",
    "WeightFormat",
]
