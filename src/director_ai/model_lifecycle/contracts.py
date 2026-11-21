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

from .base_checkpoints import BaseCheckpointRecord
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
    RESERVED_MODEL_FAMILIES,
    ActiveModelFamily,
    ArtefactKind,
    LineageEdge,
    LineageRelation,
    ModelArtefactEnvelope,
    ReservedModelFamily,
)
from .evaluation_contracts import CiMethod, PreregistrationRecord, SealedSplitPool
from .evaluator_outputs import EvaluatorOutput, SealedLookResult, SliceAggregateEstimate
from .graders import GraderRole, GraderSpec
from .licences import (
    CUSTODY_DATA_CLASSES,
    ComputeAndTuningBudget,
    CustodyDataClass,
    DualAxisLicenceRecord,
)
from .lineage import LineageComponent
from .look_admission import CustodyProgrammeRoles
from .preregistration_approvals import PreregistrationApproval
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
from .weight_archives import ArchiveMember

__all__ = [
    "CONSUMING_OUTCOMES",
    "CUSTODY_DATA_CLASSES",
    "CUSTODY_OBJECT_KINDS",
    "EMPTY_PRODUCTION_DIGEST",
    "FAULT_OUTCOMES",
    "MODEL_CUSTODY_KINDS",
    "RESERVED_MODEL_FAMILIES",
    "SAFE_TENSORS_DTYPES",
    "SEALED_CUSTODY_KINDS",
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
    "EvaluatorOutput",
    "GraderRole",
    "GraderSpec",
    "LineageComponent",
    "LineageEdge",
    "LineageRelation",
    "LookKind",
    "ModelArtefactEnvelope",
    "PreregistrationApproval",
    "PreregistrationRecord",
    "PromotionDecisionRecord",
    "ReservedModelFamily",
    "RoleAuthenticationMode",
    "SealedLookJournalEntry",
    "SealedLookOutcomeStatus",
    "SealedLookRequest",
    "SealedLookResult",
    "SealedSplitPool",
    "SliceAggregateEstimate",
    "TensorDescriptor",
    "TensorDtype",
    "TensorSetIdentity",
    "WeightFormat",
]
