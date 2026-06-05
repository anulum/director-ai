# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Customer Model Factory package

"""Customer-owned guardrail training and evidence package primitives."""

from .benchmark_selection import (
    BenchmarkMetrics,
    CustomerBenchmarkResult,
    CustomerModelSelectionReport,
    select_customer_model,
)
from .dataset_contract import (
    CustomerDatasetValidationReport,
    CustomerTraceFinding,
    CustomerWorkspace,
    validate_customer_trace_dataset,
)
from .deployment_manifest import (
    CustomerDeploymentManifest,
    DeploymentPolicy,
    build_deployment_manifest,
)
from .evidence_pack import (
    CustomerEvidencePackManifest,
    build_customer_evidence_pack,
)
from .monitoring_manifest import (
    CustomerMonitoringManifest,
    MonitoringMetrics,
    MonitoringThresholds,
    build_monitoring_manifest,
)
from .release_gate import (
    ConformalRoutingEvidence,
    CustomerReleaseGateManifest,
    DeploymentHardeningEvidence,
    FederatedPrivacyEvidence,
    MultimodalTemporalEvidence,
    ObservabilityOperationsEvidence,
    ProvenanceLineageEvidence,
    TrajectoryRollbackEvidence,
    build_release_gate_manifest,
)
from .risk_register import (
    CustomerRiskException,
    CustomerRiskRegister,
    build_risk_register,
)
from .runtime_package import (
    CustomerRuntimePackage,
    build_customer_runtime_package,
)
from .sector_extension import (
    SECTOR_REQUIRED_METADATA,
    SectorEvidenceMapping,
    build_sector_evidence_mapping,
    validate_sector_trace_metadata,
)
from .training_manifest import (
    CustomerTrainingManifest,
    TrainingLane,
    build_training_manifest,
)

__all__ = [
    "BenchmarkMetrics",
    "CustomerBenchmarkResult",
    "CustomerDatasetValidationReport",
    "CustomerDeploymentManifest",
    "CustomerEvidencePackManifest",
    "CustomerMonitoringManifest",
    "CustomerModelSelectionReport",
    "CustomerRuntimePackage",
    "CustomerRiskException",
    "CustomerRiskRegister",
    "CustomerReleaseGateManifest",
    "CustomerTraceFinding",
    "CustomerTrainingManifest",
    "ConformalRoutingEvidence",
    "CustomerWorkspace",
    "DeploymentHardeningEvidence",
    "DeploymentPolicy",
    "FederatedPrivacyEvidence",
    "MultimodalTemporalEvidence",
    "MonitoringMetrics",
    "MonitoringThresholds",
    "ObservabilityOperationsEvidence",
    "ProvenanceLineageEvidence",
    "SECTOR_REQUIRED_METADATA",
    "SectorEvidenceMapping",
    "TrainingLane",
    "TrajectoryRollbackEvidence",
    "build_customer_evidence_pack",
    "build_customer_runtime_package",
    "build_deployment_manifest",
    "build_monitoring_manifest",
    "build_risk_register",
    "build_release_gate_manifest",
    "build_sector_evidence_mapping",
    "build_training_manifest",
    "select_customer_model",
    "validate_customer_trace_dataset",
    "validate_sector_trace_metadata",
]
