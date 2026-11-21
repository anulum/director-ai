# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Port Facade

"""Re-export of every port the kernel defines for adapters to implement.

The kernel holds no adapter. Each port is a runtime-checkable protocol, and
the kernel validates what an adapter returns. ``OperatorReadinessPort`` is
not part of the kernel; its owner keeps it in its own repository.
"""

from __future__ import annotations

from .ports_compute import CloudComputeDriverPort, ComputeAndSpendPort
from .ports_custody import ChainedDeploymentPort, CustodySubprocessPort
from .ports_data import DatasetCuratorPort, DynamicsNormaliserPort, TokeniserTrainerPort
from .ports_rewards import (
    SelfPlayProblemSynthesizerPort,
    VerifiableRewardPort,
    ZpdCurriculumSchedulerPort,
)
from .ports_training import (
    AdapterMergeAndExportPort,
    EvaluationGraderPort,
    TestTimeVerifierSearchPort,
    TrainingBackendPort,
)
from .ports_verifiers import PhysicsDynamicsVerifierPort, QuantumVariationalVerifierPort

__all__ = [
    "AdapterMergeAndExportPort",
    "ChainedDeploymentPort",
    "CloudComputeDriverPort",
    "ComputeAndSpendPort",
    "CustodySubprocessPort",
    "DatasetCuratorPort",
    "DynamicsNormaliserPort",
    "EvaluationGraderPort",
    "PhysicsDynamicsVerifierPort",
    "QuantumVariationalVerifierPort",
    "SelfPlayProblemSynthesizerPort",
    "TestTimeVerifierSearchPort",
    "TokeniserTrainerPort",
    "TrainingBackendPort",
    "VerifiableRewardPort",
    "ZpdCurriculumSchedulerPort",
]
