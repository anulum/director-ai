# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Training helpers and managed job backends."""

from .dataset_fingerprint import DatasetFingerprint, fingerprint_dataset
from .experiment_tracker import ExperimentRun, ExperimentTracker
from .jobs import (
    PortableTrainingBackend,
    TrainingHardware,
    TrainingJobSpec,
    TrainingJobStatus,
    TrainingJobSubmission,
    build_internal_suite_spec,
    build_portable_container_job_request,
    build_vertex_custom_job_request,
    get_training_backend,
    submit_training_job,
)
from .model_registry import (
    DEFAULT_FINE_TUNE_MODEL_ALIAS,
    TrainingModelProfile,
    finetune_model_registry_to_dict,
    list_finetune_model_profiles,
    resolve_finetune_model,
)
from .results import (
    TrainingHarvestReport,
    TrainingResultRecord,
    harvest_training_results,
)
from .sweeps import (
    TrainingDatasetSplit,
    TrainingScenario,
    TrainingSweepPlan,
    build_training_sweep_plan,
)
from .trained_model_registry import (
    STAGE_CANDIDATE,
    STAGE_PRODUCTION,
    STAGE_RETIRED,
    TrainedModelRecord,
    TrainedModelRegistry,
)

__all__ = [
    "DEFAULT_FINE_TUNE_MODEL_ALIAS",
    "STAGE_CANDIDATE",
    "STAGE_PRODUCTION",
    "STAGE_RETIRED",
    "DatasetFingerprint",
    "ExperimentRun",
    "ExperimentTracker",
    "PortableTrainingBackend",
    "TrainedModelRecord",
    "TrainedModelRegistry",
    "TrainingDatasetSplit",
    "TrainingHardware",
    "TrainingJobSpec",
    "TrainingJobStatus",
    "TrainingJobSubmission",
    "TrainingHarvestReport",
    "TrainingModelProfile",
    "TrainingResultRecord",
    "TrainingScenario",
    "TrainingSweepPlan",
    "build_internal_suite_spec",
    "build_portable_container_job_request",
    "build_training_sweep_plan",
    "build_vertex_custom_job_request",
    "fingerprint_dataset",
    "finetune_model_registry_to_dict",
    "get_training_backend",
    "harvest_training_results",
    "list_finetune_model_profiles",
    "resolve_finetune_model",
    "submit_training_job",
]
