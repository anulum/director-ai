# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Dataset Records

"""Records of a training dataset's split audit and synthetic teacher rows.

:class:`DatasetDeduplicationReport` audits a dataset's partitions: their
shards, cluster counts and model-visible bytes, the clusters and
near-duplicate windows shared across partitions, synthetic rows outside the
training partition, the template families on each side, and that no human
biosignal or health data is present. :class:`SyntheticDistillationManifest`
names the teacher whose outputs became training rows; whether its outputs
may train another model is read from the teacher's licence record alone.
The rules are in :mod:`.dataset_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .licences import DualAxisLicenceRecord

SplitPartition = Literal["train", "selection", "calibration", "sealed"]
SPLIT_PARTITIONS: frozenset[str] = frozenset(
    {"train", "selection", "calibration", "sealed"}
)
DOMAIN_KINDS: frozenset[str] = frozenset(
    {
        "language_corpus",
        "dynamics_trajectories",
        "spiking_stimuli",
        "quantum_hamiltonians",
    }
)
CLUSTER_KEY_KINDS: frozenset[str] = frozenset(
    {
        "document_cluster",
        "episode_id",
        "tokamak_shot_or_discharge_id",
        "molecule_or_hamiltonian_family_id",
    }
)
MIN_REAL_TRAIN_CLUSTER_FRACTION_WITH_SYNTHETIC = 0.25
"""The least share of real training clusters when synthetic rows are used."""


@dataclass(frozen=True, slots=True)
class DatasetDeduplicationReport:
    """Cluster-disjoint split audit, byte accounting, and Zero-PHI attestation."""

    schema: Literal["dataset-dedup-report.v1"]
    domain_kind: Literal[
        "language_corpus",
        "dynamics_trajectories",
        "spiking_stimuli",
        "quantum_hamiltonians",
    ]
    cluster_key_kind: Literal[
        "document_cluster",
        "episode_id",
        "tokamak_shot_or_discharge_id",
        "molecule_or_hamiltonian_family_id",
    ]
    partition_shard_digests: tuple[tuple[SplitPartition, tuple[str, ...]], ...]
    partition_cluster_counts: tuple[tuple[SplitPartition, int], ...]
    partition_model_visible_bytes: tuple[tuple[SplitPartition, int], ...]
    cross_split_shared_cluster_count: int
    cross_split_near_duplicate_window_count: int
    synthetic_non_train_row_count: int
    real_train_cluster_fraction: float
    train_template_family_ids: tuple[str, ...]
    non_train_template_family_ids: tuple[str, ...]
    scrubber_report_digest: str
    contains_human_biosignal_or_phi: Literal[False]


@dataclass(frozen=True, slots=True)
class SyntheticDistillationManifest:
    """Provenance manifest for synthetic teacher rows; reads licence from teacher record."""

    schema: Literal["synthetic-distillation-manifest.v1"]
    teacher_artefact_digest: str
    teacher_licence_record: DualAxisLicenceRecord
    verifier_grader_id: str
    dedup_report_digest: str
