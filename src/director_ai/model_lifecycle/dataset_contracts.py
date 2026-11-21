# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Dataset Validators

"""Validators of split audits and synthetic distillation manifests.

:func:`validate_dedup_report` checks a report's shape; a report may record a
failed audit. :func:`validate_dedup_and_distillation` is the gate a training
run passes: no cluster or near-duplicate window shared across partitions, no
synthetic row outside training, disjoint template families, no human
biosignal or health data, real training clusters present (at least a
quarter of them when synthetic rows are used), and a teacher whose licence
permits training on its outputs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ._checks import (
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .dataset_records import (
    CLUSTER_KEY_KINDS,
    DOMAIN_KINDS,
    MIN_REAL_TRAIN_CLUSTER_FRACTION_WITH_SYNTHETIC,
    SPLIT_PARTITIONS,
    DatasetDeduplicationReport,
    SyntheticDistillationManifest,
)
from .digests import record_digest
from .errors import DatasetContractError, SplitLeakageError
from .licences import require_output_training_permitted, validate_licence_record

E = DatasetContractError
EntryT = TypeVar("EntryT")


def partition_table(
    value: object, field: str, check: Callable[[object, str], EntryT]
) -> dict[str, EntryT]:
    """Return a partition table as a mapping, refusing unknown or repeated partitions.

    Parameters
    ----------
    value : object
        A tuple of ``(partition, entry)`` pairs.
    field : str
        The field name, for error messages.
    check : Callable[[object, str], EntryT]
        Checks one entry and returns it.

    Returns
    -------
    dict[str, EntryT]
        The entries by partition.

    Raises
    ------
    DatasetContractError
        If the table is not a tuple of pairs, names an unknown partition,
        repeats one, or has no ``train`` partition.
    """
    table: dict[str, EntryT] = {}
    for pair in require_tuple(value, field, E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E(f"{field} must hold (partition, value) pairs")
        partition = require_choice(pair[0], f"{field} partition", SPLIT_PARTITIONS, E)
        if partition in table:
            raise E(f"{field} repeats partition {partition!r}")
        table[partition] = check(pair[1], f"{field}[{partition}]")
    if "train" not in table:
        raise E(f"{field} needs a train partition")
    return table


def _digests(value: object, field: str) -> tuple[object, ...]:
    """Return a non-empty tuple of distinct digests."""
    items = require_tuple(value, field, E)
    if not items:
        raise E(f"{field} needs at least one digest")
    for item in items:
        require_sha256(item, field)
    require_unique(items, field, E)
    return items


def _count(value: object, field: str) -> int:
    """Return an integer of at least zero."""
    return require_int(value, field, E, minimum=0)


def _family_ids(value: object, field: str) -> set[object]:
    """Return a set of distinct, non-empty template family ids."""
    ids = require_tuple(value, field, E)
    for item in ids:
        require_text(item, field, E)
    require_unique(ids, field, E)
    return set(ids)


def validate_dedup_report(
    report: DatasetDeduplicationReport,
) -> DatasetDeduplicationReport:
    """Check a split audit's shape; the counts it records may be non-zero.

    Parameters
    ----------
    report : DatasetDeduplicationReport
        The report to check.

    Returns
    -------
    DatasetDeduplicationReport
        The same report.

    Raises
    ------
    DatasetContractError
        If a field is malformed, the three partition tables name different
        partitions, a shard appears in two partitions, or biosignal or
        health data is attested.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(report, DatasetDeduplicationReport):
        raise E("a DatasetDeduplicationReport is required")
    require_choice(report.schema, "schema", {"dataset-dedup-report.v1"}, E)
    require_choice(report.domain_kind, "domain_kind", DOMAIN_KINDS, E)
    require_choice(report.cluster_key_kind, "cluster_key_kind", CLUSTER_KEY_KINDS, E)
    shards = partition_table(
        report.partition_shard_digests, "partition_shard_digests", _digests
    )
    clusters = partition_table(
        report.partition_cluster_counts, "partition_cluster_counts", _count
    )
    visible = partition_table(
        report.partition_model_visible_bytes, "partition_model_visible_bytes", _count
    )
    if not set(shards) == set(clusters) == set(visible):
        raise E("the partition tables must name the same partitions")
    every = [digest for items in shards.values() for digest in items]
    if len(set(every)) != len(every):
        raise E("a shard digest appears in more than one partition")
    for name in (
        "cross_split_shared_cluster_count",
        "cross_split_near_duplicate_window_count",
        "synthetic_non_train_row_count",
    ):
        _count(getattr(report, name), name)
    fraction = require_finite(
        report.real_train_cluster_fraction, "real_train_cluster_fraction", E
    )
    if not 0.0 <= fraction <= 1.0:
        raise E("real_train_cluster_fraction must lie in [0, 1]")
    _family_ids(report.train_template_family_ids, "train_template_family_ids")
    _family_ids(report.non_train_template_family_ids, "non_train_template_family_ids")
    require_sha256(report.scrubber_report_digest, "scrubber_report_digest")
    if report.contains_human_biosignal_or_phi is not False:
        raise E("human biosignal or health data is never held in custody")
    return report


def validate_distillation_manifest(
    manifest: SyntheticDistillationManifest,
) -> SyntheticDistillationManifest:
    """Check a distillation manifest's fields and its teacher's licence record.

    Parameters
    ----------
    manifest : SyntheticDistillationManifest
        The manifest to check.

    Returns
    -------
    SyntheticDistillationManifest
        The same manifest.

    Raises
    ------
    DatasetContractError
        If a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    LicenceRecordError
        If the teacher's licence record is malformed.
    """
    if not isinstance(manifest, SyntheticDistillationManifest):
        raise E("a SyntheticDistillationManifest is required")
    require_choice(manifest.schema, "schema", {"synthetic-distillation-manifest.v1"}, E)
    require_sha256(manifest.teacher_artefact_digest, "teacher_artefact_digest")
    validate_licence_record(manifest.teacher_licence_record)
    require_text(manifest.verifier_grader_id, "verifier_grader_id", E)
    require_sha256(manifest.dedup_report_digest, "dedup_report_digest")
    return manifest


def validate_dedup_and_distillation(
    dedup: DatasetDeduplicationReport,
    manifest: SyntheticDistillationManifest | None = None,
) -> DatasetDeduplicationReport:
    """Gate a training dataset on its split audit and any synthetic teacher rows.

    Parameters
    ----------
    dedup : DatasetDeduplicationReport
        The dataset's split audit.
    manifest : SyntheticDistillationManifest or None, optional
        The manifest of its synthetic teacher rows, if any.

    Returns
    -------
    DatasetDeduplicationReport
        The same report.

    Raises
    ------
    SplitLeakageError
        If a cluster, near-duplicate window or template family is shared
        across partitions, or synthetic rows lie outside training.
    DatasetContractError
        If the report is malformed, it has no real training clusters, too
        few with synthetic rows, or the manifest names another report.
    OutputTrainingNotPermittedError
        If the teacher's licence does not permit training on its outputs.
    """
    validate_dedup_report(dedup)
    if dedup.cross_split_shared_cluster_count:
        raise SplitLeakageError("clusters are shared across partitions")
    if dedup.cross_split_near_duplicate_window_count:
        raise SplitLeakageError("near-duplicate windows are shared across partitions")
    if dedup.synthetic_non_train_row_count:
        raise SplitLeakageError("synthetic rows lie outside the training partition")
    if set(dedup.train_template_family_ids) & set(dedup.non_train_template_family_ids):
        raise SplitLeakageError("a template family is used on both sides of the split")
    if dedup.real_train_cluster_fraction <= 0.0:
        raise E("training needs real clusters")
    if manifest is None:
        return dedup
    validate_distillation_manifest(manifest)
    if (
        dedup.real_train_cluster_fraction
        < MIN_REAL_TRAIN_CLUSTER_FRACTION_WITH_SYNTHETIC
    ):
        raise E("with synthetic rows, at least a quarter of training clusters are real")
    if manifest.dedup_report_digest != record_digest(dedup):
        raise E("the manifest names another dedup report")
    require_output_training_permitted(manifest.teacher_licence_record)
    return dedup
