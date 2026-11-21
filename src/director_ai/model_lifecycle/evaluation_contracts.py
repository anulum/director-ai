# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Evaluation Contracts

"""Graders, sealed split pools and preregistration records.

A preregistration freezes, before any training, what a study will compare
and how: its arms, search budgets, sealed split pool, graders, margins,
sample size, interval method, pass rule, the full list of artefacts its
looks will evaluate, its analysis code and the base checkpoints it builds
on. Its custody ingest is the anchor that "before" refers to.

A sealed grader may not also train or select. A grader that is not exact
and deterministic needs a calibration report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_choice,
    require_digests,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
    require_utc_timestamp,
)
from .errors import EvaluationContractError
from .graders import GraderSpec, validate_grader

CiMethod = Literal[
    "hierarchical_cluster_bootstrap", "wilson_score", "paired_exact_mcnemar"
]
CI_METHODS: frozenset[str] = frozenset(
    {"hierarchical_cluster_bootstrap", "wilson_score", "paired_exact_mcnemar"}
)
MINIMUM_POOL_SPLITS = 3
"""A sealed split pool holds at least this many splits."""


@dataclass(frozen=True, slots=True)
class SealedSplitPool:
    """A pool of pre-hashed sealed splits a study may look at.

    Attributes
    ----------
    pool_id : str
        Pool identifier.
    split_digests : tuple[str, ...]
        Content digests of at least three sealed splits.
    deduplication_report_digest : str
        SHA-256 of the report that checked the splits are disjoint.
    """

    pool_id: str
    split_digests: tuple[str, ...]
    deduplication_report_digest: str


@dataclass(frozen=True, slots=True)
class PreregistrationRecord:
    """A study's frozen plan, ingested by custody before any training.

    Attributes
    ----------
    schema : Literal["lifecycle-preregistration.v1"]
        Record schema.
    study_id : str
        Study identifier.
    declared_arms : tuple[str, ...]
        Names of the compared arms.
    search_spaces_and_budgets_digest : str
        SHA-256 of the search spaces and budgets.
    split_pool : SealedSplitPool
        The sealed split pool.
    graders : tuple[GraderSpec, ...]
        Every grader and its roles.
    non_inferiority_margins_bps : tuple[tuple[str, int], ...]
        Per-metric margins in basis points.
    planned_sample_size : int
        Planned paired sample size.
    target_power_permille : int
        Target power in permille.
    ci_method : CiMethod
        Interval method.
    resampling_unit : str
        The independent resampling unit.
    pass_rule_digest : str
        SHA-256 of the pass rule.
    frozen_look_candidate_digests : tuple[str, ...]
        Every artefact a look evaluates: arms, controls and exports.
    analysis_code_digest : str
        SHA-256 of the analysis code.
    declared_registered_at_utc : str
        Informational; ordering uses custody sequences.
    base_checkpoint_digests : tuple[str, ...]
        Unmodified base checkpoints the arms build on.
    """

    schema: Literal["lifecycle-preregistration.v1"]
    study_id: str
    declared_arms: tuple[str, ...]
    search_spaces_and_budgets_digest: str
    split_pool: SealedSplitPool
    graders: tuple[GraderSpec, ...]
    non_inferiority_margins_bps: tuple[tuple[str, int], ...]
    planned_sample_size: int
    target_power_permille: int
    ci_method: CiMethod
    resampling_unit: str
    pass_rule_digest: str
    frozen_look_candidate_digests: tuple[str, ...]
    analysis_code_digest: str
    declared_registered_at_utc: str
    base_checkpoint_digests: tuple[str, ...]


def validate_sealed_split_pool(pool: SealedSplitPool) -> SealedSplitPool:
    """Check a sealed split pool's shape.

    Parameters
    ----------
    pool : SealedSplitPool
        The pool to check.

    Returns
    -------
    SealedSplitPool
        The same pool.

    Raises
    ------
    EvaluationContractError
        If it holds fewer than three splits or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(pool, SealedSplitPool):
        raise EvaluationContractError("a SealedSplitPool is required")
    require_text(pool.pool_id, "pool_id", EvaluationContractError)
    splits = require_digests(
        pool.split_digests, "split_digests", EvaluationContractError, non_empty=True
    )
    if len(splits) < MINIMUM_POOL_SPLITS:
        raise EvaluationContractError(
            f"a sealed split pool needs at least {MINIMUM_POOL_SPLITS} splits"
        )
    require_sha256(pool.deduplication_report_digest, "deduplication_report_digest")
    return pool


def validate_preregistration(record: PreregistrationRecord) -> PreregistrationRecord:
    """Check a preregistration record completely.

    Parameters
    ----------
    record : PreregistrationRecord
        The record to check.

    Returns
    -------
    PreregistrationRecord
        The same record.

    Raises
    ------
    EvaluationContractError
        If a field is malformed, no grader is sealed only, or a sealed
        grader also trains or selects.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(record, PreregistrationRecord):
        raise EvaluationContractError("a PreregistrationRecord is required")
    require_choice(
        record.schema,
        "schema",
        {"lifecycle-preregistration.v1"},
        EvaluationContractError,
    )
    require_text(record.study_id, "study_id", EvaluationContractError)
    arms = require_tuple(record.declared_arms, "declared_arms", EvaluationContractError)
    if not arms:
        raise EvaluationContractError("declared_arms must not be empty")
    for arm in arms:
        require_text(arm, "declared_arms", EvaluationContractError)
    require_unique(arms, "declared_arms", EvaluationContractError)
    require_sha256(
        record.search_spaces_and_budgets_digest, "search_spaces_and_budgets_digest"
    )
    validate_sealed_split_pool(record.split_pool)
    graders = require_tuple(record.graders, "graders", EvaluationContractError)
    for grader in graders:
        if not isinstance(grader, GraderSpec):
            raise EvaluationContractError("graders must hold GraderSpec records")
        validate_grader(grader)
    require_unique(
        [g.grader_id for g in graders if isinstance(g, GraderSpec)],
        "grader ids",
        EvaluationContractError,
    )
    sealed = [g for g in graders if isinstance(g, GraderSpec) and "sealed" in g.roles]
    if not sealed:
        raise EvaluationContractError("a preregistration needs a sealed grader")
    if any({"training", "selection"} & set(g.roles) for g in sealed):
        raise EvaluationContractError("a sealed grader may not also train or select")
    margins = require_tuple(
        record.non_inferiority_margins_bps,
        "non_inferiority_margins_bps",
        EvaluationContractError,
    )
    if not margins:
        raise EvaluationContractError("non_inferiority_margins_bps must not be empty")
    metric_ids: list[str] = []
    for margin in margins:
        if not isinstance(margin, tuple) or len(margin) != 2:
            raise EvaluationContractError("each margin is a (metric_id, bps) pair")
        metric_ids.append(
            require_text(margin[0], "margin metric_id", EvaluationContractError)
        )
        require_int(margin[1], "margin bps", EvaluationContractError, minimum=1)
    require_unique(metric_ids, "margin metric ids", EvaluationContractError)
    require_int(
        record.planned_sample_size,
        "planned_sample_size",
        EvaluationContractError,
        minimum=1,
    )
    power = require_int(
        record.target_power_permille,
        "target_power_permille",
        EvaluationContractError,
        minimum=1,
    )
    if power >= 1000:
        raise EvaluationContractError("target_power_permille must be below 1000")
    require_choice(record.ci_method, "ci_method", CI_METHODS, EvaluationContractError)
    require_text(record.resampling_unit, "resampling_unit", EvaluationContractError)
    require_sha256(record.pass_rule_digest, "pass_rule_digest")
    require_digests(
        record.frozen_look_candidate_digests,
        "frozen_look_candidate_digests",
        EvaluationContractError,
        non_empty=True,
    )
    require_sha256(record.analysis_code_digest, "analysis_code_digest")
    require_utc_timestamp(
        record.declared_registered_at_utc,
        "declared_registered_at_utc",
        EvaluationContractError,
    )
    require_digests(
        record.base_checkpoint_digests,
        "base_checkpoint_digests",
        EvaluationContractError,
        non_empty=False,
    )
    return record
