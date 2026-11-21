# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Evaluator Outputs

"""Evaluator outputs and sealed look results.

An :class:`EvaluatorOutput` is what a registered evaluator emits for one
look: per-slice aggregate estimates with intervals, or a test suite's pass
count and run count, never item-level data. A :class:`SealedLookResult`
wraps it after custody journalled the look, and binds it to that released
journal entry by record digest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._checks import (
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
)
from .custody_records import CustodyIngestRecord
from .digests import record_digest
from .errors import EvaluationContractError
from .look_openings import require_opening_custodied
from .sealed_looks import SealedLookJournalEntry, validate_journal_entry

EvaluatorOutputMode = Literal["slice_estimates", "test_suite_counts"]


@dataclass(frozen=True, slots=True)
class SliceAggregateEstimate:
    """One metric's aggregate estimate and interval on one slice."""

    artefact_digest: str
    metric_id: str
    slice_id: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    cluster_count: int


@dataclass(frozen=True, slots=True)
class EvaluatorOutput:
    """What a registered evaluator emits for one look.

    Exactly one form is filled: slice estimates, or a test suite's pass
    count and run count.
    """

    schema: Literal["evaluator-output.v1"]
    output_mode: EvaluatorOutputMode
    slice_estimates: tuple[SliceAggregateEstimate, ...]
    passed_count: int | None
    run_count: int | None


@dataclass(frozen=True, slots=True)
class SealedLookResult:
    """A released look's evaluator output, bound to its journal entry."""

    schema: Literal["sealed-look-result.v1"]
    look_journal_digest: str
    derived_proposer_lineage_id: str
    split_digest: str
    evaluator_output: EvaluatorOutput


def validate_slice_estimate(estimate: SliceAggregateEstimate) -> SliceAggregateEstimate:
    """Check one slice estimate: finite values with the point inside its interval.

    Parameters
    ----------
    estimate : SliceAggregateEstimate
        The estimate to check.

    Returns
    -------
    SliceAggregateEstimate
        The same estimate.

    Raises
    ------
    EvaluationContractError
        If a value is malformed or outside its interval.
    """
    if not isinstance(estimate, SliceAggregateEstimate):
        raise EvaluationContractError("a SliceAggregateEstimate is required")
    require_sha256(estimate.artefact_digest, "artefact_digest")
    require_text(estimate.metric_id, "metric_id", EvaluationContractError)
    require_text(estimate.slice_id, "slice_id", EvaluationContractError)
    low = require_finite(estimate.ci_lower, "ci_lower", EvaluationContractError)
    point = require_finite(
        estimate.point_estimate, "point_estimate", EvaluationContractError
    )
    high = require_finite(estimate.ci_upper, "ci_upper", EvaluationContractError)
    if not low <= point <= high:
        raise EvaluationContractError("point_estimate must lie inside its interval")
    require_int(
        estimate.cluster_count, "cluster_count", EvaluationContractError, minimum=1
    )
    return estimate


def validate_evaluator_output(output: EvaluatorOutput) -> EvaluatorOutput:
    """Check that an evaluator output fills exactly one aggregate form.

    Parameters
    ----------
    output : EvaluatorOutput
        The output to check.

    Returns
    -------
    EvaluatorOutput
        The same output.

    Raises
    ------
    EvaluationContractError
        If both or neither form is filled, or a count is out of range.
    """
    if not isinstance(output, EvaluatorOutput):
        raise EvaluationContractError("an EvaluatorOutput is required")
    require_choice(
        output.schema, "schema", {"evaluator-output.v1"}, EvaluationContractError
    )
    mode = require_choice(
        output.output_mode,
        "output_mode",
        {"slice_estimates", "test_suite_counts"},
        EvaluationContractError,
    )
    estimates = require_tuple(
        output.slice_estimates, "slice_estimates", EvaluationContractError
    )
    if mode == "slice_estimates":
        if (
            not estimates
            or output.passed_count is not None
            or output.run_count is not None
        ):
            raise EvaluationContractError(
                "slice_estimates mode carries estimates and no counts"
            )
        for estimate in estimates:
            if not isinstance(estimate, SliceAggregateEstimate):
                raise EvaluationContractError(
                    "slice_estimates must hold SliceAggregateEstimate records"
                )
            validate_slice_estimate(estimate)
        return output
    if estimates:
        raise EvaluationContractError(
            "test_suite_counts mode carries no slice estimates"
        )
    runs = require_int(
        output.run_count, "run_count", EvaluationContractError, minimum=1
    )
    passed = require_int(
        output.passed_count, "passed_count", EvaluationContractError, minimum=0
    )
    if passed > runs:
        raise EvaluationContractError("passed_count cannot exceed run_count")
    return output


def validate_look_result(
    result: SealedLookResult,
    entry: SealedLookJournalEntry,
    *,
    opening_record: CustodyIngestRecord,
) -> SealedLookResult:
    """Check that a result belongs to a released look whose opening custody held.

    Parameters
    ----------
    result : SealedLookResult
        The result to check.
    entry : SealedLookJournalEntry
        The journal entry it claims.
    opening_record : CustodyIngestRecord
        Custody's record of the look's opening, which must be ingested no
        later than the look's release.

    Returns
    -------
    SealedLookResult
        The same result.

    Raises
    ------
    EvaluationContractError
        If the result is malformed, the entry was not released, the result's
        digests, lineage, split or artefacts differ from the entry, or the
        look's opening was not in custody by its release.
    CustodyRecordError
        If the opening's record is malformed.
    """
    validate_journal_entry(entry)
    if (
        not isinstance(result, SealedLookResult)
        or result.schema != "sealed-look-result.v1"
    ):
        raise EvaluationContractError("a sealed-look-result.v1 record is required")
    if entry.status != "released":
        raise EvaluationContractError("only a released look has a result")
    if result.look_journal_digest != record_digest(entry):
        raise EvaluationContractError("the result names another journal entry")
    if (result.derived_proposer_lineage_id, result.split_digest) != (
        entry.derived_proposer_lineage_id,
        entry.split_digest,
    ):
        raise EvaluationContractError(
            "the result's lineage or split differs from its entry"
        )
    output = validate_evaluator_output(result.evaluator_output)
    if any(
        item.artefact_digest not in entry.frozen_artefact_digests
        for item in output.slice_estimates
    ):
        raise EvaluationContractError(
            "the result reports an artefact the look did not freeze"
        )
    require_opening_custodied(entry, opening_record)
    return result
