# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Promotion Export Checks

"""The export parity reports a promotion's serving binaries must carry.

A promotion needs at least one report. Each is of the candidate, pinned as a
witness, and covers every ``(slice, metric)`` pair the confirmatory look
reported for the candidate; a look that reports test-suite counts has no
slices to cover. Every non-rate slice's allowed margin spans at least three
noise floors (E-4). Rate-metric slices are not held to three flip-rate
ceilings: a smaller allowed margin only makes an export harder to pass. For
a study with a dynamics evaluation contract, every report's rollout horizon
reaches the contract's.
"""

from __future__ import annotations

from collections.abc import Sequence

from .digests import record_digest
from .errors import PromotionDecisionError
from .evaluator_outputs import SealedLookResult
from .export_contracts import validate_export_parity_report
from .promotion_evidence import spans_noise_floors
from .promotion_records import PromotionDependencyPins
from .study_contracts import validate_dynamics_evaluation_contract
from .study_records import (
    DynamicsEvaluationContract,
    ExportParityReport,
    PreregistrationBuild2Terms,
)

E = PromotionDecisionError


def _dynamics_horizon(
    terms: PreregistrationBuild2Terms, contract: DynamicsEvaluationContract | None
) -> int | None:
    """Return the rollout horizon a dynamics study's exports must reach."""
    if terms.dynamics_eval_contract_digest is None:
        if contract is not None:
            raise E("the terms name no dynamics evaluation contract")
        return None
    if contract is None:
        raise E("the terms' dynamics evaluation contract is required")
    validate_dynamics_evaluation_contract(contract)
    if record_digest(contract) != terms.dynamics_eval_contract_digest:
        raise E("the dynamics evaluation contract is not the terms'")
    return contract.onnx_parity_rollout_horizon_h


def check_promotion_exports(
    reports: Sequence[ExportParityReport],
    *,
    candidate: str,
    pins: PromotionDependencyPins,
    look_result: SealedLookResult,
    terms: PreregistrationBuild2Terms,
    dynamics_contract: DynamicsEvaluationContract | None,
) -> None:
    """Check a promoted candidate's export reports against its look and pins.

    Parameters
    ----------
    reports : Sequence[ExportParityReport]
        The candidate's serving-binary parity reports.
    candidate : str
        The promoted candidate's content digest.
    pins : PromotionDependencyPins
        The promotion's validated dependency pins.
    look_result : SealedLookResult
        The confirmatory look's bound result.
    terms : PreregistrationBuild2Terms
        The study's validated build-2 terms.
    dynamics_contract : DynamicsEvaluationContract or None
        The terms' dynamics contract, when they name one.

    Raises
    ------
    PromotionDecisionError
        If there is no report, a report is of another parent, is not a pinned
        witness or misses a look slice of the candidate, a non-rate slice's
        allowed margin spans fewer than three noise floors, a dynamics
        export's horizon falls short, or the dynamics contract is missing,
        stray or not the terms'.
    ExportParityError, EvaluationContractError
        If a report or the dynamics contract is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    horizon = _dynamics_horizon(terms, dynamics_contract)
    if not reports:
        raise E("a promotion needs at least one export parity report")
    required = {
        (item.slice_id, item.metric_id)
        for item in look_result.evaluator_output.slice_estimates
        if item.artefact_digest == candidate
    }
    floors = dict(terms.non_rate_metric_noise_floors)
    for report in reports:
        validate_export_parity_report(report)
        if report.parent_content_digest != candidate:
            raise E("an export report is of another parent than the candidate")
        if record_digest(report) not in pins.witness_and_parity_digests:
            raise E("an export report is not among the pinned witnesses")
        if not required <= {(s.slice_id, s.metric_id) for s in report.slice_checks}:
            raise E(
                "an export report does not cover the look's slices of the candidate"
            )
        for check in report.slice_checks:
            floor = floors.get(check.metric_id)
            if floor is not None and not spans_noise_floors(
                check.allowed_margin, floor
            ):
                raise E("E-4: an export slice's allowed margin spans < 3 noise floors")
        if horizon is not None and (
            report.dynamics_rollout_horizon_h is None
            or report.dynamics_rollout_horizon_h < horizon
        ):
            raise E("a dynamics export's rollout horizon is below the contract's")
