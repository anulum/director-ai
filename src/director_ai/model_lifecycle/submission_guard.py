# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Job Submission Guard

"""The last check every compute driver runs before it submits a job.

A driver holds a spend port and calls :func:`guard_job_submission` as its
final step before any network or IPC submission, then submits with the
projection the guard returned. The guard never takes a live reservation
from the caller: it queries the ledger itself, so a reservation settled
after the caller looked is refused before anything is sent.

Before that query the guard binds the job together. The funding kind is
the recipe's, the estimate is of the recipe, and a paid run carries the
owner's approval of exactly this estimate and binding, while no other run
carries one. Every ``quantum_variational`` recipe funded by cloud credit, a
granted tier or a paid run (every class but local owned hardware) needs a
barren-plateau preflight witness that passed and names the recipe's family
contract. The rule follows the family, not an execution mode the caller
could misstate.
"""

from __future__ import annotations

from ._checks import require_choice
from .architecture_specs import POOL_FUNDING_KINDS, PoolFundingKind
from .digests import record_digest
from .errors import QuantumAcceptanceError, SpendBindingError
from .family_witnesses import validate_barren_plateau_receipt
from .ports_compute import ComputeAndSpendPort
from .quantum_records import BarrenPlateauWitnessReceipt
from .recipe_contracts import validate_recipe_and_curriculum
from .recipe_records import TrainingRecipeManifest
from .spend_contracts import (
    validate_cost_estimate,
    validate_paid_run_approval,
    validate_spend_binding,
    validate_spend_reservation_live,
)
from .spend_records import (
    CostEstimateRecord,
    LiveReservationProjection,
    PaidRunApprovalRecord,
    SpendReservationBinding,
)

E = SpendBindingError
LOCAL_FUNDING = "local_owned_hardware"
"""Class A: the only funding class that needs no preflight witness."""
PAID_FUNDING = "per_run_owner_approved_paid"
"""Class C: the only funding class that needs the owner's approval."""


def _check_paid_approval(
    funding: str,
    paid_approval: PaidRunApprovalRecord | None,
    cost_estimate: CostEstimateRecord,
    spend_binding: SpendReservationBinding,
) -> None:
    """Require the owner's approval of exactly this run, and only for a paid run."""
    if funding != PAID_FUNDING:
        if paid_approval is not None:
            raise E("only a paid run carries a paid-run approval")
        return
    if paid_approval is None:
        raise E("a paid run needs the owner's paid-run approval")
    validate_paid_run_approval(paid_approval)
    if paid_approval.cost_estimate_digest != record_digest(cost_estimate):
        raise E("the paid-run approval is of another cost estimate")
    if paid_approval.spend_binding != spend_binding:
        raise E("the paid-run approval is of another spend binding")


def _check_preflight(
    funding: str,
    recipe: TrainingRecipeManifest,
    receipt: BarrenPlateauWitnessReceipt | None,
) -> None:
    """Require a passed preflight of a credit-funded or paid variational job."""
    if receipt is not None:
        validate_barren_plateau_receipt(receipt)
        if receipt.companion_digest != recipe.family_contract_digest:
            raise QuantumAcceptanceError(
                "the preflight witness names another family contract"
            )
    if recipe.family != "quantum_variational" or funding == LOCAL_FUNDING:
        return
    if receipt is None:
        raise QuantumAcceptanceError(
            "a credit-funded or paid variational job needs its preflight witness"
        )
    if receipt.passed is not True:
        raise QuantumAcceptanceError("the barren-plateau preflight did not pass")


def guard_job_submission(
    spend_port: ComputeAndSpendPort,
    *,
    recipe: TrainingRecipeManifest,
    funding_kind: PoolFundingKind,
    cost_estimate: CostEstimateRecord,
    spend_binding: SpendReservationBinding,
    paid_approval: PaidRunApprovalRecord | None,
    barren_plateau_receipt: BarrenPlateauWitnessReceipt | None,
) -> LiveReservationProjection:
    """Check a job just before submission and return the live reservation checked.

    Parameters
    ----------
    spend_port : ComputeAndSpendPort
        The driver's own spend port, queried here for the live reservation.
    recipe : TrainingRecipeManifest
        The job's recipe.
    funding_kind : PoolFundingKind
        The job's funding class, which must be the recipe's.
    cost_estimate : CostEstimateRecord
        The job's cost estimate, of this recipe.
    spend_binding : SpendReservationBinding
        The reservation the job spends against.
    paid_approval : PaidRunApprovalRecord or None
        The owner's approval, exactly for a paid run.
    barren_plateau_receipt : BarrenPlateauWitnessReceipt or None
        The preflight witness, required for a credit-funded or paid
        variational quantum job.

    Returns
    -------
    LiveReservationProjection
        The live projection the guard read and checked; the driver submits
        with it.

    Raises
    ------
    SpendBindingError
        If the funding, the estimate or the approval does not bind to the
        job, or the live reservation does not cover it.
    QuantumAcceptanceError
        If a required preflight witness is missing, malformed, of another
        family contract, or did not pass.
    RecipeContractError
        If the recipe is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    validate_recipe_and_curriculum(recipe)
    funding = require_choice(funding_kind, "funding_kind", POOL_FUNDING_KINDS, E)
    if funding != recipe.funding_kind:
        raise E("the job's funding kind is not its recipe's")
    validate_cost_estimate(cost_estimate)
    if cost_estimate.recipe_digest != record_digest(recipe):
        raise E("the cost estimate is of another recipe")
    validate_spend_binding(spend_binding)
    _check_paid_approval(funding, paid_approval, cost_estimate, spend_binding)
    _check_preflight(funding, recipe, barren_plateau_receipt)
    live = spend_port.verify_live_reservation(binding=spend_binding)
    validate_spend_reservation_live(spend_binding, live, cost_estimate)
    return live
