# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Spend Validators

"""Validators of spend bindings, cost estimates, paid-run approvals and live cover.

Money is plain decimal text such as ``45.00``, the only spelling a ledger
writes, read exactly. Sums and differences are taken in a 50-digit decimal
context that traps invalid operations, any mixing with binary floats and
any inexact result, so no comparison of cover ever rounds. A job may start only
on a reservation the ledger reports as found, open and not overrun, that
matches the binding exactly, in the estimate's unit, whose remaining
exposure covers the larger of the hard upper bound and the minimum charge
plus the fixed fee, and whose unsettled requested amount covers the hard
upper bound, which must be above zero.
"""

from __future__ import annotations

import re
from decimal import (
    Context,
    Decimal,
    FloatOperation,
    Inexact,
    InvalidOperation,
    localcontext,
)

from ._checks import (
    require_choice,
    require_int,
    require_sha256,
    require_text,
    require_utc_timestamp,
)
from .errors import SpendBindingError
from .spend_records import (
    SCHEDULING_STRATEGIES,
    CostEstimateRecord,
    LiveReservationProjection,
    PaidRunApprovalRecord,
    SpendReservationBinding,
)

E = SpendBindingError
MONEY_CONTEXT = Context(prec=50, traps=[InvalidOperation, FloatOperation, Inexact])
"""The decimal context of every sum and difference; a rounded result is refused."""
PLAIN_AMOUNT = re.compile(r"(0|[1-9][0-9]*)(\.[0-9]+)?")
"""The one amount spelling read: digits, an optional fraction, nothing else."""


def amount(value: object, field: str) -> Decimal:
    """Return a non-negative decimal amount read exactly from plain decimal text.

    Parameters
    ----------
    value : object
        The amount as plain decimal text, such as ``45.00``.
    field : str
        The field name, for error messages.

    Returns
    -------
    Decimal
        The amount.

    Raises
    ------
    SpendBindingError
        If the value is not plain decimal text: a sign, an exponent, a digit
        separator, a leading zero, NaN or an infinity is refused.
    """
    text = require_text(value, field, E)
    if PLAIN_AMOUNT.fullmatch(text) is None:
        raise E(f"{field} must be plain decimal text such as 45.00")
    return Decimal(text)


def validate_spend_binding(binding: SpendReservationBinding) -> SpendReservationBinding:
    """Check a binding to a SpendLedger reservation.

    Parameters
    ----------
    binding : SpendReservationBinding
        The binding to check.

    Returns
    -------
    SpendReservationBinding
        The same binding.

    Raises
    ------
    SpendBindingError
        If a field is missing, including the task and the operation.
    """
    if not isinstance(binding, SpendReservationBinding):
        raise E("a SpendReservationBinding is required")
    require_choice(binding.schema, "schema", {"spend-reservation-binding.v1"}, E)
    for name in (
        "pool_id",
        "reservation_id",
        "caller",
        "project",
        "task",
        "operation",
        "idempotency_key",
    ):
        require_text(getattr(binding, name), name, E)
    return binding


def validate_cost_estimate(estimate: CostEstimateRecord) -> CostEstimateRecord:
    """Check a cost estimate; its expected cost must not exceed its hard bound.

    Parameters
    ----------
    estimate : CostEstimateRecord
        The estimate to check.

    Returns
    -------
    CostEstimateRecord
        The same estimate.

    Raises
    ------
    SpendBindingError
        If an amount is malformed, the expected cost exceeds the hard bound,
        or another field is malformed.
    MalformedDigestError
        If the recipe digest is malformed.
    """
    if not isinstance(estimate, CostEstimateRecord):
        raise E("a CostEstimateRecord is required")
    require_choice(estimate.schema, "schema", {"training-cost-estimate.v1"}, E)
    require_sha256(estimate.recipe_digest, "recipe_digest")
    for name in (
        "driver_name",
        "accelerator_type",
        "unit",
        "price_source_and_revision",
    ):
        require_text(getattr(estimate, name), name, E)
    for name in ("accelerator_count", "replica_count", "max_wall_seconds"):
        require_int(getattr(estimate, name), name, E, minimum=1)
    require_choice(
        estimate.scheduling_strategy, "scheduling_strategy", SCHEDULING_STRATEGIES, E
    )
    expected = amount(estimate.estimated_expected_cost, "estimated_expected_cost")
    if expected > amount(estimate.hard_upper_bound_cost, "hard_upper_bound_cost"):
        raise E("the expected cost exceeds the hard upper bound")
    require_utc_timestamp(estimate.estimated_at_utc, "estimated_at_utc", E)
    return estimate


def validate_paid_run_approval(
    approval: PaidRunApprovalRecord,
) -> PaidRunApprovalRecord:
    """Check an owner's approval of one paid run.

    Parameters
    ----------
    approval : PaidRunApprovalRecord
        The approval to check.

    Returns
    -------
    PaidRunApprovalRecord
        The same approval.

    Raises
    ------
    SpendBindingError
        If a field is malformed or the binding names another pool.
    MalformedDigestError
        If the cost estimate digest is malformed.
    """
    if not isinstance(approval, PaidRunApprovalRecord):
        raise E("a PaidRunApprovalRecord is required")
    require_choice(approval.schema, "schema", {"paid-run-approval.v1"}, E)
    require_sha256(approval.cost_estimate_digest, "cost_estimate_digest")
    require_text(approval.owner_approval_ref, "owner_approval_ref", E)
    require_text(approval.pool_id, "pool_id", E)
    if validate_spend_binding(approval.spend_binding).pool_id != approval.pool_id:
        raise E("the approval's binding names another pool")
    return approval


def validate_spend_reservation_live(
    binding: SpendReservationBinding,
    live: LiveReservationProjection,
    estimate: CostEstimateRecord,
) -> LiveReservationProjection:
    """Refuse a job unless the ledger's live reservation covers its hard bound.

    Parameters
    ----------
    binding : SpendReservationBinding
        The job's binding.
    live : LiveReservationProjection
        The ledger's live view of the reservation.
    estimate : CostEstimateRecord
        The job's cost estimate.

    Returns
    -------
    LiveReservationProjection
        The same projection.

    Raises
    ------
    SpendBindingError
        If the reservation is not found, not open or overrun, differs from
        the binding, is in another unit, does not cover the hard bound, or
        an amount needs more precision than the cover check holds.
    """
    validate_spend_binding(binding)
    validate_cost_estimate(estimate)
    if not isinstance(live, LiveReservationProjection):
        raise E("a LiveReservationProjection is required")
    if live.found is not True or live.state != "open" or live.overrun_open is not False:
        raise E("the reservation must be found, open and not overrun")
    for name in ("pool_id", "reservation_id", "project", "task", "operation"):
        if getattr(live, name) != getattr(binding, name):
            raise E(f"the live reservation's {name} differs from the binding")
    if live.unit != estimate.unit:
        raise E("the live reservation is in another unit than the estimate")
    hard = amount(estimate.hard_upper_bound_cost, "hard_upper_bound_cost")
    if hard <= 0:
        raise E("the hard upper bound must be above zero")
    minimum = amount(live.minimum_charge, "minimum_charge")
    fee = amount(live.fixed_fee, "fixed_fee")
    remaining = amount(live.remaining_exposure, "remaining_exposure")
    requested = amount(live.requested_upper, "requested_upper")
    settled = amount(live.settled, "settled")
    try:
        with localcontext(MONEY_CONTEXT):
            floor = max(hard, minimum) + fee
            unsettled = requested - settled
    except Inexact as error:
        raise E("an amount needs more precision than the cover check holds") from error
    if remaining < floor:
        raise E(
            "the remaining exposure does not cover the bound, minimum charge and fee"
        )
    if unsettled < hard:
        raise E("the unsettled requested amount does not cover the hard bound")
    return live
