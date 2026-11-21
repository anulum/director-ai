# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Compute and Spend Ports

"""Ports that run compute jobs and reserve, verify and settle their spend.

A spend port talks to the SpendLedger. It never returns an empty
reservation id: a refused reservation raises ``SpendReservationRefusedError``
and a refused settlement ``SpendSettlementRefusedError``. A compute driver
never takes a live reservation from its caller. It holds a spend port, and
the last thing it does before any network or IPC submission is
``guard_job_submission``, which queries and checks the live reservation
itself; the driver then submits with the projection the guard returned,
never with a snapshot of its own. Adapters outside the kernel implement
these ports.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .architecture_specs import PoolFundingKind
from .quantum_records import BarrenPlateauWitnessReceipt
from .recipe_records import TrainingRecipeManifest
from .spend_records import (
    CloudJobCancelReceipt,
    CloudJobStatusReport,
    CloudJobSubmissionReceipt,
    CostEstimateRecord,
    LiveReservationProjection,
    PaidRunApprovalRecord,
    SpendReservationBinding,
)


@runtime_checkable
class ComputeAndSpendPort(Protocol):
    """Reserves, live-verifies, and settles exposure with typed refusal exceptions."""

    def reserve(
        self,
        *,
        binding: SpendReservationBinding,
        unit: str,
        tax: str,
        price_revision: str,
        upper_bound: str,
        depth: int,
        wall_seconds: int,
    ) -> str:
        """Reserve exposure for one job and return the reservation id.

        Parameters
        ----------
        binding : SpendReservationBinding
            The binding the reservation is for.
        unit : str
            The money unit.
        tax : str
            The tax treatment.
        price_revision : str
            The price sheet's revision.
        upper_bound : str
            The requested upper bound, as plain decimal text.
        depth : int
            The reservation depth.
        wall_seconds : int
            The job's wall-clock cap.

        Returns
        -------
        str
            The reservation id, never empty.

        Raises
        ------
        SpendReservationRefusedError
            If the ledger refuses the reservation.
        """
        ...

    def verify_live_reservation(
        self,
        *,
        binding: SpendReservationBinding,
    ) -> LiveReservationProjection:
        """Return the ledger's live view of the binding's reservation.

        Parameters
        ----------
        binding : SpendReservationBinding
            The binding.

        Returns
        -------
        LiveReservationProjection
            The live projection, with the grant-folded upper bound, minimum
            charge and fixed fee.
        """
        ...

    def settle(
        self,
        *,
        binding: SpendReservationBinding,
        actual_amount: str,
        final: bool,
    ) -> None:
        """Settle an amount against the binding's reservation.

        Parameters
        ----------
        binding : SpendReservationBinding
            The binding.
        actual_amount : str
            The amount spent, as plain decimal text.
        final : bool
            Whether this settlement closes the reservation.

        Raises
        ------
        SpendSettlementRefusedError
            If the ledger refuses the settlement.
        """
        ...


@runtime_checkable
class CloudComputeDriverPort(Protocol):
    """Submits, polls, and cancels compute jobs; verifies live reservation internally."""

    @property
    def driver_name(self) -> str:
        """The driver's name, as cost estimates and receipts record it."""
        ...

    def estimate_cost(
        self,
        *,
        recipe: TrainingRecipeManifest,
    ) -> CostEstimateRecord:
        """Estimate a recipe's expected cost and its hard upper bound.

        Parameters
        ----------
        recipe : TrainingRecipeManifest
            The recipe.

        Returns
        -------
        CostEstimateRecord
            The estimate.
        """
        ...

    def submit_job(
        self,
        *,
        recipe: TrainingRecipeManifest,
        funding_kind: PoolFundingKind,
        cost_estimate: CostEstimateRecord,
        spend_binding: SpendReservationBinding,
        owner_live_deploy_approval_ref: str | None,
        paid_approval: PaidRunApprovalRecord | None,
        barren_plateau_receipt: BarrenPlateauWitnessReceipt | None = None,
        dry_run: bool = True,
    ) -> CloudJobSubmissionReceipt:
        """Submit a job after ``guard_job_submission`` passes, as the last step.

        Parameters
        ----------
        recipe : TrainingRecipeManifest
            The recipe.
        funding_kind : PoolFundingKind
            The funding class of the job.
        cost_estimate : CostEstimateRecord
            The job's cost estimate.
        spend_binding : SpendReservationBinding
            The reservation the job spends against.
        owner_live_deploy_approval_ref : str or None
            The owner's live deployment approval, when one is needed.
        paid_approval : PaidRunApprovalRecord or None
            The owner's approval of a paid run.
        barren_plateau_receipt : BarrenPlateauWitnessReceipt or None
            The preflight witness a credit-funded or paid variational
            quantum job needs.
        dry_run : bool
            Whether to stop before the real submission, the default.

        Returns
        -------
        CloudJobSubmissionReceipt
            The submission receipt.
        """
        ...

    def poll_status(self, *, job_id: str) -> CloudJobStatusReport:
        """Return a job's status.

        Parameters
        ----------
        job_id : str
            The job.

        Returns
        -------
        CloudJobStatusReport
            Its status.
        """
        ...

    def cancel_job(
        self,
        *,
        job_id: str,
        reason: str,
    ) -> CloudJobCancelReceipt:
        """Cancel a job.

        Parameters
        ----------
        job_id : str
            The job.
        reason : str
            Why it is cancelled.

        Returns
        -------
        CloudJobCancelReceipt
            The cancellation receipt.
        """
        ...
