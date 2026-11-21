# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Spend and Job Records

"""Records that bind a training job to money and to its execution.

A job is bound to an open SpendLedger reservation
(:class:`SpendReservationBinding`), checked against the ledger's live view of
it (:class:`LiveReservationProjection`) and against a cost estimate with a
hard upper bound. A paid run also carries the owner's approval. A submitted
job leaves a submission receipt with its sandbox confinement, status reports
and a cancellation receipt. The rules are in :mod:`.spend_contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .architecture_specs import PoolFundingKind

RESERVATION_STATES: frozenset[str] = frozenset(
    {"open", "settled", "reconciled", "revoked", "expired"}
)
SCHEDULING_STRATEGIES: frozenset[str] = frozenset(
    {"LOCAL", "SPOT", "FLEX_START", "STANDARD"}
)
NETWORK_ISOLATION_LAYERS: frozenset[str] = frozenset(
    {"landlock_v4", "nftables_skuid", "cloud_vpc_disabled_egress"}
)
JOB_STATES: frozenset[str] = frozenset(
    {"queued", "running", "succeeded", "failed", "cancelled", "preempted"}
)


@dataclass(frozen=True, slots=True)
class SpendReservationBinding:
    """Binding to an active SpendLedger reservation for local or cloud compute."""

    schema: Literal["spend-reservation-binding.v1"]
    pool_id: str
    reservation_id: str
    caller: str
    project: str
    task: str
    operation: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class LiveReservationProjection:
    """Typed mirror of SpendLedger.verify_reservation output."""

    found: bool
    reservation_id: str | None
    pool_id: str | None
    state: Literal["open", "settled", "reconciled", "revoked", "expired"] | None
    exposure: str | None
    requested_upper: str | None
    settled: str | None
    remaining_exposure: str | None
    minimum_charge: str | None
    fixed_fee: str | None
    unit: str | None
    epoch: int | None
    window: str | None
    seat: str | None
    project: str | None
    task: str | None
    operation: str | None
    expires_at: str | None
    overrun_open: bool | None


@dataclass(frozen=True, slots=True)
class CostEstimateRecord:
    """Pre-submission compute cost estimate with expected and hard upper bounds."""

    schema: Literal["training-cost-estimate.v1"]
    recipe_digest: str
    driver_name: str
    accelerator_type: str
    accelerator_count: int
    replica_count: int
    scheduling_strategy: Literal["LOCAL", "SPOT", "FLEX_START", "STANDARD"]
    max_wall_seconds: int
    unit: str
    estimated_expected_cost: str
    hard_upper_bound_cost: str
    price_source_and_revision: str
    estimated_at_utc: str


@dataclass(frozen=True, slots=True)
class PaidRunApprovalRecord:
    """Per-run owner approval record linking cost estimate to spend pool."""

    schema: Literal["paid-run-approval.v1"]
    cost_estimate_digest: str
    owner_approval_ref: str
    pool_id: str
    spend_binding: SpendReservationBinding


@dataclass(frozen=True, slots=True)
class SandboxConfinementSpec:
    """Mandatory filesystem, seccomp, and network confinement attestation."""

    schema: Literal["sandbox-confinement-spec.v1"]
    filesystem_and_seccomp_enforced: Literal[True]
    network_isolation_layer: Literal[
        "landlock_v4",
        "nftables_skuid",
        "cloud_vpc_disabled_egress",
    ]


@dataclass(frozen=True, slots=True)
class CloudJobSubmissionReceipt:
    """Immutable submission receipt for local-fleet or cloud training jobs."""

    schema: Literal["cloud-job-submission.v1"]
    driver_name: str
    funding_kind: PoolFundingKind
    job_id: str
    dry_run: bool
    container_image_digest: str
    confinement: SandboxConfinementSpec
    input_shards_digests: tuple[tuple[str, str], ...]
    spend_binding_digest: str
    paid_approval_digest: str | None
    stop_lease_operation_id: str
    submitted_at_utc: str


@dataclass(frozen=True, slots=True)
class CloudJobStatusReport:
    """Point-in-time execution status of a submitted compute job."""

    schema: Literal["cloud-job-status.v1"]
    job_id: str
    state: Literal[
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "preempted",
    ]
    latest_checkpoint_digest: str | None
    consumed_wall_seconds: int
    observed_at_utc: str


@dataclass(frozen=True, slots=True)
class CloudJobCancelReceipt:
    """Terminal cancellation confirmation for a compute job."""

    schema: Literal["cloud-job-cancel.v1"]
    job_id: str
    cancelled: bool
    reason: str
    cancelled_at_utc: str
