# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Job Record Validators

"""Validators of a compute job's confinement, submission, status and cancellation.

Every job runs with filesystem and seccomp confinement and one network
isolation layer. A paid run's submission names the owner's approval; any
other submission names none.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
    require_utc_timestamp,
)
from .architecture_specs import POOL_FUNDING_KINDS
from .errors import SpendBindingError
from .spend_records import (
    JOB_STATES,
    NETWORK_ISOLATION_LAYERS,
    CloudJobCancelReceipt,
    CloudJobStatusReport,
    CloudJobSubmissionReceipt,
    SandboxConfinementSpec,
)

E = SpendBindingError
PAID = "per_run_owner_approved_paid"


def validate_sandbox_confinement(
    spec: SandboxConfinementSpec,
) -> SandboxConfinementSpec:
    """Check a job's sandbox confinement attestation.

    Parameters
    ----------
    spec : SandboxConfinementSpec
        The confinement to check.

    Returns
    -------
    SandboxConfinementSpec
        The same confinement.

    Raises
    ------
    SpendBindingError
        If filesystem and seccomp confinement is not enforced or the network
        layer is unknown.
    """
    if not isinstance(spec, SandboxConfinementSpec):
        raise E("a SandboxConfinementSpec is required")
    require_choice(spec.schema, "schema", {"sandbox-confinement-spec.v1"}, E)
    if spec.filesystem_and_seccomp_enforced is not True:
        raise E("filesystem and seccomp confinement must be enforced")
    require_choice(
        spec.network_isolation_layer,
        "network_isolation_layer",
        NETWORK_ISOLATION_LAYERS,
        E,
    )
    return spec


def validate_job_submission(
    receipt: CloudJobSubmissionReceipt,
) -> CloudJobSubmissionReceipt:
    """Check a job's submission receipt.

    Parameters
    ----------
    receipt : CloudJobSubmissionReceipt
        The receipt to check.

    Returns
    -------
    CloudJobSubmissionReceipt
        The same receipt.

    Raises
    ------
    SpendBindingError
        If the confinement is refused, an input shard name repeats, a paid
        run names no approval or another run names one, or a field is
        malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(receipt, CloudJobSubmissionReceipt):
        raise E("a CloudJobSubmissionReceipt is required")
    require_choice(receipt.schema, "schema", {"cloud-job-submission.v1"}, E)
    for name in ("driver_name", "job_id", "stop_lease_operation_id"):
        require_text(getattr(receipt, name), name, E)
    funding = require_choice(
        receipt.funding_kind, "funding_kind", POOL_FUNDING_KINDS, E
    )
    require_bool(receipt.dry_run, "dry_run", E)
    require_sha256(receipt.container_image_digest, "container_image_digest")
    validate_sandbox_confinement(receipt.confinement)
    names: list[str] = []
    for pair in require_tuple(receipt.input_shards_digests, "input_shards_digests", E):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E("input_shards_digests must hold (name, digest) pairs")
        names.append(require_text(pair[0], "input shard name", E))
        require_sha256(pair[1], "input shard digest")
    require_unique(names, "input shard names", E)
    require_sha256(receipt.spend_binding_digest, "spend_binding_digest")
    if funding == PAID:
        require_sha256(receipt.paid_approval_digest, "paid_approval_digest")
    elif receipt.paid_approval_digest is not None:
        raise E("only a paid run names a paid-run approval")
    require_utc_timestamp(receipt.submitted_at_utc, "submitted_at_utc", E)
    return receipt


def validate_job_status(report: CloudJobStatusReport) -> CloudJobStatusReport:
    """Check a job's status report.

    Parameters
    ----------
    report : CloudJobStatusReport
        The report to check.

    Returns
    -------
    CloudJobStatusReport
        The same report.

    Raises
    ------
    SpendBindingError
        If a field is malformed.
    MalformedDigestError
        If the checkpoint digest is malformed.
    """
    if not isinstance(report, CloudJobStatusReport):
        raise E("a CloudJobStatusReport is required")
    require_choice(report.schema, "schema", {"cloud-job-status.v1"}, E)
    require_text(report.job_id, "job_id", E)
    require_choice(report.state, "state", JOB_STATES, E)
    if report.latest_checkpoint_digest is not None:
        require_sha256(report.latest_checkpoint_digest, "latest_checkpoint_digest")
    require_int(report.consumed_wall_seconds, "consumed_wall_seconds", E, minimum=0)
    require_utc_timestamp(report.observed_at_utc, "observed_at_utc", E)
    return report


def validate_job_cancel(receipt: CloudJobCancelReceipt) -> CloudJobCancelReceipt:
    """Check a job's cancellation receipt.

    Parameters
    ----------
    receipt : CloudJobCancelReceipt
        The receipt to check.

    Returns
    -------
    CloudJobCancelReceipt
        The same receipt.

    Raises
    ------
    SpendBindingError
        If a field is malformed.
    """
    if not isinstance(receipt, CloudJobCancelReceipt):
        raise E("a CloudJobCancelReceipt is required")
    require_choice(receipt.schema, "schema", {"cloud-job-cancel.v1"}, E)
    require_text(receipt.job_id, "job_id", E)
    require_bool(receipt.cancelled, "cancelled", E)
    require_text(receipt.reason, "reason", E)
    require_utc_timestamp(receipt.cancelled_at_utc, "cancelled_at_utc", E)
    return receipt
