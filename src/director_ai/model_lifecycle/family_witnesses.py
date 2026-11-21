# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Family Witness Validators

"""Witnesses a quantum or spiking model carries: the preflight and the parity.

A barren-plateau witness records the smallest gradient variance observed
over sampled parameter points against its threshold; its verdict must follow
those numbers. A spiking lowering parity witness ties a checkpoint to the
native engine with no blocking under- or overflow, and a promotion takes it
only when it passed (TSN-3).
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
)
from .errors import QuantumAcceptanceError, SpikingParityError
from .quantum_records import (
    SPIKING_RESET_SEMANTICS,
    BarrenPlateauWitnessReceipt,
    SpikingLoweringParityReport,
)

E = QuantumAcceptanceError


def validate_barren_plateau_receipt(
    receipt: BarrenPlateauWitnessReceipt,
) -> BarrenPlateauWitnessReceipt:
    """Check a barren-plateau preflight witness, whose verdict follows its numbers.

    Parameters
    ----------
    receipt : BarrenPlateauWitnessReceipt
        The witness to check.

    Returns
    -------
    BarrenPlateauWitnessReceipt
        The same witness.

    Raises
    ------
    QuantumAcceptanceError
        If ``passed`` disagrees with the observed variance against the
        threshold, or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(receipt, BarrenPlateauWitnessReceipt):
        raise E("a BarrenPlateauWitnessReceipt is required")
    require_choice(receipt.schema, "schema", {"barren-plateau-witness-receipt.v1"}, E)
    require_sha256(receipt.companion_digest, "companion_digest")
    require_sha256(receipt.split_pool_digest, "split_pool_digest")
    require_int(
        receipt.sampled_parameter_points, "sampled_parameter_points", E, minimum=1
    )
    observed = require_finite(
        receipt.min_observed_grad_variance, "min_observed_grad_variance", E
    )
    threshold = require_finite(
        receipt.threshold_grad_variance, "threshold_grad_variance", E
    )
    if observed < 0.0 or threshold <= 0.0:
        raise E("the variance is not negative and the threshold is above zero")
    if require_bool(receipt.passed, "passed", E) != (observed >= threshold):
        raise E("passed must say whether the observed variance reaches the threshold")
    return receipt


def validate_spiking_lowering_parity(
    report: SpikingLoweringParityReport,
) -> SpikingLoweringParityReport:
    """Check a spiking lowering parity report's shape, passed or not.

    Parameters
    ----------
    report : SpikingLoweringParityReport
        The parity report.

    Returns
    -------
    SpikingLoweringParityReport
        The same report.

    Raises
    ------
    SpikingParityError
        If the report counts a blocking underflow or overflow, or a field is
        malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(report, SpikingLoweringParityReport):
        raise SpikingParityError("a SpikingLoweringParityReport is required")
    require_choice(
        report.schema,
        "schema",
        {"spiking-lowering-parity-report.v1"},
        SpikingParityError,
    )
    require_sha256(report.checkpoint_content_digest, "checkpoint_content_digest")
    require_sha256(
        report.hardware_numeric_contract_digest, "hardware_numeric_contract_digest"
    )
    count = report.blocking_underflow_or_overflow_count
    if type(count) is not int or count != 0:
        raise SpikingParityError(
            "a parity witness counts no blocking under- or overflow"
        )
    require_choice(
        report.reset_semantics,
        "reset_semantics",
        SPIKING_RESET_SEMANTICS,
        SpikingParityError,
    )
    require_text(report.surrogate_family, "surrogate_family", SpikingParityError)
    if require_finite(report.dt_seconds, "dt_seconds", SpikingParityError) <= 0.0:
        raise SpikingParityError("dt_seconds must be above zero")
    require_int(
        report.simulation_steps_t, "simulation_steps_t", SpikingParityError, minimum=1
    )
    distance = require_finite(
        report.victor_purpura_or_van_rossum_distance,
        "victor_purpura_or_van_rossum_distance",
        SpikingParityError,
    )
    agreement = require_finite(
        report.classification_agreement_rate,
        "classification_agreement_rate",
        SpikingParityError,
    )
    if distance < 0.0 or not 0.0 <= agreement <= 1.0:
        raise SpikingParityError(
            "the distance is not negative and the agreement in [0, 1]"
        )
    require_bool(report.passed, "passed", SpikingParityError)
    return report


def require_spiking_parity_passed(
    report: SpikingLoweringParityReport,
) -> SpikingLoweringParityReport:
    """Refuse a spiking lowering parity witness that did not pass (TSN-3).

    Parameters
    ----------
    report : SpikingLoweringParityReport
        The parity report.

    Returns
    -------
    SpikingLoweringParityReport
        The same report.

    Raises
    ------
    SpikingParityError
        If the report did not pass or is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if validate_spiking_lowering_parity(report).passed is not True:
        raise SpikingParityError("the spiking lowering parity witness did not pass")
    return report
