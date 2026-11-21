# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Width and Position Validators

"""Validators of YaRN position scaling and maximal-update width transfer.

YaRN extends a rotary model past its original context: the factor exceeds
one, the original length lies below the new context length and the fast
boundary exceeds the slow one. Width transfer names its exponents and the
target width, which must be the model's ``d_model``; its coordinate-check
receipt tests at least two increasing proxy widths.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_tuple,
    require_utc_timestamp,
)
from .architecture_specs import (
    MaximalUpdateParametrisationSpec,
    MupValidationReceipt,
    YarnRopeScalingSpec,
)
from .errors import ArchitectureSpecError

E = ArchitectureSpecError


def positive_float(value: object, field: str) -> float:
    """Return a finite float above zero, else refuse."""
    number = require_finite(value, field, E)
    if number <= 0.0:
        raise E(f"{field} must be above zero")
    return number


def optional_count(value: object, field: str) -> int | None:
    """Return ``None`` or an integer of at least one."""
    return None if value is None else require_int(value, field, E, minimum=1)


def validate_yarn_rope_scaling(
    scaling: YarnRopeScalingSpec, *, max_context_length: int
) -> YarnRopeScalingSpec:
    """Check YaRN scaling against the context length it extends to.

    Parameters
    ----------
    scaling : YarnRopeScalingSpec
        The scaling to check.
    max_context_length : int
        The model's context length.

    Returns
    -------
    YarnRopeScalingSpec
        The same scaling.

    Raises
    ------
    ArchitectureSpecError
        If ``factor <= 1``, the original length is not below the context
        length, ``beta_fast <= beta_slow``, ``beta_slow <= 0``, or a field
        is malformed.
    """
    if not isinstance(scaling, YarnRopeScalingSpec):
        raise E("rope_scaling must be a YarnRopeScalingSpec")
    require_choice(scaling.schema, "schema", {"yarn-rope-scaling.v1"}, E)
    if require_finite(scaling.factor, "factor", E) <= 1.0:
        raise E("a YaRN factor must exceed 1")
    original = require_int(
        scaling.original_max_position_embeddings,
        "original_max_position_embeddings",
        E,
        minimum=1,
    )
    if original >= max_context_length:
        raise E("original_max_position_embeddings must be below max_context_length")
    for name in ("attention_factor", "extrapolation_factor"):
        if getattr(scaling, name) is not None:
            require_finite(getattr(scaling, name), name, E)
    if not require_finite(scaling.beta_fast, "beta_fast", E) > positive_float(
        scaling.beta_slow, "beta_slow"
    ):
        raise E("beta_fast must exceed beta_slow")
    return scaling


def validate_mup_validation_receipt(
    receipt: MupValidationReceipt,
) -> MupValidationReceipt:
    """Check a width-transfer coordinate-check receipt.

    Parameters
    ----------
    receipt : MupValidationReceipt
        The receipt to check.

    Returns
    -------
    MupValidationReceipt
        The same receipt.

    Raises
    ------
    ArchitectureSpecError
        If fewer than two widths were tested, the widths are not strictly
        increasing, or a field is malformed.
    MalformedDigestError
        If the dedup report digest is malformed.
    """
    if not isinstance(receipt, MupValidationReceipt):
        raise E("a MupValidationReceipt is required")
    require_choice(receipt.schema, "schema", {"mup-validation-receipt.v1"}, E)
    widths = [
        require_int(width, "proxy_widths_tested", E, minimum=1)
        for width in require_tuple(
            receipt.proxy_widths_tested, "proxy_widths_tested", E
        )
    ]
    if len(widths) < 2 or any(a >= b for a, b in zip(widths, widths[1:], strict=False)):
        raise E("a coordinate check needs at least two strictly increasing widths")
    positive_float(receipt.optimal_base_lr, "optimal_base_lr")
    if (
        require_finite(
            receipt.max_optimal_lr_drift_ratio, "max_optimal_lr_drift_ratio", E
        )
        < 0.0
    ):
        raise E("max_optimal_lr_drift_ratio must not be negative")
    require_int(receipt.coordinate_check_steps, "coordinate_check_steps", E, minimum=1)
    require_sha256(receipt.dedup_report_digest, "dedup_report_digest")
    require_utc_timestamp(receipt.verified_at_utc, "verified_at_utc", E)
    return receipt


def validate_mup_spec(
    mup: MaximalUpdateParametrisationSpec, *, d_model: int
) -> MaximalUpdateParametrisationSpec:
    """Check a width-transfer spec's fields and its target width.

    Parameters
    ----------
    mup : MaximalUpdateParametrisationSpec
        The spec to check.
    d_model : int
        The width of the model it transfers to.

    Returns
    -------
    MaximalUpdateParametrisationSpec
        The same spec.

    Raises
    ------
    ArchitectureSpecError
        If a field is malformed or ``target_d_model`` differs from ``d_model``.
    MalformedDigestError
        If the receipt digest is malformed.
    """
    if not isinstance(mup, MaximalUpdateParametrisationSpec):
        raise E("mup_spec must be a MaximalUpdateParametrisationSpec")
    require_choice(mup.schema, "schema", {"mup-spec.v1"}, E)
    require_int(mup.base_d_model, "base_d_model", E, minimum=1)
    if require_int(mup.target_d_model, "target_d_model", E, minimum=1) != d_model:
        raise E("mup_spec.target_d_model must equal d_model")
    positive_float(mup.input_scale, "input_scale")
    require_bool(mup.output_zero_init, "output_zero_init", E)
    require_choice(
        mup.attention_logit_scale_mode,
        "attention_logit_scale_mode",
        {"standard_inv_sqrt_d", "mup_inv_d"},
        E,
    )
    for name in (
        "hidden_init_std_scale_power",
        "embedding_lr_scale_power",
        "hidden_matrix_lr_scale_power",
        "norm_and_bias_lr_scale_power",
        "readout_lr_scale_power",
    ):
        require_finite(getattr(mup, name), name, E)
    if mup.proxy_validation_receipt_digest is not None:
        require_sha256(
            mup.proxy_validation_receipt_digest, "proxy_validation_receipt_digest"
        )
    return mup
