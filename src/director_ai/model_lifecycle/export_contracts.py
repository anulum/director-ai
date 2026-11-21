# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Export Parity Validators

"""Validators of ONNX export profiles and serving-format parity reports.

An exported binary is non-inferior to its parent only if every slice passed
with its paired lower bound above minus its margin, at a family-wise alpha of
at most 0.01. An ONNX export names its profile, opset 17 or 18 and static,
positive input shapes, and fits its profile's byte cap; INT8 on x86-64-v2
uses the reduced range; an AudioWorklet model is WASM SIMD, allocates
nothing per audio quantum, and its cap stays below 2 MiB. A GGUF export
must pass its stop-token and unused-token checks.
"""

from __future__ import annotations

from ._checks import (
    require_bool,
    require_choice,
    require_finite,
    require_int,
    require_sha256,
    require_text,
    require_tuple,
    require_unique,
)
from .errors import ExportParityError
from .study_records import (
    INT8_QUANTISATION_MODES,
    MAX_FAMILYWISE_ALPHA,
    WASM_AUDIOWORKLET_MAX_BYTES,
    ExportParityMetricSlice,
    ExportParityReport,
    OnnxExportProfileSpec,
)

E = ExportParityError


def _rate(value: object, field: str) -> float:
    """Return a finite float in ``[0, 1]``."""
    rate = require_finite(value, field, E)
    if not 0.0 <= rate <= 1.0:
        raise E(f"{field} must lie in [0, 1]")
    return rate


def validate_onnx_export_profile(
    profile: OnnxExportProfileSpec,
) -> OnnxExportProfileSpec:
    """Check an ONNX export profile on its own.

    Parameters
    ----------
    profile : OnnxExportProfileSpec
        The profile to check.

    Returns
    -------
    OnnxExportProfileSpec
        The same profile.

    Raises
    ------
    ExportParityError
        If the opset, mode or target is unknown, an input shape is empty,
        dynamic or repeated, the byte cap is not positive, or a flag is
        malformed.
    """
    if not isinstance(profile, OnnxExportProfileSpec):
        raise E("an OnnxExportProfileSpec is required")
    require_choice(profile.schema, "schema", {"onnx-export-profile.v1"}, E)
    if (
        profile.onnx_opset_version not in (17, 18)
        or type(profile.onnx_opset_version) is not int
    ):
        raise E("onnx_opset_version must be 17 or 18")
    require_choice(
        profile.quantisation_mode,
        "quantisation_mode",
        {"fp32", "fp16", "dynamic_qint8", "static_qoperator_int8"},
        E,
    )
    require_choice(profile.target_isa, "target_isa", {"x86-64-v2", "wasm32_simd128"}, E)
    for name in (
        "int8_reduce_range",
        "wasm_audioworklet_target",
        "zero_per_quantum_heap_alloc",
    ):
        require_bool(getattr(profile, name), name, E)
    shapes = require_tuple(profile.input_tensor_shapes, "input_tensor_shapes", E)
    if not shapes:
        raise E("an ONNX export needs static input shapes")
    names: list[str] = []
    for pair in shapes:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise E("input_tensor_shapes must hold (name, shape) pairs")
        names.append(require_text(pair[0], "input name", E))
        dims = require_tuple(pair[1], "input shape", E)
        if not dims:
            raise E(f"input {pair[0]!r} needs a static shape")
        for dim in dims:
            require_int(dim, f"input {pair[0]!r} dimension", E, minimum=1)
    require_unique(names, "input names", E)
    require_int(profile.max_exported_bytes, "max_exported_bytes", E, minimum=1)
    return profile


def _check_slices(report: ExportParityReport, alpha: float) -> None:
    """Check that every slice passed non-inferiority at a valid adjusted alpha."""
    slices = require_tuple(report.slice_checks, "slice_checks", E)
    if not slices:
        raise E("a parity report needs at least one slice check")
    keys: list[tuple[str, str]] = []
    for item in slices:
        if not isinstance(item, ExportParityMetricSlice):
            raise E("slice_checks must hold ExportParityMetricSlice records")
        keys.append(
            (
                require_text(item.slice_id, "slice_id", E),
                require_text(item.metric_id, "metric_id", E),
            )
        )
        for name in ("parent_estimate", "export_estimate", "paired_diff_lcb"):
            require_finite(getattr(item, name), name, E)
        if require_finite(item.allowed_margin, "allowed_margin", E) < 0.0:
            raise E("allowed_margin must not be negative")
        adjusted = require_finite(item.holm_adjusted_alpha, "holm_adjusted_alpha", E)
        if not 0.0 < adjusted <= alpha:
            raise E("holm_adjusted_alpha must lie in (0, familywise_alpha]")
        require_bool(item.passed, "passed", E)
        if not (item.passed and item.paired_diff_lcb >= -item.allowed_margin):
            raise E(f"slice {item.slice_id!r} / {item.metric_id!r} is not non-inferior")
    require_unique(keys, "slice checks", E)


def _check_onnx(report: ExportParityReport) -> None:
    """Check an ONNX report's profile, INT8 range and AudioWorklet bounds."""
    if report.onnx_profile is None:
        raise E("an ONNX export needs its profile")
    if report.gguf_stop_token_and_zero_unused_passed is not None:
        raise E("an ONNX report carries no GGUF check")
    profile = validate_onnx_export_profile(report.onnx_profile)
    if (
        profile.target_isa == "x86-64-v2"
        and profile.quantisation_mode in INT8_QUANTISATION_MODES
        and profile.int8_reduce_range is not True
    ):
        raise E("INT8 on x86-64-v2 must use the reduced range")
    if report.exported_byte_length > profile.max_exported_bytes:
        raise E("an ONNX export must fit its profile's max_exported_bytes")
    if profile.wasm_audioworklet_target:
        if (
            profile.target_isa != "wasm32_simd128"
            or profile.zero_per_quantum_heap_alloc is not True
        ):
            raise E("an AudioWorklet model is WASM SIMD with no per-quantum allocation")
        if profile.max_exported_bytes >= WASM_AUDIOWORKLET_MAX_BYTES:
            raise E("an AudioWorklet model's byte cap must stay below 2 MiB")


def validate_export_parity_report(report: ExportParityReport) -> ExportParityReport:
    """Check that an exported serving binary is non-inferior to its parent.

    Parameters
    ----------
    report : ExportParityReport
        The report to check.

    Returns
    -------
    ExportParityReport
        The same report.

    Raises
    ------
    ExportParityError
        If the family-wise alpha is outside ``(0, 0.01]``, a slice is not
        non-inferior, the tokeniser or template differs, the format's own
        checks fail, or a field is malformed.
    MalformedDigestError
        If a digest is malformed.
    """
    if not isinstance(report, ExportParityReport):
        raise E("an ExportParityReport is required")
    require_choice(report.schema, "schema", {"export-parity-report.v1"}, E)
    require_sha256(report.parent_content_digest, "parent_content_digest")
    require_sha256(report.exported_file_sha256, "exported_file_sha256")
    require_int(report.exported_byte_length, "exported_byte_length", E, minimum=1)
    export_format = require_choice(
        report.export_format, "export_format", {"gguf", "onnx"}, E
    )
    require_text(
        report.serving_runtime_id_and_version, "serving_runtime_id_and_version", E
    )
    if report.tokeniser_and_template_parity_exact is not True:
        raise E("the exported tokeniser and template must match the parent exactly")
    _rate(report.greedy_dev_identity_rate, "greedy_dev_identity_rate")
    if report.structured_json_validity_rate is not None:
        _rate(report.structured_json_validity_rate, "structured_json_validity_rate")
    if report.dynamics_rollout_horizon_h is not None:
        require_int(
            report.dynamics_rollout_horizon_h,
            "dynamics_rollout_horizon_h",
            E,
            minimum=1,
        )
    alpha = require_finite(report.familywise_alpha, "familywise_alpha", E)
    if not 0.0 < alpha <= MAX_FAMILYWISE_ALPHA:
        raise E("familywise_alpha must lie in (0, 0.01]")
    _check_slices(report, alpha)
    if export_format == "onnx":
        _check_onnx(report)
    elif (
        report.onnx_profile is not None
        or report.gguf_stop_token_and_zero_unused_passed is not True
    ):
        raise E(
            "a GGUF export carries no ONNX profile and must pass its stop-token checks"
        )
    return report
