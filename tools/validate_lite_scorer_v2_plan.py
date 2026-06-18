#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 plan validator

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

PLAN = Path("benchmarks/lite_scorer_v2_plan.toml")
DEFAULT_EVIDENCE_PACKET = Path("benchmarks/lite_scorer_v2_evidence_packet.toml")
CLAIM_SURFACES = (
    Path("README.md"),
    Path("docs/BENCHMARKS.md"),
    Path("docs-site/guide/scoring.md"),
    Path("docs-site/installation.md"),
    Path("src/director_ai/core/scoring/backends.py"),
    Path("src/director_ai/core/scoring/distilled_scorer.py"),
)
REQUIRED_FIELDS = {
    "schema_version",
    "plan_id",
    "public_score_claim",
    "claim_boundary",
    "student_candidates",
    "teacher_artifact_required",
    "heldout_eval_required",
    "onnx_export_required",
    "quantized_latency_required",
    "minimum_real_eval_rows",
    "status",
    "evidence_packet",
}
REQUIRED_EVIDENCE_FIELDS = {
    "schema_version",
    "packet_id",
    "public_score_claim",
    "student_artifact_status",
    "teacher_artifact_status",
    "heldout_eval_status",
    "onnx_export_status",
    "quantized_latency_status",
    "model_card_status",
    "benchmark_claim_review_status",
    "claim_boundary",
}
REQUIRED_STUDENTS = {"minilm_l6", "mobilebert", "distilbert"}
ALLOWED_STATUS = {"design_ready", "training_ready", "validated"}
ALLOWED_EVIDENCE_STATUS = {"pending", "recorded", "validated"}
RECORDED_STATUS = {"recorded", "validated"}
EVIDENCE_STATUS_KEYS = (
    "student_artifact_status",
    "teacher_artifact_status",
    "heldout_eval_status",
    "onnx_export_status",
    "quantized_latency_status",
    "model_card_status",
    "benchmark_claim_review_status",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
UNVERIFIED_CLAIM = re.compile(
    r"(?:~\s*70\s*%\s*BA|70\s*%\s*BA|5\s*ms\s*CPU|5ms\s*CPU)",
    re.IGNORECASE,
)


def _load_plan(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"{PLAN}: missing Lite Scorer v2 plan"]
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{PLAN}: invalid TOML: {exc}"]
    return data, []


def _validate_plan(data: dict[str, Any]) -> list[str]:
    missing = sorted(REQUIRED_FIELDS - set(data))
    if missing:
        return [f"{PLAN}: missing required fields {', '.join(missing)}"]

    errors: list[str] = []
    if data["public_score_claim"] is not False:
        errors.append(f"{PLAN}: public_score_claim must be false")

    boundary = data["claim_boundary"]
    if not isinstance(boundary, str) or "no public score claim" not in boundary.lower():
        errors.append(f"{PLAN}: claim_boundary must state no public score claim")

    candidates = data["student_candidates"]
    if not isinstance(candidates, list) or set(candidates) != REQUIRED_STUDENTS:
        errors.append(
            f"{PLAN}: student_candidates must be {sorted(REQUIRED_STUDENTS)!r}"
        )

    for key in (
        "teacher_artifact_required",
        "heldout_eval_required",
        "onnx_export_required",
        "quantized_latency_required",
    ):
        if data[key] is not True:
            errors.append(f"{PLAN}: {key} must be true")

    rows = data["minimum_real_eval_rows"]
    if not isinstance(rows, int) or rows < 1000:
        errors.append(f"{PLAN}: minimum_real_eval_rows must be at least 1000")

    status = data["status"]
    if status not in ALLOWED_STATUS:
        errors.append(f"{PLAN}: unsupported status {status!r}")
    if status == "validated" and data["public_score_claim"] is not True:
        errors.append(
            f"{PLAN}: validated status requires a separate scored release plan"
        )

    if data["evidence_packet"] != DEFAULT_EVIDENCE_PACKET.as_posix():
        errors.append(
            f"{PLAN}: evidence_packet must be {DEFAULT_EVIDENCE_PACKET.as_posix()}"
        )

    return errors


def _require_string(
    packet: dict[str, Any],
    label: Path,
    status_key: str,
    field: str,
) -> list[str]:
    value = packet.get(field)
    if isinstance(value, str) and value.strip():
        return []
    return [f"{label}: {status_key} {packet[status_key]} requires {field}"]


def _require_sha256(
    packet: dict[str, Any],
    label: Path,
    status_key: str,
    field: str,
) -> list[str]:
    value = packet.get(field)
    if isinstance(value, str) and SHA256.fullmatch(value):
        return []
    return [f"{label}: {status_key} {packet[status_key]} requires {field}"]


def _require_float_range(
    packet: dict[str, Any],
    label: Path,
    status_key: str,
    field: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> list[str]:
    value = packet.get(field)
    if isinstance(value, int | float) and not isinstance(value, bool):
        numeric = float(value)
        if numeric >= minimum and (maximum is None or numeric <= maximum):
            return []
    return [f"{label}: {status_key} {packet[status_key]} requires {field}"]


def _require_int_at_least(
    packet: dict[str, Any],
    label: Path,
    status_key: str,
    field: str,
    minimum: int,
) -> list[str]:
    value = packet.get(field)
    if isinstance(value, int) and not isinstance(value, bool) and value >= minimum:
        return []
    return [f"{label}: {status_key} {packet[status_key]} requires {field}"]


def _validate_recorded_evidence(
    packet: dict[str, Any],
    label: Path,
    minimum_real_eval_rows: int,
) -> list[str]:
    errors: list[str] = []

    if packet["student_artifact_status"] in RECORDED_STATUS:
        errors.extend(
            _require_string(
                packet, label, "student_artifact_status", "student_candidate"
            )
        )
        if packet.get("student_candidate") not in REQUIRED_STUDENTS:
            errors.append(
                f"{label}: student_artifact_status {packet['student_artifact_status']} requires student_candidate from {sorted(REQUIRED_STUDENTS)!r}"
            )
        errors.extend(
            _require_string(
                packet, label, "student_artifact_status", "student_artifact_path"
            )
        )
        errors.extend(
            _require_sha256(
                packet, label, "student_artifact_status", "student_artifact_sha256"
            )
        )

    if packet["teacher_artifact_status"] in RECORDED_STATUS:
        errors.extend(
            _require_string(
                packet, label, "teacher_artifact_status", "teacher_artifact_path"
            )
        )
        errors.extend(
            _require_sha256(
                packet, label, "teacher_artifact_status", "teacher_artifact_sha256"
            )
        )

    if packet["heldout_eval_status"] in RECORDED_STATUS:
        errors.extend(
            _require_string(
                packet, label, "heldout_eval_status", "heldout_eval_dataset"
            )
        )
        errors.extend(
            _require_int_at_least(
                packet,
                label,
                "heldout_eval_status",
                "heldout_eval_rows",
                minimum_real_eval_rows,
            )
        )
        errors.extend(
            _require_float_range(
                packet,
                label,
                "heldout_eval_status",
                "heldout_eval_balanced_accuracy",
                minimum=0.0,
                maximum=1.0,
            )
        )
        errors.extend(
            _require_float_range(
                packet,
                label,
                "heldout_eval_status",
                "heldout_eval_threshold",
                minimum=0.0,
                maximum=1.0,
            )
        )

    if packet["onnx_export_status"] in RECORDED_STATUS:
        errors.extend(
            _require_string(packet, label, "onnx_export_status", "onnx_artifact_path")
        )
        errors.extend(
            _require_sha256(packet, label, "onnx_export_status", "onnx_artifact_sha256")
        )

    if packet["quantized_latency_status"] in RECORDED_STATUS:
        errors.extend(
            _require_string(
                packet, label, "quantized_latency_status", "latency_backend"
            )
        )
        errors.extend(
            _require_string(packet, label, "quantized_latency_status", "latency_device")
        )
        errors.extend(
            _require_int_at_least(
                packet, label, "quantized_latency_status", "latency_sample_count", 100
            )
        )
        errors.extend(
            _require_float_range(
                packet,
                label,
                "quantized_latency_status",
                "latency_p50_ms",
                minimum=1e-12,
            )
        )
        errors.extend(
            _require_float_range(
                packet,
                label,
                "quantized_latency_status",
                "latency_p95_ms",
                minimum=1e-12,
            )
        )
        p50 = packet.get("latency_p50_ms")
        p95 = packet.get("latency_p95_ms")
        if (
            isinstance(p50, int | float)
            and not isinstance(p50, bool)
            and isinstance(p95, int | float)
            and not isinstance(p95, bool)
            and float(p95) < float(p50)
        ):
            errors.append(
                f"{label}: latency_p95_ms must be greater than latency_p50_ms"
            )

    recorded_core_statuses = {
        packet["student_artifact_status"],
        packet["teacher_artifact_status"],
        packet["heldout_eval_status"],
        packet["onnx_export_status"],
        packet["quantized_latency_status"],
    } & RECORDED_STATUS
    if recorded_core_statuses and packet["model_card_status"] not in RECORDED_STATUS:
        errors.append(
            f"{label}: recorded Lite Scorer v2 evidence requires model_card_status recorded"
        )
    if (
        recorded_core_statuses
        and packet["benchmark_claim_review_status"] not in RECORDED_STATUS
    ):
        errors.append(
            f"{label}: recorded Lite Scorer v2 evidence requires benchmark_claim_review_status recorded"
        )

    if packet["model_card_status"] in RECORDED_STATUS:
        errors.extend(
            _require_string(packet, label, "model_card_status", "model_card_path")
        )
        errors.extend(
            _require_sha256(packet, label, "model_card_status", "model_card_sha256")
        )

    if packet["benchmark_claim_review_status"] in RECORDED_STATUS:
        errors.extend(
            _require_string(
                packet,
                label,
                "benchmark_claim_review_status",
                "benchmark_claim_review_path",
            )
        )
        errors.extend(
            _require_sha256(
                packet,
                label,
                "benchmark_claim_review_status",
                "benchmark_claim_review_sha256",
            )
        )

    return errors


def _require_recorded_evidence_packet(packet: dict[str, Any], label: Path) -> list[str]:
    errors: list[str] = []
    for key in EVIDENCE_STATUS_KEYS:
        if packet[key] not in RECORDED_STATUS:
            errors.append(
                f"{label}: {key} must be recorded or validated for release evidence"
            )
    return errors


def _validate_evidence_packet(
    root: Path,
    data: dict[str, Any],
    *,
    require_recorded_evidence: bool,
) -> list[str]:
    packet_ref = data.get("evidence_packet")
    if not isinstance(packet_ref, str):
        return [f"{PLAN}: evidence_packet must be a string path"]
    packet_path = Path(packet_ref)
    label = packet_path
    absolute = root / packet_path
    if not absolute.exists():
        return [f"{label}: missing Lite Scorer v2 evidence packet"]
    try:
        packet = tomllib.loads(absolute.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [f"{label}: invalid TOML: {exc}"]

    missing = sorted(REQUIRED_EVIDENCE_FIELDS - set(packet))
    if missing:
        return [f"{label}: missing required fields {', '.join(missing)}"]

    errors: list[str] = []
    if packet["public_score_claim"] is not False:
        errors.append(f"{label}: public_score_claim must be false")
    boundary = packet["claim_boundary"]
    if not isinstance(boundary, str) or "no public score claim" not in boundary.lower():
        errors.append(f"{label}: claim_boundary must state no public score claim")

    for key in EVIDENCE_STATUS_KEYS:
        if packet[key] not in ALLOWED_EVIDENCE_STATUS:
            errors.append(f"{label}: unsupported {key} {packet[key]!r}")
    if require_recorded_evidence:
        errors.extend(_require_recorded_evidence_packet(packet, label))
    rows = data.get("minimum_real_eval_rows")
    minimum_real_eval_rows = rows if isinstance(rows, int) else 1000
    errors.extend(_validate_recorded_evidence(packet, label, minimum_real_eval_rows))
    return errors


def _validate_claim_surfaces(root: Path) -> list[str]:
    errors: list[str] = []
    for doc in CLAIM_SURFACES:
        path = root / doc
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if UNVERIFIED_CLAIM.search(text):
            errors.append(
                f"{doc}: remove unverified Lite Scorer v2 accuracy or latency claim"
            )
    return errors


def validate_lite_scorer_v2_plan(
    root: Path,
    *,
    require_recorded_evidence: bool = False,
) -> list[str]:
    """Validate the Lite Scorer v2 plan and optional release evidence gate.

    Parameters
    ----------
    root:
        Repository root containing the Lite Scorer v2 plan and evidence packet.
    require_recorded_evidence:
        When true, every release-relevant evidence status must be ``recorded``
        or ``validated``. The default accepts the pending no-claim packet used
        while training and review artefacts are still being produced.

    Returns
    -------
    list[str]
        Human-readable validation errors. An empty list means the checked
        policy surface is valid.
    """
    root = root.resolve()
    data, errors = _load_plan(root / PLAN)
    if errors:
        return errors
    errors.extend(_validate_plan(data))
    errors.extend(
        _validate_evidence_packet(
            root,
            data,
            require_recorded_evidence=require_recorded_evidence,
        )
    )
    errors.extend(_validate_claim_surfaces(root))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Repository root containing benchmarks/lite_scorer_v2_plan.toml",
    )
    parser.add_argument(
        "--require-recorded-evidence",
        action="store_true",
        help=(
            "Require student, teacher, ONNX, held-out eval, quantized latency, "
            "model-card, and benchmark-review statuses to be recorded or validated."
        ),
    )
    args = parser.parse_args(argv)

    errors = validate_lite_scorer_v2_plan(
        args.root,
        require_recorded_evidence=args.require_recorded_evidence,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("lite_scorer_v2_plan_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
