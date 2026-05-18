#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evidence packet recorder

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

TOOLS_DIR = Path(__file__).resolve().parent
VALIDATOR = TOOLS_DIR / "validate_lite_scorer_v2_plan.py"
SPEC = importlib.util.spec_from_file_location("validate_lite_scorer_v2_plan", VALIDATOR)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load {VALIDATOR}")
VALIDATOR_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VALIDATOR_MODULE
SPEC.loader.exec_module(VALIDATOR_MODULE)

DEFAULT_EVIDENCE_PACKET: Path = VALIDATOR_MODULE.DEFAULT_EVIDENCE_PACKET
REQUIRED_STUDENTS: set[str] = VALIDATOR_MODULE.REQUIRED_STUDENTS
validate_lite_scorer_v2_plan = cast(
    Callable[[Path], list[str]],
    VALIDATOR_MODULE.validate_lite_scorer_v2_plan,
)
EVAL_RESULT_FIELDS = {
    "heldout_eval_dataset",
    "heldout_eval_rows",
    "heldout_eval_balanced_accuracy",
    "heldout_eval_threshold",
    "latency_sample_count",
    "latency_p50_ms",
    "latency_p95_ms",
}


@dataclass(frozen=True)
class EvidenceRecord:
    student_candidate: str
    student_artifact: Path
    teacher_artifact: Path
    heldout_eval_dataset: Path
    heldout_eval_rows: int
    heldout_eval_balanced_accuracy: float
    heldout_eval_threshold: float
    onnx_artifact: Path
    latency_backend: str
    latency_device: str
    latency_sample_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    output: Path = DEFAULT_EVIDENCE_PACKET


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(root: Path, path: Path) -> str:
    absolute = path if path.is_absolute() else root / path
    try:
        return absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _validate_record_inputs(root: Path, record: EvidenceRecord) -> list[str]:
    errors: list[str] = []
    if record.student_candidate not in REQUIRED_STUDENTS:
        errors.append(
            f"{DEFAULT_EVIDENCE_PACKET}: unsupported student_candidate {record.student_candidate!r}"
        )
    for artifact in (
        record.student_artifact,
        record.teacher_artifact,
        record.onnx_artifact,
    ):
        path = artifact if artifact.is_absolute() else root / artifact
        if not path.is_file():
            errors.append(f"{artifact}: artifact file does not exist")

    if record.heldout_eval_rows < 1:
        errors.append(f"{DEFAULT_EVIDENCE_PACKET}: heldout_eval_rows must be positive")
    for field, value in (
        ("heldout_eval_balanced_accuracy", record.heldout_eval_balanced_accuracy),
        ("heldout_eval_threshold", record.heldout_eval_threshold),
    ):
        if value < 0.0 or value > 1.0:
            errors.append(f"{DEFAULT_EVIDENCE_PACKET}: {field} must be in [0, 1]")

    if not record.latency_backend.strip():
        errors.append(f"{DEFAULT_EVIDENCE_PACKET}: latency_backend must be non-empty")
    if not record.latency_device.strip():
        errors.append(f"{DEFAULT_EVIDENCE_PACKET}: latency_device must be non-empty")
    if record.latency_sample_count < 100:
        errors.append(
            f"{DEFAULT_EVIDENCE_PACKET}: latency_sample_count must be at least 100"
        )
    if record.latency_p50_ms <= 0.0:
        errors.append(
            f"{DEFAULT_EVIDENCE_PACKET}: quantized_latency_status recorded requires latency_p50_ms"
        )
    if record.latency_p95_ms <= 0.0:
        errors.append(
            f"{DEFAULT_EVIDENCE_PACKET}: quantized_latency_status recorded requires latency_p95_ms"
        )
    if record.latency_p95_ms < record.latency_p50_ms:
        errors.append(
            f"{DEFAULT_EVIDENCE_PACKET}: latency_p95_ms must be greater than latency_p50_ms"
        )
    return errors


def _render_packet(root: Path, record: EvidenceRecord) -> str:
    student_artifact = (
        record.student_artifact
        if record.student_artifact.is_absolute()
        else root / record.student_artifact
    )
    teacher_artifact = (
        record.teacher_artifact
        if record.teacher_artifact.is_absolute()
        else root / record.teacher_artifact
    )
    onnx_artifact = (
        record.onnx_artifact
        if record.onnx_artifact.is_absolute()
        else root / record.onnx_artifact
    )

    fields: list[tuple[str, str]] = [
        ("schema_version", _toml_string("1.0.0")),
        ("packet_id", _toml_string("lite-scorer-v2-evidence")),
        ("public_score_claim", "false"),
        ("student_artifact_status", _toml_string("recorded")),
        ("student_candidate", _toml_string(record.student_candidate)),
        (
            "student_artifact_path",
            _toml_string(_display_path(root, record.student_artifact)),
        ),
        ("student_artifact_sha256", _toml_string(_sha256(student_artifact))),
        ("teacher_artifact_status", _toml_string("recorded")),
        (
            "teacher_artifact_path",
            _toml_string(_display_path(root, record.teacher_artifact)),
        ),
        ("teacher_artifact_sha256", _toml_string(_sha256(teacher_artifact))),
        ("heldout_eval_status", _toml_string("recorded")),
        (
            "heldout_eval_dataset",
            _toml_string(_display_path(root, record.heldout_eval_dataset)),
        ),
        ("heldout_eval_rows", str(record.heldout_eval_rows)),
        ("heldout_eval_balanced_accuracy", repr(record.heldout_eval_balanced_accuracy)),
        ("heldout_eval_threshold", repr(record.heldout_eval_threshold)),
        ("onnx_export_status", _toml_string("recorded")),
        ("onnx_artifact_path", _toml_string(_display_path(root, record.onnx_artifact))),
        ("onnx_artifact_sha256", _toml_string(_sha256(onnx_artifact))),
        ("quantized_latency_status", _toml_string("recorded")),
        ("latency_backend", _toml_string(record.latency_backend)),
        ("latency_device", _toml_string(record.latency_device)),
        ("latency_sample_count", str(record.latency_sample_count)),
        ("latency_p50_ms", repr(record.latency_p50_ms)),
        ("latency_p95_ms", repr(record.latency_p95_ms)),
        (
            "claim_boundary",
            _toml_string(
                "Recorded evidence packet only; no public score claim until an operator reviews and approves the scored release."
            ),
        ),
    ]
    header = [
        "# SPDX-License-Identifier: AGPL-3.0-or-later",
        "# Commercial licence available",
        "# Concepts 1996-2026 Miroslav Sotek. All rights reserved.",
        "# Code 2020-2026 Miroslav Sotek. All rights reserved.",
        "# ORCID: 0009-0009-3560-0851",
        "# Contact: www.anulum.li | protoscience@anulum.li",
        "# Director-Class AI - Lite Scorer v2 recorded evidence packet",
        "",
    ]
    body = [f"{key} = {value}" for key, value in fields]
    return "\n".join(header + body) + "\n"


def record_lite_scorer_v2_evidence(root: Path, record: EvidenceRecord) -> list[str]:
    root = root.resolve()
    errors = _validate_record_inputs(root, record)
    if errors:
        return errors

    output = record.output if record.output.is_absolute() else root / record.output
    output.parent.mkdir(parents=True, exist_ok=True)
    original = output.read_text(encoding="utf-8") if output.exists() else None
    output.write_text(_render_packet(root, record), encoding="utf-8")
    validation_errors = validate_lite_scorer_v2_plan(root)
    if validation_errors:
        if original is None:
            output.unlink(missing_ok=True)
        else:
            output.write_text(original, encoding="utf-8")
        return validation_errors
    return []


def _load_eval_result(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, [f"{path}: eval result file does not exist"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, [f"{path}: invalid JSON: {exc}"]
    if not isinstance(payload, dict):
        return {}, [f"{path}: eval result must be a JSON object"]
    missing = sorted(EVAL_RESULT_FIELDS - set(payload))
    if missing:
        return {}, [f"{path}: missing required fields {', '.join(missing)}"]
    return payload, []


def _number(payload: dict[str, Any], path: Path, field: str) -> tuple[float, list[str]]:
    value = payload[field]
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value), []
    return 0.0, [f"{path}: {field} must be numeric"]


def _integer(payload: dict[str, Any], path: Path, field: str) -> tuple[int, list[str]]:
    value = payload[field]
    if isinstance(value, int) and not isinstance(value, bool):
        return value, []
    return 0, [f"{path}: {field} must be an integer"]


def record_lite_scorer_v2_evidence_from_eval_result(
    *,
    root: Path,
    eval_result: Path,
    student_candidate: str,
    student_artifact: Path,
    teacher_artifact: Path,
    onnx_artifact: Path,
    latency_backend: str,
    latency_device: str,
    output: Path = DEFAULT_EVIDENCE_PACKET,
) -> list[str]:
    payload, errors = _load_eval_result(eval_result)
    if errors:
        return errors

    rows, row_errors = _integer(payload, eval_result, "heldout_eval_rows")
    balanced_accuracy, ba_errors = _number(
        payload,
        eval_result,
        "heldout_eval_balanced_accuracy",
    )
    threshold, threshold_errors = _number(
        payload,
        eval_result,
        "heldout_eval_threshold",
    )
    latency_sample_count, sample_errors = _integer(
        payload,
        eval_result,
        "latency_sample_count",
    )
    latency_p50_ms, p50_errors = _number(payload, eval_result, "latency_p50_ms")
    latency_p95_ms, p95_errors = _number(payload, eval_result, "latency_p95_ms")
    errors.extend(
        row_errors
        + ba_errors
        + threshold_errors
        + sample_errors
        + p50_errors
        + p95_errors
    )
    dataset = payload["heldout_eval_dataset"]
    if not isinstance(dataset, str) or not dataset.strip():
        errors.append(f"{eval_result}: heldout_eval_dataset must be a non-empty string")
    if errors:
        return errors

    record = EvidenceRecord(
        student_candidate=student_candidate,
        student_artifact=student_artifact,
        teacher_artifact=teacher_artifact,
        heldout_eval_dataset=Path(dataset),
        heldout_eval_rows=rows,
        heldout_eval_balanced_accuracy=balanced_accuracy,
        heldout_eval_threshold=threshold,
        onnx_artifact=onnx_artifact,
        latency_backend=latency_backend,
        latency_device=latency_device,
        latency_sample_count=latency_sample_count,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        output=output,
    )
    return record_lite_scorer_v2_evidence(root, record)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Repository root")
    parser.add_argument("--eval-result", type=Path)
    parser.add_argument(
        "--student-candidate", required=True, choices=sorted(REQUIRED_STUDENTS)
    )
    parser.add_argument("--student-artifact", required=True, type=Path)
    parser.add_argument("--teacher-artifact", required=True, type=Path)
    parser.add_argument("--heldout-eval-dataset", type=Path)
    parser.add_argument("--heldout-eval-rows", type=int)
    parser.add_argument("--heldout-eval-balanced-accuracy", type=float)
    parser.add_argument("--heldout-eval-threshold", type=float)
    parser.add_argument("--onnx-artifact", required=True, type=Path)
    parser.add_argument("--latency-backend", required=True)
    parser.add_argument("--latency-device", required=True)
    parser.add_argument("--latency-sample-count", type=int)
    parser.add_argument("--latency-p50-ms", type=float)
    parser.add_argument("--latency-p95-ms", type=float)
    parser.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE_PACKET)
    return parser


def _missing_manual_args(args: argparse.Namespace) -> list[str]:
    required = (
        "heldout_eval_dataset",
        "heldout_eval_rows",
        "heldout_eval_balanced_accuracy",
        "heldout_eval_threshold",
        "latency_sample_count",
        "latency_p50_ms",
        "latency_p95_ms",
    )
    return [
        f"--{field.replace('_', '-')}"
        for field in required
        if getattr(args, field) is None
    ]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.eval_result is not None:
        errors = record_lite_scorer_v2_evidence_from_eval_result(
            root=args.root,
            eval_result=args.eval_result,
            student_candidate=args.student_candidate,
            student_artifact=args.student_artifact,
            teacher_artifact=args.teacher_artifact,
            onnx_artifact=args.onnx_artifact,
            latency_backend=args.latency_backend,
            latency_device=args.latency_device,
            output=args.output,
        )
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("lite_scorer_v2_evidence_recorded")
        return 0

    missing = _missing_manual_args(args)
    if missing:
        print(
            f"missing required arguments without --eval-result: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 1
    record = EvidenceRecord(
        student_candidate=args.student_candidate,
        student_artifact=args.student_artifact,
        teacher_artifact=args.teacher_artifact,
        heldout_eval_dataset=args.heldout_eval_dataset,
        heldout_eval_rows=args.heldout_eval_rows,
        heldout_eval_balanced_accuracy=args.heldout_eval_balanced_accuracy,
        heldout_eval_threshold=args.heldout_eval_threshold,
        onnx_artifact=args.onnx_artifact,
        latency_backend=args.latency_backend,
        latency_device=args.latency_device,
        latency_sample_count=args.latency_sample_count,
        latency_p50_ms=args.latency_p50_ms,
        latency_p95_ms=args.latency_p95_ms,
        output=args.output,
    )
    errors = record_lite_scorer_v2_evidence(args.root, record)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("lite_scorer_v2_evidence_recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
