#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 reproducible run planner

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

MANIFEST = Path("benchmarks/lite_scorer_v2_run_manifest.toml")
CLAIM_BOUNDARY = (
    "Run plan only; evidence remains unrecorded until every emitted command succeeds "
    "and the recorder validates real artefacts."
)
REQUIRED_STUDENTS = {"minilm_l6", "mobilebert", "distilbert"}
REQUIRED_STRINGS = {
    "schema_version",
    "plan_id",
    "student_candidate",
    "teacher_model",
    "teacher_artifact",
    "student_base_model",
    "training_script",
    "train_output_dir",
    "student_artifact",
    "onnx_output_dir",
    "onnx_artifact",
    "model_card",
    "benchmark_claim_review",
    "heldout_eval_dataset",
    "heldout_source_dataset",
    "heldout_manifest",
    "eval_result",
    "evidence_packet",
    "latency_backend",
    "latency_device",
    "device",
}
REQUIRED_INTS = {
    "epochs": 1,
    "batch_size": 1,
    "max_length": 1,
    "summ_target": 1,
    "general_target": 1,
    "latency_sample_count": 100,
    "heldout_target_rows": 2,
    "heldout_seed": 0,
    "heldout_min_sources": 1,
    "training_seed": 0,
    "eval_limit": 1,
    "num_workers": 0,
}
REQUIRED_FLOATS = {
    "temperature": 0.0,
    "alpha": 0.0,
    "learning_rate": 0.0,
}
PATH_FIELDS = {
    "teacher_artifact",
    "student_base_model",
    "training_script",
    "train_output_dir",
    "student_artifact",
    "onnx_output_dir",
    "onnx_artifact",
    "model_card",
    "benchmark_claim_review",
    "heldout_eval_dataset",
    "heldout_source_dataset",
    "heldout_manifest",
    "eval_result",
    "evidence_packet",
}
GENERATED_PATH_FIELDS = {
    "teacher_artifact",
    "train_output_dir",
    "student_artifact",
    "onnx_output_dir",
    "onnx_artifact",
    "model_card",
    "benchmark_claim_review",
    "heldout_eval_dataset",
    "heldout_source_dataset",
    "heldout_manifest",
    "eval_result",
    "evidence_packet",
}


def _display(path: Path) -> str:
    return path.as_posix()


def _load_manifest(manifest: Path) -> tuple[dict[str, Any], list[str]]:
    if not manifest.exists():
        return {}, [f"{_display(manifest)}: missing Lite Scorer v2 run manifest"]
    try:
        return tomllib.loads(manifest.read_text(encoding="utf-8")), []
    except tomllib.TOMLDecodeError as exc:
        return {}, [f"{_display(manifest)}: invalid TOML: {exc}"]


def _manifest_label(root: Path, manifest: Path) -> str:
    try:
        return manifest.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return manifest.as_posix()


def _is_safe_relative_path(value: str) -> bool:
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts and value.strip() == value


def _validate_strings(data: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    for field in sorted(REQUIRED_STRINGS):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label}: {field} must be a non-empty string")
    return errors


def _validate_numbers(data: dict[str, Any], label: str) -> list[str]:
    errors: list[str] = []
    for field, minimum in REQUIRED_INTS.items():
        value = data.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            errors.append(f"{label}: {field} must be an integer >= {minimum}")
    for field, min_value in REQUIRED_FLOATS.items():
        value = data.get(field)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or value <= min_value
        ):
            errors.append(f"{label}: {field} must be numeric and > {min_value}")
    alpha = data.get("alpha")
    if isinstance(alpha, int | float) and not isinstance(alpha, bool) and alpha > 1.0:
        errors.append(f"{label}: alpha must be in (0, 1]")
    return errors


def _validate_paths(
    root: Path,
    data: dict[str, Any],
    label: str,
    *,
    require_local_inputs: bool,
) -> list[str]:
    errors: list[str] = []
    for field in sorted(PATH_FIELDS):
        value = data.get(field)
        if not isinstance(value, str):
            continue
        if not _is_safe_relative_path(value):
            errors.append(
                f"{label}: {field} must be a relative path inside the repository"
            )
    script = data.get("training_script")
    if (
        isinstance(script, str)
        and _is_safe_relative_path(script)
        and not (root / script).is_file()
    ):
        errors.append(f"{label}: training_script does not exist")
    if require_local_inputs:
        student_base = data.get("student_base_model")
        if (
            isinstance(student_base, str)
            and _is_safe_relative_path(student_base)
            and not (root / student_base).exists()
        ):
            errors.append(f"{label}: student_base_model does not exist")
        heldout_source = data.get("heldout_source_dataset")
        if (
            isinstance(heldout_source, str)
            and _is_safe_relative_path(heldout_source)
            and not (root / heldout_source).exists()
        ):
            errors.append(f"{label}: heldout_source_dataset does not exist")
    return errors


def validate_lite_scorer_v2_run_manifest(
    root: Path,
    manifest_path: Path = MANIFEST,
    *,
    require_local_inputs: bool = False,
) -> list[str]:
    root = root.resolve()
    manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    label = _manifest_label(root, manifest)
    data, errors = _load_manifest(manifest)
    if errors:
        return errors

    errors.extend(_validate_strings(data, label))
    errors.extend(_validate_numbers(data, label))
    if data.get("schema_version") != "1.0.0":
        errors.append(f"{label}: schema_version must be '1.0.0'")
    if data.get("plan_id") != "lite-scorer-v2-run-plan":
        errors.append(f"{label}: plan_id must be 'lite-scorer-v2-run-plan'")
    candidate = data.get("student_candidate")
    if isinstance(candidate, str) and candidate not in REQUIRED_STUDENTS:
        errors.append(f"{label}: unsupported student_candidate {candidate!r}")
    quantise = data.get("quantise_onnx")
    if not isinstance(quantise, bool):
        errors.append(f"{label}: quantise_onnx must be boolean")
    device = data.get("device")
    if isinstance(device, str) and device not in {"auto", "cpu", "cuda"}:
        errors.append(f"{label}: device must be one of auto, cpu, or cuda")
    target_rows = data.get("heldout_target_rows")
    if (
        isinstance(target_rows, int)
        and not isinstance(target_rows, bool)
        and target_rows % 2 != 0
    ):
        errors.append(f"{label}: heldout_target_rows must be even")
    errors.extend(
        _validate_paths(
            root,
            data,
            label,
            require_local_inputs=require_local_inputs,
        )
    )
    return errors


def _path(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, str):
        raise TypeError(f"{field} must be string")
    return value


def _int_arg(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be integer")
    return str(value)


def _float_arg(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    return str(value)


def _build_commands(data: dict[str, Any]) -> list[dict[str, list[str] | str]]:
    build_heldout = [
        "uv",
        "run",
        "--frozen",
        "python",
        "tools/build_lite_scorer_v2_heldout.py",
        "--source",
        _path(data, "heldout_source_dataset"),
        "--output",
        _path(data, "heldout_eval_dataset"),
        "--manifest",
        _path(data, "heldout_manifest"),
        "--target-rows",
        _int_arg(data, "heldout_target_rows"),
        "--seed",
        _int_arg(data, "heldout_seed"),
        "--min-sources",
        _int_arg(data, "heldout_min_sources"),
    ]
    train = [
        "uv",
        "run",
        "--frozen",
        "python",
        _path(data, "training_script"),
        "--teacher",
        _path(data, "teacher_model"),
        "--student",
        _path(data, "student_base_model"),
        "--epochs",
        _int_arg(data, "epochs"),
        "--batch-size",
        _int_arg(data, "batch_size"),
        "--max-length",
        _int_arg(data, "max_length"),
        "--temperature",
        _float_arg(data, "temperature"),
        "--alpha",
        _float_arg(data, "alpha"),
        "--lr",
        _float_arg(data, "learning_rate"),
        "--seed",
        _int_arg(data, "training_seed"),
        "--eval-limit",
        _int_arg(data, "eval_limit"),
        "--num-workers",
        _int_arg(data, "num_workers"),
        "--device",
        _path(data, "device"),
        "--summ-target",
        _int_arg(data, "summ_target"),
        "--general-target",
        _int_arg(data, "general_target"),
        "--output-dir",
        _path(data, "train_output_dir"),
    ]
    export = [
        "uv",
        "run",
        "--frozen",
        "director-ai",
        "export",
        "--format",
        "onnx",
        "--model",
        _path(data, "train_output_dir"),
        "--output",
        _path(data, "onnx_output_dir"),
    ]
    if data["quantise_onnx"] is True:
        export.extend(["--quantize", "int8"])
    evaluate = [
        "uv",
        "run",
        "--frozen",
        "python",
        "tools/eval_lite_scorer_v2.py",
        "--dataset",
        _path(data, "heldout_eval_dataset"),
        "--model-path",
        _path(data, "onnx_output_dir"),
        "--latency-sample-count",
        _int_arg(data, "latency_sample_count"),
        "--output",
        _path(data, "eval_result"),
    ]
    record = [
        "uv",
        "run",
        "--frozen",
        "python",
        "tools/record_lite_scorer_v2_evidence.py",
        ".",
        "--eval-result",
        _path(data, "eval_result"),
        "--student-candidate",
        _path(data, "student_candidate"),
        "--student-artifact",
        _path(data, "student_artifact"),
        "--teacher-artifact",
        _path(data, "teacher_artifact"),
        "--onnx-artifact",
        _path(data, "onnx_artifact"),
        "--model-card",
        _path(data, "model_card"),
        "--benchmark-claim-review",
        _path(data, "benchmark_claim_review"),
        "--latency-backend",
        _path(data, "latency_backend"),
        "--latency-device",
        _path(data, "latency_device"),
        "--output",
        _path(data, "evidence_packet"),
    ]
    return [
        {"name": "build_heldout", "argv": build_heldout},
        {"name": "train", "argv": train},
        {"name": "export_onnx", "argv": export},
        {"name": "evaluate", "argv": evaluate},
        {"name": "record_evidence", "argv": record},
    ]


def build_lite_scorer_v2_run_plan(
    root: Path,
    manifest_path: Path = MANIFEST,
) -> tuple[dict[str, Any], list[str]]:
    root = root.resolve()
    manifest = manifest_path if manifest_path.is_absolute() else root / manifest_path
    errors = validate_lite_scorer_v2_run_manifest(root, manifest)
    if errors:
        return {}, errors
    data, load_errors = _load_manifest(manifest)
    if load_errors:
        return {}, load_errors
    plan = {
        "schema_version": "1.0.0",
        "plan_id": data["plan_id"],
        "manifest": _manifest_label(root, manifest),
        "student_candidate": data["student_candidate"],
        "claim_boundary": CLAIM_BOUNDARY,
        "generated_paths": {
            field: data[field] for field in sorted(GENERATED_PATH_FIELDS)
        },
        "commands": _build_commands(data),
    }
    return plan, []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Repository root")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="Lite Scorer v2 run manifest path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan, errors = build_lite_scorer_v2_run_plan(args.root, args.manifest)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
