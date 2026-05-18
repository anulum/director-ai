# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 run planner tests

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "tools" / "plan_lite_scorer_v2_run.py"
SPEC = importlib.util.spec_from_file_location("plan_lite_scorer_v2_run", PLANNER)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

build_lite_scorer_v2_run_plan = MODULE.build_lite_scorer_v2_run_plan
validate_lite_scorer_v2_run_manifest = MODULE.validate_lite_scorer_v2_run_manifest


def _write_training_script(root: Path) -> None:
    (root / "training").mkdir()
    (root / "training" / "train_distillation.py").write_text(
        "print('training placeholder')\n",
        encoding="utf-8",
    )
    (root / "training" / "data" / "eval").mkdir(parents=True)
    (root / "training" / "output" / "minilm-safetensors").mkdir(parents=True)


def _write_manifest(root: Path, **overrides: object) -> Path:
    (root / "benchmarks").mkdir()
    values: dict[str, object] = {
        "schema_version": "1.0.0",
        "plan_id": "lite-scorer-v2-run-plan",
        "student_candidate": "minilm_l6",
        "teacher_model": "training/output/deberta-v3-base-hallucination",
        "teacher_artifact": "training/output/deberta-v3-base-hallucination/model.safetensors",
        "student_base_model": "training/output/minilm-safetensors",
        "training_script": "training/train_distillation.py",
        "train_output_dir": "MODELS/lite-scorer-v2/student",
        "student_artifact": "MODELS/lite-scorer-v2/student/model.safetensors",
        "onnx_output_dir": "MODELS/lite-scorer-v2/onnx",
        "onnx_artifact": "MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
        "heldout_eval_dataset": "benchmarks/heldout/lite_scorer_v2.jsonl",
        "heldout_source_dataset": "training/data/eval",
        "heldout_manifest": "benchmarks/heldout/lite_scorer_v2.manifest.toml",
        "heldout_target_rows": 1000,
        "heldout_seed": 20260518,
        "heldout_min_sources": 5,
        "eval_result": "benchmarks/results/lite_scorer_v2_eval.json",
        "evidence_packet": "benchmarks/lite_scorer_v2_evidence_packet.toml",
        "latency_backend": "onnxruntime",
        "latency_device": "cpu",
        "epochs": 5,
        "batch_size": 32,
        "max_length": 256,
        "temperature": 3.0,
        "alpha": 0.5,
        "learning_rate": 0.00005,
        "training_seed": 20260518,
        "eval_limit": 5000,
        "num_workers": 2,
        "device": "auto",
        "summ_target": 15000,
        "general_target": 15000,
        "latency_sample_count": 100,
        "quantise_onnx": True,
    }
    values.update(overrides)
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int | float):
            rendered = str(value)
        else:
            rendered = f'"{value}"'
        lines.append(f"{key} = {rendered}")
    manifest = root / "benchmarks" / "lite_scorer_v2_run_manifest.toml"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_lite_scorer_v2_run_manifest_validates_current_package() -> None:
    assert validate_lite_scorer_v2_run_manifest(ROOT) == []


def test_lite_scorer_v2_run_manifest_can_require_local_training_inputs(
    tmp_path: Path,
) -> None:
    _write_training_script(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        student_base_model="training/output/missing",
        heldout_source_dataset="training/data/missing",
    )

    assert validate_lite_scorer_v2_run_manifest(tmp_path, manifest) == []
    assert validate_lite_scorer_v2_run_manifest(
        tmp_path,
        manifest,
        require_local_inputs=True,
    ) == [
        "benchmarks/lite_scorer_v2_run_manifest.toml: student_base_model does not exist",
        "benchmarks/lite_scorer_v2_run_manifest.toml: heldout_source_dataset does not exist",
    ]


def test_lite_scorer_v2_run_plan_emits_ordered_argv_commands(tmp_path: Path) -> None:
    _write_training_script(tmp_path)
    manifest = _write_manifest(tmp_path)

    plan, errors = build_lite_scorer_v2_run_plan(tmp_path, manifest)

    assert errors == []
    assert plan["claim_boundary"] == (
        "Run plan only; evidence remains unrecorded until every emitted command succeeds "
        "and the recorder validates real artefacts."
    )
    commands = plan["commands"]
    assert [command["name"] for command in commands] == [
        "build_heldout",
        "train",
        "export_onnx",
        "evaluate",
        "record_evidence",
    ]
    assert all(isinstance(command["argv"], list) for command in commands)
    assert all("command" not in command for command in commands)
    assert commands[0]["argv"] == [
        "uv",
        "run",
        "--frozen",
        "python",
        "tools/build_lite_scorer_v2_heldout.py",
        "--source",
        "training/data/eval",
        "--output",
        "benchmarks/heldout/lite_scorer_v2.jsonl",
        "--manifest",
        "benchmarks/heldout/lite_scorer_v2.manifest.toml",
        "--target-rows",
        "1000",
        "--seed",
        "20260518",
        "--min-sources",
        "5",
    ]
    train_argv = commands[1]["argv"]
    assert train_argv[:4] == [
        "uv",
        "run",
        "--frozen",
        "python",
    ]
    assert "--teacher" in train_argv
    assert "--student" in train_argv
    assert "--output-dir" in train_argv
    assert "--seed" in train_argv
    assert "--eval-limit" in train_argv
    assert "--num-workers" in train_argv
    assert "--device" in train_argv
    assert commands[2]["argv"] == [
        "uv",
        "run",
        "--frozen",
        "director-ai",
        "export",
        "--format",
        "onnx",
        "--model",
        "MODELS/lite-scorer-v2/student",
        "--output",
        "MODELS/lite-scorer-v2/onnx",
        "--quantize",
        "int8",
    ]
    assert commands[4]["argv"][-2:] == [
        "--output",
        "benchmarks/lite_scorer_v2_evidence_packet.toml",
    ]


def test_lite_scorer_v2_run_plan_rejects_unsupported_student(tmp_path: Path) -> None:
    _write_training_script(tmp_path)
    manifest = _write_manifest(tmp_path, student_candidate="unsupported")

    errors = validate_lite_scorer_v2_run_manifest(
        tmp_path,
        manifest,
        require_local_inputs=True,
    )

    assert errors == [
        "benchmarks/lite_scorer_v2_run_manifest.toml: unsupported student_candidate 'unsupported'"
    ]


def test_lite_scorer_v2_run_plan_rejects_path_traversal(tmp_path: Path) -> None:
    _write_training_script(tmp_path)
    manifest = _write_manifest(tmp_path, train_output_dir="../outside")

    errors = validate_lite_scorer_v2_run_manifest(
        tmp_path,
        manifest,
        require_local_inputs=True,
    )

    assert errors == [
        "benchmarks/lite_scorer_v2_run_manifest.toml: train_output_dir must be a relative path inside the repository"
    ]


def test_lite_scorer_v2_run_plan_rejects_missing_heldout_source(
    tmp_path: Path,
) -> None:
    _write_training_script(tmp_path)
    manifest = _write_manifest(tmp_path, heldout_source_dataset="training/data/missing")

    errors = validate_lite_scorer_v2_run_manifest(
        tmp_path,
        manifest,
        require_local_inputs=True,
    )

    assert errors == [
        "benchmarks/lite_scorer_v2_run_manifest.toml: heldout_source_dataset does not exist"
    ]


def test_lite_scorer_v2_run_plan_rejects_missing_student_base(
    tmp_path: Path,
) -> None:
    _write_training_script(tmp_path)
    manifest = _write_manifest(tmp_path, student_base_model="training/output/missing")

    errors = validate_lite_scorer_v2_run_manifest(
        tmp_path,
        manifest,
        require_local_inputs=True,
    )

    assert errors == [
        "benchmarks/lite_scorer_v2_run_manifest.toml: student_base_model does not exist"
    ]


def test_lite_scorer_v2_run_plan_omits_quantisation_flag_when_disabled(
    tmp_path: Path,
) -> None:
    _write_training_script(tmp_path)
    manifest = _write_manifest(
        tmp_path,
        quantise_onnx=False,
        onnx_artifact="MODELS/lite-scorer-v2/onnx/model.onnx",
    )

    plan, errors = build_lite_scorer_v2_run_plan(tmp_path, manifest)

    assert errors == []
    assert "--quantize" not in plan["commands"][2]["argv"]
    assert "MODELS/lite-scorer-v2/onnx/model.onnx" in plan["commands"][4]["argv"]
