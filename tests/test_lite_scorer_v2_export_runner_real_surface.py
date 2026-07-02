# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 export runner real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 export runner CLI."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_lite_scorer_v2_export.py"


def _write_run_manifest(root: Path) -> Path:
    """Write a complete Lite Scorer v2 run manifest under ``root``."""
    manifest = root / "benchmarks" / "lite_scorer_v2_run_manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        """
schema_version = "1.0.0"
plan_id = "lite-scorer-v2-run-plan"
student_candidate = "minilm_l6"
teacher_model = "training/output/deberta-v3-base-hallucination"
teacher_artifact = "training/output/deberta-v3-base-hallucination/model.safetensors"
student_base_model = "training/output/minilm-safetensors"
training_script = "training/train_distillation.py"
train_output_dir = "MODELS/lite-scorer-v2/student"
student_artifact = "MODELS/lite-scorer-v2/student/model.safetensors"
onnx_output_dir = "MODELS/lite-scorer-v2/onnx"
onnx_artifact = "MODELS/lite-scorer-v2/onnx/model_quantized.onnx"
model_card = "MODELS/lite-scorer-v2/model_card.md"
benchmark_claim_review = "benchmarks/lite_scorer_v2_claim_review.md"
heldout_eval_dataset = "benchmarks/heldout/lite_scorer_v2.jsonl"
heldout_source_dataset = "training/data/eval"
heldout_manifest = "benchmarks/heldout/lite_scorer_v2.manifest.toml"
heldout_target_rows = 1000
heldout_seed = 20260518
heldout_min_sources = 5
eval_result = "benchmarks/results/lite_scorer_v2_eval.json"
evidence_packet = "benchmarks/lite_scorer_v2_evidence_packet.toml"
latency_backend = "onnxruntime"
latency_device = "cpu"
epochs = 5
batch_size = 32
max_length = 256
temperature = 3.0
alpha = 0.5
learning_rate = 0.00005
training_seed = 20260518
eval_limit = 5000
num_workers = 2
device = "auto"
summ_target = 15000
general_target = 15000
latency_sample_count = 100
quantise_onnx = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_training_status(root: Path) -> Path:
    """Write a completed training run and student artefact for export readiness."""
    run_dir = (
        root
        / ".coordination"
        / "runs"
        / "DIRECTOR-AI"
        / "lite_scorer_v2_train_20260703T000000Z"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train.log").write_text(
        "EXIT 0 2026-07-03T00:00:00Z\n", encoding="utf-8"
    )
    student = root / "MODELS" / "lite-scorer-v2" / "student" / "model.safetensors"
    student.parent.mkdir(parents=True, exist_ok=True)
    student.write_bytes(b"student-weights")
    manifest = student.parent / "training_run_manifest.json"
    manifest.write_text('{"status":"completed"}\n', encoding="utf-8")
    return run_dir


def _write_training_inputs(root: Path) -> None:
    """Write local inputs referenced by the temporary run manifest."""
    (root / "training" / "output" / "minilm-safetensors").mkdir(parents=True)
    (root / "training" / "data" / "eval").mkdir(parents=True)
    script = root / "training" / "train_distillation.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('training placeholder')\n", encoding="utf-8")


def _write_uv_export_shim(root: Path) -> Path:
    """Write a protocol-preserving uv shim that materialises the ONNX artefact."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "uv"
    shim.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if sys.argv[1:4] != ["run", "--frozen", "director-ai"]:
    raise SystemExit("unexpected uv command: " + " ".join(sys.argv[1:]))
try:
    output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
except (ValueError, IndexError) as exc:
    raise SystemExit("missing --output") from exc
output.mkdir(parents=True, exist_ok=True)
(output / "model_quantized.onnx").write_bytes(b"onnx-bytes")
""",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run_export_cli(
    root: Path,
    *,
    manifest: Path,
    run_dir: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the production export runner CLI against ``root``."""
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(root),
            "--manifest",
            str(manifest),
            "--run-dir",
            str(run_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
        env=env,
    )


def test_lite_scorer_v2_export_runner_unit_guard_has_real_cli_companion() -> None:
    """The export unit guard should be backed by real subprocess CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_export_runner.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_export_runner_real_surface.py" in category


def test_lite_scorer_v2_export_runner_cli_records_export_receipt(
    tmp_path: Path,
) -> None:
    """The production CLI should execute the planned export command and emit JSON."""
    _write_training_inputs(tmp_path)
    manifest = _write_run_manifest(tmp_path)
    run_dir = _write_training_status(tmp_path)
    shim_dir = _write_uv_export_shim(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"

    result = _run_export_cli(tmp_path, manifest=manifest, run_dir=run_dir, env=env)

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload == {
        "command": [
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
        ],
        "onnx_artifact": {
            "exists": True,
            "path": "MODELS/lite-scorer-v2/onnx/model_quantized.onnx",
            "size_bytes": len(b"onnx-bytes"),
        },
        "public_score_claim": False,
        "schema_version": "1.0.0",
        "status": "recorded",
    }


def test_lite_scorer_v2_export_runner_cli_rejects_unready_training(
    tmp_path: Path,
) -> None:
    """The production CLI should fail before training is export-ready."""
    _write_training_inputs(tmp_path)
    manifest = _write_run_manifest(tmp_path)
    run_dir = (
        tmp_path
        / ".coordination"
        / "runs"
        / "DIRECTOR-AI"
        / "lite_scorer_v2_train_20260703T000000Z"
    )
    run_dir.mkdir(parents=True)

    result = _run_export_cli(tmp_path, manifest=manifest, run_dir=run_dir)

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "Lite Scorer v2 training is not export-ready: state=stale, export_ready=False"
        in result.stderr
    )
