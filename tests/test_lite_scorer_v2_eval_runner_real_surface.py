# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evaluation runner real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 evaluation runner CLI."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_lite_scorer_v2_eval.py"


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


def _write_training_inputs(root: Path) -> None:
    """Write local inputs referenced by the temporary run manifest."""
    (root / "training" / "output" / "minilm-safetensors").mkdir(parents=True)
    (root / "training" / "data" / "eval").mkdir(parents=True)
    script = root / "training" / "train_distillation.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("print('training placeholder')\n", encoding="utf-8")


def _write_onnx_artifact(root: Path) -> Path:
    """Write the ONNX artefact required before evaluation."""
    artifact = root / "MODELS" / "lite-scorer-v2" / "onnx" / "model_quantized.onnx"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"onnx-bytes")
    return artifact


def _write_uv_eval_shim(root: Path) -> Path:
    """Write a protocol-preserving uv shim that materialises eval JSON."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "uv"
    shim.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import sys

expected_prefix = ["run", "--frozen", "python", "tools/eval_lite_scorer_v2.py"]
if sys.argv[1:5] != expected_prefix:
    raise SystemExit("unexpected uv command: " + " ".join(sys.argv[1:]))
try:
    output = pathlib.Path(sys.argv[sys.argv.index("--output") + 1])
except (ValueError, IndexError) as exc:
    raise SystemExit("missing --output") from exc
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({
    "heldout_eval_balanced_accuracy": 0.71,
    "heldout_eval_rows": 1000,
    "heldout_eval_threshold": 0.53,
    "latency_p50_ms": 2.4,
    "latency_p95_ms": 4.8,
    "latency_sample_count": 100,
}) + "\\n", encoding="utf-8")
argv_receipt = pathlib.Path("benchmarks/results/lite_scorer_v2_eval_argv.json")
argv_receipt.parent.mkdir(parents=True, exist_ok=True)
argv_receipt.write_text(json.dumps(sys.argv[1:]) + "\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR)
    return bin_dir


def _run_eval_cli(
    root: Path,
    *,
    manifest: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the production evaluation runner CLI against ``root``."""
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(root),
            "--manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
        env=env,
    )


def test_lite_scorer_v2_eval_runner_unit_guard_has_real_cli_companion() -> None:
    """The eval unit guard should be backed by real subprocess CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_lite_scorer_v2_eval_runner.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_lite_scorer_v2_eval_runner_real_surface.py" in category


def test_lite_scorer_v2_eval_runner_cli_records_evaluation_receipt(
    tmp_path: Path,
) -> None:
    """The production CLI should execute the planned evaluation command."""
    _write_training_inputs(tmp_path)
    manifest = _write_run_manifest(tmp_path)
    _write_onnx_artifact(tmp_path)
    shim_dir = _write_uv_eval_shim(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{shim_dir}{os.pathsep}{env['PATH']}"

    result = _run_eval_cli(tmp_path, manifest=manifest, env=env)

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    expected_command = [
        "uv",
        "run",
        "--frozen",
        "python",
        "tools/eval_lite_scorer_v2.py",
        "--dataset",
        "benchmarks/heldout/lite_scorer_v2.jsonl",
        "--model-path",
        "MODELS/lite-scorer-v2/onnx",
        "--latency-sample-count",
        "100",
        "--output",
        "benchmarks/results/lite_scorer_v2_eval.json",
    ]
    assert payload == {
        "command": expected_command,
        "eval_result": {
            "exists": True,
            "path": "benchmarks/results/lite_scorer_v2_eval.json",
            "size_bytes": (
                tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
            )
            .stat()
            .st_size,
        },
        "public_score_claim": False,
        "schema_version": "1.0.0",
        "status": "recorded",
    }
    argv_receipt = tmp_path / "benchmarks" / "results" / "lite_scorer_v2_eval_argv.json"
    assert json.loads(argv_receipt.read_text(encoding="utf-8")) == expected_command[1:]


def test_lite_scorer_v2_eval_runner_cli_rejects_missing_onnx(
    tmp_path: Path,
) -> None:
    """The production CLI should fail before eval when the ONNX file is absent."""
    _write_training_inputs(tmp_path)
    manifest = _write_run_manifest(tmp_path)

    result = _run_eval_cli(tmp_path, manifest=manifest)

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "Lite Scorer v2 ONNX artifact is missing: "
        "MODELS/lite-scorer-v2/onnx/model_quantized.onnx"
    ) in result.stderr
