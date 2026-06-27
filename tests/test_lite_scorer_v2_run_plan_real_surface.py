# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 run planner real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 run planner CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "tools" / "plan_lite_scorer_v2_run.py"


def _write_training_inputs(root: Path) -> None:
    """Write local training inputs required by the temporary run manifest."""
    (root / "training" / "data" / "eval").mkdir(parents=True)
    (root / "training" / "output" / "minilm-safetensors").mkdir(parents=True)
    training_script = root / "training" / "train_distillation.py"
    training_script.parent.mkdir(parents=True, exist_ok=True)
    training_script.write_text(
        "from __future__ import annotations\n\nprint('training placeholder')\n",
        encoding="utf-8",
    )


def _write_run_manifest(root: Path, *, student_candidate: str = "minilm_l6") -> Path:
    """Write a complete Lite Scorer v2 run manifest under ``root``."""
    manifest = root / "benchmarks" / "lite_scorer_v2_run_manifest.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"""
schema_version = "1.0.0"
plan_id = "lite-scorer-v2-run-plan"
student_candidate = "{student_candidate}"
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


def _load_plan(stdout: str) -> Mapping[str, object]:
    """Decode the planner JSON object emitted by the production CLI."""
    data = json.loads(stdout)
    assert isinstance(data, dict)
    return cast(Mapping[str, object], data)


def _command_argv(command: Mapping[str, object]) -> list[str]:
    """Return the argv vector from an emitted planner command."""
    argv = command["argv"]
    assert isinstance(argv, list)
    assert all(isinstance(part, str) for part in argv)
    return cast(list[str], argv)


def _commands(plan: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return typed command mappings from the decoded run plan."""
    commands = plan["commands"]
    assert isinstance(commands, list)
    assert all(isinstance(command, dict) for command in commands)
    return cast(list[Mapping[str, object]], commands)


def test_lite_scorer_v2_run_planner_cli_emits_reproducible_argv(
    tmp_path: Path,
) -> None:
    """The production CLI should emit deterministic argv commands, not shell text."""
    _write_training_inputs(tmp_path)
    manifest = _write_run_manifest(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(PLANNER),
            str(tmp_path),
            "--manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    plan = _load_plan(result.stdout)
    assert plan["claim_boundary"] == (
        "Run plan only; evidence remains unrecorded until every emitted command "
        "succeeds and the recorder validates real artefacts."
    )
    commands = _commands(plan)
    assert [command["name"] for command in commands] == [
        "build_heldout",
        "train",
        "export_onnx",
        "evaluate",
        "record_evidence",
    ]
    assert all("command" not in command for command in commands)
    assert _command_argv(commands[0]) == [
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
    assert _command_argv(commands[2]) == [
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
    assert _command_argv(commands[4])[-2:] == [
        "--output",
        "benchmarks/lite_scorer_v2_evidence_packet.toml",
    ]


def test_lite_scorer_v2_run_planner_cli_rejects_invalid_manifest(
    tmp_path: Path,
) -> None:
    """The production CLI should reject unsupported student candidates."""
    _write_training_inputs(tmp_path)
    manifest = _write_run_manifest(tmp_path, student_candidate="unsupported")

    result = subprocess.run(
        [
            sys.executable,
            str(PLANNER),
            str(tmp_path),
            "--manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "benchmarks/lite_scorer_v2_run_manifest.toml: unsupported "
        "student_candidate 'unsupported'"
    ) in result.stderr
