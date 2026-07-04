# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 launcher real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 durable launcher CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "tools" / "launch_lite_scorer_v2_training.py"


def _write_training_script(root: Path) -> None:
    """Write the local training shim executed through the production launcher."""
    training_script = root / "training" / "train_distillation.py"
    training_script.parent.mkdir(parents=True, exist_ok=True)
    training_script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import argparse",
                "import json",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--teacher', required=True)",
                "parser.add_argument('--student', required=True)",
                "parser.add_argument('--epochs', required=True)",
                "parser.add_argument('--batch-size', required=True)",
                "parser.add_argument('--max-length', required=True)",
                "parser.add_argument('--temperature', required=True)",
                "parser.add_argument('--alpha', required=True)",
                "parser.add_argument('--lr', required=True)",
                "parser.add_argument('--seed', required=True)",
                "parser.add_argument('--eval-limit', required=True)",
                "parser.add_argument('--num-workers', required=True)",
                "parser.add_argument('--device', required=True)",
                "parser.add_argument('--summ-target', required=True)",
                "parser.add_argument('--general-target', required=True)",
                "parser.add_argument('--output-dir', required=True)",
                "args = parser.parse_args()",
                "output_dir = Path(args.output_dir)",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "(output_dir / 'launcher_receipt.json').write_text(",
                "    json.dumps(vars(args), sort_keys=True) + '\\n',",
                "    encoding='utf-8',",
                ")",
                "print('training shim completed')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_uv_shim(root: Path) -> Path:
    """Write a local ``uv`` executable that preserves the planned argv contract."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "",
                "import json",
                "import subprocess",
                "import sys",
                "from pathlib import Path",
                "",
                "argv = sys.argv[1:]",
                "Path('uv_argv.json').write_text(",
                "    json.dumps(argv, sort_keys=True) + '\\n',",
                "    encoding='utf-8',",
                ")",
                "if argv[:2] != ['run', '--frozen']:",
                "    raise SystemExit(64)",
                "raise SystemExit(subprocess.run(argv[2:], check=False).returncode)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    return bin_dir


def _write_run_manifest(root: Path) -> Path:
    """Write a complete Lite Scorer v2 run manifest for a temporary repository."""
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
epochs = 1
batch_size = 2
max_length = 32
temperature = 3.0
alpha = 0.5
learning_rate = 0.00005
training_seed = 20260518
eval_limit = 4
num_workers = 0
device = "cpu"
summ_target = 2
general_target = 2
latency_sample_count = 100
quantise_onnx = true
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_json_object(path: Path) -> Mapping[str, object]:
    """Load a JSON object from ``path`` with a narrow runtime type check."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast(Mapping[str, object], payload)


def _wait_for_text(path: Path, needle: str) -> str:
    """Return file contents once ``needle`` appears or fail after a short wait."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            if needle in text:
                return text
        time.sleep(0.05)
    raise AssertionError(f"{needle!r} did not appear in {path}")


def test_lite_scorer_v2_launcher_cli_starts_durable_training_run(
    tmp_path: Path,
) -> None:
    """The production launcher CLI should start a durable run via the run plan."""
    _write_training_script(tmp_path)
    manifest = _write_run_manifest(tmp_path)
    bin_dir = _write_uv_shim(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--timestamp",
            "2026-07-04T120000",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
        env=env,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    run_dir = (
        tmp_path
        / ".coordination"
        / "runs"
        / "DIRECTOR-AI"
        / "lite_scorer_v2_train_2026-07-04T120000"
    )
    assert payload["run_dir"] == run_dir.as_posix()
    assert payload["log"] == (run_dir / "train.log").as_posix()
    assert isinstance(payload["pid"], int)
    assert isinstance(payload["session_id"], int | type(None))

    log_text = _wait_for_text(run_dir / "train.log", "EXIT 0")
    assert "START " in log_text
    assert "PWD " in log_text
    assert "training shim completed" in log_text
    assert _load_json_object(run_dir / "metadata.json")["public_score_claim"] is False
    assert (
        (run_dir / "command.txt")
        .read_text(encoding="utf-8")
        .startswith("uv run --frozen python training/train_distillation.py")
    )
    receipt = _load_json_object(
        tmp_path / "MODELS" / "lite-scorer-v2" / "student" / "launcher_receipt.json"
    )
    assert receipt["teacher"] == "training/output/deberta-v3-base-hallucination"
    assert receipt["student"] == "training/output/minilm-safetensors"
    assert receipt["output_dir"] == "MODELS/lite-scorer-v2/student"
    assert json.loads((tmp_path / "uv_argv.json").read_text(encoding="utf-8"))[:4] == [
        "run",
        "--frozen",
        "python",
        "training/train_distillation.py",
    ]
