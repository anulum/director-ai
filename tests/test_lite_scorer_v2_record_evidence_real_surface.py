# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evidence recorder real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 evidence recorder CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "tools" / "record_lite_scorer_v2_evidence.py"


def _write_plan(root: Path) -> None:
    """Write the Lite Scorer v2 plan required by the validator."""
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    (benchmarks / "lite_scorer_v2_plan.toml").write_text(
        """
schema_version = "1.0.0"
plan_id = "lite-scorer-v2-distillation"
public_score_claim = false
claim_boundary = "Design and readiness plan only; no public score claim."
student_candidates = ["minilm_l6", "mobilebert", "distilbert"]
teacher_artifact_required = true
heldout_eval_required = true
onnx_export_required = true
quantized_latency_required = true
minimum_real_eval_rows = 1000
status = "training_ready"
evidence_packet = "benchmarks/lite_scorer_v2_evidence_packet.toml"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _write_public_claim_surfaces(root: Path) -> None:
    """Write public documentation surfaces checked by the plan validator."""
    guide = root / "docs-site" / "guide"
    guide.mkdir(parents=True, exist_ok=True)
    (guide / "scoring.md").write_text(
        "Lite Scorer v2 requires recorded evidence.\n",
        encoding="utf-8",
    )
    (root / "docs-site" / "installation.md").write_text(
        "Lite Scorer v2 requires recorded evidence.\n",
        encoding="utf-8",
    )


def _write_artifacts(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    """Write local artefacts whose hashes must appear in the evidence packet."""
    model_dir = root / "MODELS" / "lite-scorer-v2"
    model_dir.mkdir(parents=True, exist_ok=True)
    student = model_dir / "student.safetensors"
    teacher = model_dir / "teacher.safetensors"
    onnx = model_dir / "model_quantized.onnx"
    model_card = model_dir / "model_card.md"
    claim_review = root / "benchmarks" / "lite_scorer_v2_claim_review.md"
    student.write_bytes(b"student artefact bytes\n")
    teacher.write_bytes(b"teacher artefact bytes\n")
    onnx.write_bytes(b"onnx artefact bytes\n")
    model_card.write_text("Lite Scorer v2 model card.\n", encoding="utf-8")
    claim_review.write_text(
        "Benchmark claim review: no public score claim.\n",
        encoding="utf-8",
    )
    return student, teacher, onnx, model_card, claim_review


def _write_eval_result(root: Path) -> Path:
    """Write evaluator JSON consumed by the evidence recorder CLI."""
    eval_result = root / "benchmarks" / "results" / "lite_scorer_v2_eval.json"
    eval_result.parent.mkdir(parents=True, exist_ok=True)
    eval_result.write_text(
        json.dumps(
            {
                "heldout_eval_dataset": "benchmarks/heldout/lite_scorer_v2.jsonl",
                "heldout_eval_rows": 1500,
                "heldout_eval_balanced_accuracy": 0.8125,
                "heldout_eval_threshold": 0.57,
                "latency_sample_count": 300,
                "latency_p50_ms": 2.9,
                "latency_p95_ms": 5.7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return eval_result


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest for ``path``."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lite_scorer_v2_evidence_recorder_cli_records_eval_result(
    tmp_path: Path,
) -> None:
    """The production CLI should record validated evidence from evaluator JSON."""
    _write_plan(tmp_path)
    _write_public_claim_surfaces(tmp_path)
    student, teacher, onnx, model_card, claim_review = _write_artifacts(tmp_path)
    eval_result = _write_eval_result(tmp_path)
    output = tmp_path / "benchmarks" / "lite_scorer_v2_evidence_packet.toml"

    result = subprocess.run(
        [
            sys.executable,
            str(RECORDER),
            str(tmp_path),
            "--eval-result",
            str(eval_result),
            "--student-candidate",
            "minilm_l6",
            "--student-artifact",
            str(student),
            "--teacher-artifact",
            str(teacher),
            "--onnx-artifact",
            str(onnx),
            "--model-card",
            str(model_card),
            "--benchmark-claim-review",
            str(claim_review),
            "--latency-backend",
            "onnxruntime",
            "--latency-device",
            "cpu:local-test",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert result.stdout == "lite_scorer_v2_evidence_recorded\n"
    assert result.stderr == ""
    packet = tomllib.loads(output.read_text(encoding="utf-8"))
    assert packet["public_score_claim"] is False
    assert packet["student_artifact_status"] == "recorded"
    assert packet["student_artifact_sha256"] == _sha256(student)
    assert packet["teacher_artifact_sha256"] == _sha256(teacher)
    assert packet["onnx_artifact_sha256"] == _sha256(onnx)
    assert packet["model_card_sha256"] == _sha256(model_card)
    assert packet["benchmark_claim_review_sha256"] == _sha256(claim_review)
    assert packet["heldout_eval_rows"] == 1500
    assert packet["heldout_eval_balanced_accuracy"] == 0.8125
    assert packet["latency_sample_count"] == 300
    assert packet["latency_p95_ms"] == 5.7


def test_lite_scorer_v2_evidence_recorder_cli_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    """The production CLI should fail closed when an artefact is absent."""
    _write_plan(tmp_path)
    _write_public_claim_surfaces(tmp_path)
    student, teacher, onnx, model_card, claim_review = _write_artifacts(tmp_path)
    missing_student = student.with_name("missing-student.safetensors")
    eval_result = _write_eval_result(tmp_path)
    output = tmp_path / "benchmarks" / "lite_scorer_v2_evidence_packet.toml"

    result = subprocess.run(
        [
            sys.executable,
            str(RECORDER),
            str(tmp_path),
            "--eval-result",
            str(eval_result),
            "--student-candidate",
            "minilm_l6",
            "--student-artifact",
            str(missing_student),
            "--teacher-artifact",
            str(teacher),
            "--onnx-artifact",
            str(onnx),
            "--model-card",
            str(model_card),
            "--benchmark-claim-review",
            str(claim_review),
            "--latency-backend",
            "onnxruntime",
            "--latency-device",
            "cpu:local-test",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert f"{missing_student}: artifact file does not exist" in result.stderr
    assert not output.exists()
