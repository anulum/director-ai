# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 evidence recording tests

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDER = ROOT / "tools" / "record_lite_scorer_v2_evidence.py"
VALIDATOR = ROOT / "tools" / "validate_lite_scorer_v2_plan.py"

RECORDER_SPEC = importlib.util.spec_from_file_location(
    "record_lite_scorer_v2_evidence", RECORDER
)
assert RECORDER_SPEC is not None
assert RECORDER_SPEC.loader is not None
RECORDER_MODULE = importlib.util.module_from_spec(RECORDER_SPEC)
sys.modules[RECORDER_SPEC.name] = RECORDER_MODULE
RECORDER_SPEC.loader.exec_module(RECORDER_MODULE)

VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_lite_scorer_v2_plan", VALIDATOR
)
assert VALIDATOR_SPEC is not None
assert VALIDATOR_SPEC.loader is not None
VALIDATOR_MODULE = importlib.util.module_from_spec(VALIDATOR_SPEC)
sys.modules[VALIDATOR_SPEC.name] = VALIDATOR_MODULE
VALIDATOR_SPEC.loader.exec_module(VALIDATOR_MODULE)

EvidenceRecord = RECORDER_MODULE.EvidenceRecord
record_lite_scorer_v2_evidence = RECORDER_MODULE.record_lite_scorer_v2_evidence
validate_lite_scorer_v2_plan = VALIDATOR_MODULE.validate_lite_scorer_v2_plan


def _write_plan(root: Path) -> None:
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "lite_scorer_v2_plan.toml").write_text(
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
""".strip(),
        encoding="utf-8",
    )


def _write_public_claim_surfaces(root: Path) -> None:
    (root / "docs-site" / "guide").mkdir(parents=True)
    (root / "docs-site" / "guide" / "scoring.md").write_text(
        "Lite Scorer v2 requires recorded evidence.\n",
        encoding="utf-8",
    )
    (root / "docs-site" / "installation.md").write_text(
        "Lite Scorer v2 requires recorded evidence.\n",
        encoding="utf-8",
    )


def _write_artifacts(root: Path) -> tuple[Path, Path, Path]:
    model_dir = root / "MODELS" / "lite-scorer-v2"
    model_dir.mkdir(parents=True)
    student = model_dir / "student.bin"
    teacher = model_dir / "teacher.bin"
    onnx = model_dir / "model.onnx"
    student.write_bytes(b"student artefact\n")
    teacher.write_bytes(b"teacher artefact\n")
    onnx.write_bytes(b"onnx artefact\n")
    return student, teacher, onnx


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_lite_scorer_v2_evidence_recorder_writes_valid_packet(
    tmp_path: Path,
) -> None:
    _write_plan(tmp_path)
    _write_public_claim_surfaces(tmp_path)
    student, teacher, onnx = _write_artifacts(tmp_path)

    record = EvidenceRecord(
        student_candidate="minilm_l6",
        student_artifact=student,
        teacher_artifact=teacher,
        heldout_eval_dataset=Path("benchmarks/heldout/lite_scorer_v2.jsonl"),
        heldout_eval_rows=1200,
        heldout_eval_balanced_accuracy=0.7425,
        heldout_eval_threshold=0.51,
        onnx_artifact=onnx,
        latency_backend="onnxruntime",
        latency_device="cpu:amd-ryzen-9-7950x",
        latency_sample_count=250,
        latency_p50_ms=3.4,
        latency_p95_ms=6.8,
    )

    errors = record_lite_scorer_v2_evidence(tmp_path, record)

    assert errors == []
    packet_path = tmp_path / "benchmarks" / "lite_scorer_v2_evidence_packet.toml"
    packet = tomllib.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["student_artifact_status"] == "recorded"
    assert packet["student_artifact_sha256"] == _sha256(student)
    assert packet["teacher_artifact_sha256"] == _sha256(teacher)
    assert packet["onnx_artifact_sha256"] == _sha256(onnx)
    assert packet["heldout_eval_rows"] == 1200
    assert packet["latency_p95_ms"] == 6.8
    assert validate_lite_scorer_v2_plan(tmp_path) == []


def test_lite_scorer_v2_evidence_recorder_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    _write_plan(tmp_path)
    _write_public_claim_surfaces(tmp_path)
    student, teacher, onnx = _write_artifacts(tmp_path)
    missing_student = student.with_name("missing-student.bin")
    record = EvidenceRecord(
        student_candidate="minilm_l6",
        student_artifact=missing_student,
        teacher_artifact=teacher,
        heldout_eval_dataset=Path("benchmarks/heldout/lite_scorer_v2.jsonl"),
        heldout_eval_rows=1200,
        heldout_eval_balanced_accuracy=0.7425,
        heldout_eval_threshold=0.51,
        onnx_artifact=onnx,
        latency_backend="onnxruntime",
        latency_device="cpu:amd-ryzen-9-7950x",
        latency_sample_count=250,
        latency_p50_ms=3.4,
        latency_p95_ms=6.8,
    )

    errors = record_lite_scorer_v2_evidence(tmp_path, record)

    assert errors == [f"{missing_student}: artifact file does not exist"]


def test_lite_scorer_v2_evidence_recorder_rejects_impossible_latency(
    tmp_path: Path,
) -> None:
    _write_plan(tmp_path)
    _write_public_claim_surfaces(tmp_path)
    student, teacher, onnx = _write_artifacts(tmp_path)
    record = EvidenceRecord(
        student_candidate="minilm_l6",
        student_artifact=student,
        teacher_artifact=teacher,
        heldout_eval_dataset=Path("benchmarks/heldout/lite_scorer_v2.jsonl"),
        heldout_eval_rows=1200,
        heldout_eval_balanced_accuracy=0.7425,
        heldout_eval_threshold=0.51,
        onnx_artifact=onnx,
        latency_backend="onnxruntime",
        latency_device="cpu:amd-ryzen-9-7950x",
        latency_sample_count=250,
        latency_p50_ms=6.8,
        latency_p95_ms=3.4,
    )

    errors = record_lite_scorer_v2_evidence(tmp_path, record)

    assert errors == [
        "benchmarks/lite_scorer_v2_evidence_packet.toml: latency_p95_ms must be greater than latency_p50_ms"
    ]
