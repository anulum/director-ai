# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 plan validation tests

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_lite_scorer_v2_plan.py"
SPEC = importlib.util.spec_from_file_location("validate_lite_scorer_v2_plan", VALIDATOR)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

validate_lite_scorer_v2_plan = MODULE.validate_lite_scorer_v2_plan


def test_lite_scorer_v2_plan_validates_current_package() -> None:
    assert validate_lite_scorer_v2_plan(ROOT) == []


def test_lite_scorer_v2_plan_rejects_public_score_claim(tmp_path: Path) -> None:
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "docs-site" / "guide").mkdir(parents=True)
    (tmp_path / "docs-site").mkdir(exist_ok=True)
    (tmp_path / "benchmarks" / "lite_scorer_v2_plan.toml").write_text(
        """
schema_version = "1.0.0"
plan_id = "lite-scorer-v2"
public_score_claim = true
claim_boundary = "Design and readiness plan only; no public score claim."
student_candidates = ["minilm_l6", "mobilebert", "distilbert"]
teacher_artifact_required = true
heldout_eval_required = true
onnx_export_required = true
quantized_latency_required = true
minimum_real_eval_rows = 1000
status = "design_ready"
evidence_packet = "benchmarks/lite_scorer_v2_evidence_packet.toml"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "benchmarks" / "lite_scorer_v2_evidence_packet.toml").write_text(
        """
schema_version = "1.0.0"
packet_id = "lite-scorer-v2-evidence"
public_score_claim = false
student_artifact_status = "pending"
teacher_artifact_status = "pending"
heldout_eval_status = "pending"
onnx_export_status = "pending"
quantized_latency_status = "pending"
claim_boundary = "Evidence packet placeholder only; no public score claim."
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "docs-site" / "guide" / "scoring.md").write_text(
        "Distilled NLI is planned and requires validation.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs-site" / "installation.md").write_text(
        "Distilled NLI is planned and requires validation.\n",
        encoding="utf-8",
    )

    errors = validate_lite_scorer_v2_plan(tmp_path)

    assert (
        "benchmarks/lite_scorer_v2_plan.toml: public_score_claim must be false"
        in errors
    )


def test_lite_scorer_v2_plan_rejects_unverified_public_claims(tmp_path: Path) -> None:
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "docs-site" / "guide").mkdir(parents=True)
    (tmp_path / "benchmarks" / "lite_scorer_v2_plan.toml").write_text(
        """
schema_version = "1.0.0"
plan_id = "lite-scorer-v2"
public_score_claim = false
claim_boundary = "Design and readiness plan only; no public score claim."
student_candidates = ["minilm_l6", "mobilebert", "distilbert"]
teacher_artifact_required = true
heldout_eval_required = true
onnx_export_required = true
quantized_latency_required = true
minimum_real_eval_rows = 1000
status = "design_ready"
evidence_packet = "benchmarks/lite_scorer_v2_evidence_packet.toml"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "benchmarks" / "lite_scorer_v2_evidence_packet.toml").write_text(
        """
schema_version = "1.0.0"
packet_id = "lite-scorer-v2-evidence"
public_score_claim = false
student_artifact_status = "pending"
teacher_artifact_status = "pending"
heldout_eval_status = "pending"
onnx_export_status = "pending"
quantized_latency_status = "pending"
claim_boundary = "Evidence packet placeholder only; no public score claim."
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "docs-site" / "guide" / "scoring.md").write_text(
        "Distilled NLI reaches ~70% BA at 5ms CPU.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs-site" / "installation.md").write_text(
        "Distilled NLI reaches ~70% BA at 5ms CPU.\n",
        encoding="utf-8",
    )

    errors = validate_lite_scorer_v2_plan(tmp_path)

    assert (
        "docs-site/guide/scoring.md: remove unverified Lite Scorer v2 accuracy or latency claim"
        in errors
    )


def test_lite_scorer_v2_plan_rejects_claiming_evidence_packet(tmp_path: Path) -> None:
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "docs-site" / "guide").mkdir(parents=True)
    (tmp_path / "benchmarks" / "lite_scorer_v2_plan.toml").write_text(
        """
schema_version = "1.0.0"
plan_id = "lite-scorer-v2"
public_score_claim = false
claim_boundary = "Design and readiness plan only; no public score claim."
student_candidates = ["minilm_l6", "mobilebert", "distilbert"]
teacher_artifact_required = true
heldout_eval_required = true
onnx_export_required = true
quantized_latency_required = true
minimum_real_eval_rows = 1000
status = "design_ready"
evidence_packet = "benchmarks/lite_scorer_v2_evidence_packet.toml"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "benchmarks" / "lite_scorer_v2_evidence_packet.toml").write_text(
        """
schema_version = "1.0.0"
packet_id = "lite-scorer-v2-evidence"
public_score_claim = true
student_artifact_status = "pending"
teacher_artifact_status = "pending"
heldout_eval_status = "pending"
onnx_export_status = "pending"
quantized_latency_status = "pending"
claim_boundary = "Evidence packet placeholder only; no public score claim."
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "docs-site" / "guide" / "scoring.md").write_text(
        "Distilled NLI is planned and requires validation.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs-site" / "installation.md").write_text(
        "Distilled NLI is planned and requires validation.\n",
        encoding="utf-8",
    )

    errors = validate_lite_scorer_v2_plan(tmp_path)

    assert (
        "benchmarks/lite_scorer_v2_evidence_packet.toml: public_score_claim must be false"
        in errors
    )
