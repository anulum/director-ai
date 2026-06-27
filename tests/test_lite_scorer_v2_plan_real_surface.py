# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Lite Scorer v2 plan validator real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 plan validator CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_lite_scorer_v2_plan.py"


def _write_pending_packet(root: Path, *, public_score_claim: bool = False) -> None:
    """Write a minimal Lite Scorer v2 evidence packet under ``root``."""
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    (benchmarks / "lite_scorer_v2_evidence_packet.toml").write_text(
        f"""
schema_version = "1.0.0"
packet_id = "lite-scorer-v2-evidence"
public_score_claim = {str(public_score_claim).lower()}
student_artifact_status = "pending"
teacher_artifact_status = "pending"
heldout_eval_status = "pending"
onnx_export_status = "pending"
quantized_latency_status = "pending"
model_card_status = "pending"
benchmark_claim_review_status = "pending"
claim_boundary = "Evidence packet placeholder only; no public score claim."
""".strip(),
        encoding="utf-8",
    )


def _write_plan(root: Path, *, public_score_claim: bool = False) -> None:
    """Write a minimal Lite Scorer v2 readiness plan under ``root``."""
    benchmarks = root / "benchmarks"
    benchmarks.mkdir(parents=True, exist_ok=True)
    (benchmarks / "lite_scorer_v2_plan.toml").write_text(
        f"""
schema_version = "1.0.0"
plan_id = "lite-scorer-v2"
public_score_claim = {str(public_score_claim).lower()}
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


def test_lite_scorer_v2_validator_cli_accepts_checked_in_recorded_evidence() -> None:
    """The production CLI should accept the checked-in recorded evidence gate."""
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--require-recorded-evidence",
            str(ROOT),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert result.stdout == "lite_scorer_v2_plan_ok\n"
    assert result.stderr == ""


def test_lite_scorer_v2_validator_cli_rejects_public_score_claim(
    tmp_path: Path,
) -> None:
    """The production CLI should reject premature public score claims."""
    _write_plan(tmp_path, public_score_claim=True)
    _write_pending_packet(tmp_path)

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "benchmarks/lite_scorer_v2_plan.toml: public_score_claim must be false"
        in result.stderr
    )
