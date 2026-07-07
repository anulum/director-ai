# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Sentinel-Judge analyser real-surface tests
"""Real subprocess coverage for the Sentinel-Judge analyser CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSER_SCRIPT = REPO_ROOT / "benchmarks" / "sentinel_judge_analyser.py"


def _write_judge_packet(
    path: Path,
    *,
    predictions: list[int],
    labels: list[int],
    datasets: list[str],
    scores: list[float] | None,
) -> None:
    """Write a production-schema judge packet for subprocess CLI tests."""
    packet: dict[str, Any] = {
        "model": path.stem,
        "predictions": predictions,
        "labels": labels,
        "datasets_per_sample": datasets,
    }
    if scores is not None:
        packet["scores"] = scores
    path.write_text(json.dumps(packet), encoding="utf-8")


def test_sentinel_judge_unit_guard_has_real_cli_companion() -> None:
    """The helper-heavy unit guard should name this real CLI companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_sentinel_judge_analyser.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_sentinel_judge_analyser_real_surface.py" in reason


def test_sentinel_judge_analyser_cli_writes_report(
    tmp_path: Path,
) -> None:
    """Run the production analyser CLI against local judge result files."""
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    datasets = ["alpha"] * 6 + ["beta"] * 6
    judge_a = tmp_path / "judge_a.json"
    judge_b = tmp_path / "judge_b.json"
    output = tmp_path / "reports" / "sentinel.json"
    _write_judge_packet(
        judge_a,
        predictions=[1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        labels=labels,
        datasets=datasets,
        scores=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.8, 0.6, 0.3, 0.4, 0.7, 0.2],
    )
    _write_judge_packet(
        judge_b,
        predictions=[1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1],
        labels=labels,
        datasets=datasets,
        scores=[0.8, 0.6, 0.3, 0.2, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.6],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYSER_SCRIPT),
            "--judges",
            str(judge_a),
            str(judge_b),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SENTINEL-JUDGE ENSEMBLE REPORT" in completed.stdout
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["judges"] == ["judge_a", "judge_b"]
    assert report["samples"] == 12
    assert set(report) == {
        "individual",
        "judges",
        "lr_fusion",
        "oracle_upper_bound",
        "routed",
        "samples",
        "voting",
    }
    assert report["lr_fusion"]["method"] == (
        "5-fold stratified CV, score+dataset_onehot features"
    )
    assert set(report["routed"]["routing_table"]) == {"alpha", "beta"}


def test_sentinel_judge_analyser_cli_rejects_score_length_mismatch(
    tmp_path: Path,
) -> None:
    """Score payload length errors should fail before ensemble fitting."""
    labels = [1, 0, 1, 0]
    datasets = ["alpha", "alpha", "beta", "beta"]
    judge = tmp_path / "judge_bad_scores.json"
    output = tmp_path / "sentinel.json"
    _write_judge_packet(
        judge,
        predictions=[1, 0, 1, 0],
        labels=labels,
        datasets=datasets,
        scores=[0.9],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYSER_SCRIPT),
            "--judges",
            str(judge),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 1
    assert "inconsistent score length" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output.exists()
