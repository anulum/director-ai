# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Distillation reproducibility real-surface tests
"""Real subprocess coverage for the Lite Scorer v2 distillation CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "training" / "train_distillation.py"


def _run_training_cli(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run the production distillation CLI with the provided arguments."""
    return subprocess.run(
        [sys.executable, str(TRAINER), *argv],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_train_distillation_reproducibility_unit_guard_has_real_cli_companion() -> None:
    """The reproducibility unit guard should be backed by CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_train_distillation_reproducibility.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_train_distillation_reproducibility_real_surface.py" in category


def test_train_distillation_cli_rejects_reproducibility_config_before_io(
    tmp_path: Path,
) -> None:
    """Invalid deterministic-run flags should fail before model or data writes."""
    output_dir = tmp_path / "student"

    result = _run_training_cli(
        "--teacher",
        "training/output/deberta-v3-base-hallucination",
        "--student",
        "microsoft/MiniLM-L6-H384-uncased",
        "--output-dir",
        str(output_dir),
        "--epochs",
        "0",
        "--alpha",
        "1.5",
        "--seed",
        "-1",
        "--eval-limit",
        "0",
        "--num-workers",
        "-1",
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        "epochs must be positive",
        "alpha must be in (0, 1]",
        "seed must be non-negative",
        "eval_limit must be positive",
        "num_workers must be non-negative",
    ]
    assert not output_dir.exists()
