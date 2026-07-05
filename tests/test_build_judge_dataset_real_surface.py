# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - judge dataset builder real-surface tests
"""Real CLI-surface coverage for the judge dataset builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

datasets = pytest.importorskip("datasets")

ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    """Return an environment that forces local repository imports only."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["DIRECTOR_NLI_MODEL"] = "missing-model-that-must-not-load"
    return env


def _run_builder(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the production judge-dataset builder script in a subprocess."""
    return subprocess.run(
        [sys.executable, "training/build_judge_dataset.py", *args],
        cwd=ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _write_prescored_input(path: Path) -> None:
    """Write a real on-disk DatasetDict with precomputed NLI divergences."""
    dataset = datasets.Dataset.from_dict(
        {
            "premise": [
                "Paris is the capital of France.",
                "The deployment receipt was signed.",
                "The control plane rejected the unsafe action.",
                "The invoice remains unpaid.",
            ],
            "hypothesis": [
                "Paris is France's capital city.",
                "A signed receipt exists.",
                "The unsafe action was allowed.",
                "The invoice has been paid.",
            ],
            "label": [0, 0, 2, 1],
            "source": ["fixture"] * 4,
            "nli_divergence": [0.12, 0.41, 0.87, 0.64],
        }
    )
    datasets.DatasetDict({"train": dataset}).save_to_disk(str(path))


def test_build_judge_dataset_unit_guard_declares_real_surface_companion() -> None:
    """The helper-heavy judge-dataset guard should name this CLI companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_build_judge_dataset.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_build_judge_dataset_real_surface.py" in category


def test_prescored_judge_dataset_cli_writes_real_output_dataset(
    tmp_path: Path,
) -> None:
    """The CLI should transform a real pre-scored dataset without model loading."""
    input_dir = tmp_path / "input_dataset"
    output_dir = tmp_path / "judge_dataset"
    _write_prescored_input(input_dir)

    result = _run_builder(
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--precomputed-divergence-column",
        "nli_divergence",
        "--subsample",
        "0",
        "--borderline-keep",
        "0",
        "--confident-keep",
        "0",
        "--eval-ratio",
        "0.25",
        "--seed",
        "7",
    )

    assert result.returncode == 0, result.stderr
    assert "missing-model-that-must-not-load" not in result.stderr
    assert (output_dir / "dataset_dict.json").exists()
    assert (output_dir / "stats.json").exists()

    output = datasets.load_from_disk(str(output_dir))
    stats = cast(
        dict[str, int],
        json.loads((output_dir / "stats.json").read_text(encoding="utf-8")),
    )

    assert stats == {
        "total": 4,
        "train": 3,
        "eval": 1,
        "train_approve": 1,
        "train_reject": 2,
        "eval_approve": 1,
        "eval_reject": 0,
    }
    assert set(output.keys()) == {"train", "eval"}
    all_rows = list(output["train"]) + list(output["eval"])
    labels = sorted(cast(int, row["label"]) for row in all_rows)
    texts = [cast(str, row["text"]) for row in all_rows]
    assert labels == [0, 0, 1, 1]
    assert any("NLI divergence: 0.41" in text for text in texts)
    assert all(text.count("\n") == 2 for text in texts)


def test_prescored_judge_dataset_cli_fails_closed_on_missing_column(
    tmp_path: Path,
) -> None:
    """The CLI should reject pre-scored mode when the requested column is absent."""
    input_dir = tmp_path / "input_dataset"
    output_dir = tmp_path / "judge_dataset"
    _write_prescored_input(input_dir)

    result = _run_builder(
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--precomputed-divergence-column",
        "missing_divergence",
        "--subsample",
        "0",
    )

    assert result.returncode != 0
    assert "missing_divergence" in result.stderr
    assert "available columns" in result.stderr
    assert not (output_dir / "stats.json").exists()
