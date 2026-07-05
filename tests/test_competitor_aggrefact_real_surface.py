# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - competitor AggreFact real-surface tests
"""Real CLI-surface coverage for the competitor AggreFact benchmark harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write newline-delimited JSON rows using the benchmark input schema."""
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _fixture_rows() -> list[dict[str, object]]:
    """Return a realistic two-dataset AggreFact-style input sample."""
    return [
        {
            "doc": "The observatory confirmed the candidate comet on Tuesday.",
            "claim": "The comet candidate was confirmed on Tuesday.",
            "label": 1,
            "dataset": "AggreFact-CNN",
        },
        {
            "doc": "The grant review remains pending until September.",
            "claim": "The grant was approved in July.",
            "label": 0,
            "dataset": "AggreFact-CNN",
        },
        {
            "doc": "The audit trail contains a signed deployment receipt.",
            "claim": "A signed deployment receipt exists.",
            "label": 1,
            "dataset": "RAGTruth",
        },
        {
            "doc": "The safety gate rejected the ungrounded answer.",
            "claim": "The safety gate accepted the ungrounded answer.",
            "label": 0,
            "dataset": "RAGTruth",
        },
    ]


def _run_harness(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the production benchmark harness in an isolated subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, "benchmarks/competitor_aggrefact.py", *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def test_competitor_aggrefact_unit_guard_declares_real_surface_companion() -> None:
    """The mocked competitor guard should name this CLI companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_competitor_aggrefact.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_competitor_aggrefact_real_surface.py" in category


def test_competitor_aggrefact_cli_replays_precomputed_scores(
    tmp_path: Path,
) -> None:
    """The CLI should aggregate real local inputs without loading a model."""
    input_jsonl = tmp_path / "aggrefact_sample.jsonl"
    score_file = tmp_path / "scores.json"
    output_file = tmp_path / "competitor_result.json"
    _write_jsonl(input_jsonl, _fixture_rows())
    score_file.write_text(json.dumps([0.92, 0.08, 0.77, 0.13]), encoding="utf-8")

    result = _run_harness(
        "--model",
        "vectara/hallucination_evaluation_model",
        "--input-jsonl",
        str(input_jsonl),
        "--precomputed-scores",
        str(score_file),
        "--output",
        str(output_file),
        "--threshold",
        "0.5",
        "--log-every",
        "2",
    )

    assert result.returncode == 0, result.stderr
    assert "Loading:" not in result.stderr
    output = cast(
        dict[str, object],
        json.loads(output_file.read_text(encoding="utf-8")),
    )

    assert output["model"] == "vectara/hallucination_evaluation_model"
    assert output["backend"] == "precomputed-score-replay"
    assert output["samples"] == 4
    assert output["scores"] == [0.92, 0.08, 0.77, 0.13]
    assert output["predictions"] == [1, 0, 1, 0]
    assert output["labels"] == [1, 0, 1, 0]
    assert output["global_balanced_accuracy"] == 1.0

    per_dataset = cast(dict[str, dict[str, object]], output["per_dataset"])
    assert per_dataset == {
        "AggreFact-CNN": {"samples": 2, "balanced_accuracy": 1.0},
        "RAGTruth": {"samples": 2, "balanced_accuracy": 1.0},
    }


def test_competitor_aggrefact_cli_rejects_score_count_mismatch(
    tmp_path: Path,
) -> None:
    """The CLI should fail closed when replay scores do not align to rows."""
    input_jsonl = tmp_path / "aggrefact_sample.jsonl"
    score_file = tmp_path / "scores.json"
    output_file = tmp_path / "competitor_result.json"
    _write_jsonl(input_jsonl, _fixture_rows())
    score_file.write_text(json.dumps([0.92]), encoding="utf-8")

    result = _run_harness(
        "--model",
        "vectara/hallucination_evaluation_model",
        "--input-jsonl",
        str(input_jsonl),
        "--precomputed-scores",
        str(score_file),
        "--output",
        str(output_file),
    )

    assert result.returncode != 0
    assert "precomputed scores length" in result.stderr
    assert not output_file.exists()
