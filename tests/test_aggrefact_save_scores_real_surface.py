# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - AggreFact score-cache real-surface tests
"""Real public-surface coverage for AggreFact cached-score replay."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from benchmarks.aggrefact_eval import load_cached_scores as load_benchmark_scores
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]


def _subprocess_env() -> dict[str, str]:
    """Return a deterministic environment for CLI score-cache replay."""
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    env["PYTHONPATH"] = str(ROOT)
    env["DIRECTOR_NLI_MODEL"] = "missing-local-model-that-must-not-load"
    return env


def _run_aggrefact_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the production AggreFact benchmark module in a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "benchmarks.aggrefact_eval", *args],
        cwd=ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _run_training_loader(score_cache: Path) -> subprocess.CompletedProcess[str]:
    """Run the training classifier cached-score loader in a subprocess."""
    script = "\n".join(
        [
            "from __future__ import annotations",
            "import json",
            "import sys",
            "from tools.train_dataset_classifier import load_cached_scores",
            "scores = load_cached_scores(sys.argv[1])",
            "print(json.dumps({str(k): v for k, v in scores.items()}, sort_keys=True))",
        ]
    )
    return subprocess.run(
        [sys.executable, "-c", script, str(score_cache)],
        cwd=ROOT,
        env=_subprocess_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def _write_score_cache(path: Path) -> None:
    """Write a score cache matching ``score_and_save`` schema version 2."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "model": "local-score-cache-fixture",
                "backend": "transformers",
                "samples": 4,
                "scores": [0.91, 0.08, 0.73, 0.22],
                "predictions": [1, 0, 1, 0],
                "labels": [1, 0, 1, 0],
                "datasets_per_sample": [
                    "AggreFact-CNN",
                    "AggreFact-CNN",
                    "RAGTruth",
                    "RAGTruth",
                ],
                "latencies_per_sample": [0.001, 0.002, 0.003, 0.004],
                "unknown_predictions": 0,
                "total_time_seconds": 0.01,
                "mean_latency_ms": 2.5,
                "p50_latency_ms": 3.0,
                "p99_latency_ms": 4.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_aggrefact_save_scores_unit_guard_has_real_surface_companion() -> None:
    """The helper-heavy score-cache guard should name this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_aggrefact_save_scores.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_aggrefact_save_scores_real_surface.py" in category


def test_cached_score_replay_uses_real_cli_without_gated_dependencies(
    tmp_path: Path,
) -> None:
    """The CLI should replay saved scores without loading gated data or models."""
    score_cache = tmp_path / "aggrefact_scores.json"
    _write_score_cache(score_cache)

    result = _run_aggrefact_cli("--load-scores", str(score_cache))

    assert result.returncode == 0, result.stderr
    assert "Optimal threshold:" in result.stdout
    assert "LLM-AggreFact" in result.stdout
    assert "cached scores" in result.stdout
    assert "AggreFact-CNN" in result.stdout
    assert "RAGTruth" in result.stdout
    assert "missing-local-model-that-must-not-load" not in result.stderr


def test_cached_score_replay_rejects_misaligned_v2_arrays(tmp_path: Path) -> None:
    """The CLI should fail closed when score-cache arrays are misaligned."""
    score_cache = tmp_path / "bad_aggrefact_scores.json"
    score_cache.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scores": [0.91, 0.08],
                "labels": [1],
                "datasets_per_sample": ["AggreFact-CNN", "AggreFact-CNN"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_aggrefact_cli("--load-scores", str(score_cache))

    assert result.returncode != 0
    assert "inconsistent list lengths" in result.stderr
    assert "scores=2 labels=1 datasets=2" in result.stderr


def test_cached_score_schema_feeds_benchmark_and_training_loaders(
    tmp_path: Path,
) -> None:
    """Saved score caches should stay compatible with downstream tooling."""
    score_cache = tmp_path / "aggrefact_scores.json"
    _write_score_cache(score_cache)

    benchmark_scores = load_benchmark_scores(score_cache)
    training_result = _run_training_loader(score_cache)
    training_scores = cast(
        dict[str, dict[str, object]],
        json.loads(training_result.stdout),
    )

    assert training_result.returncode == 0, training_result.stderr
    assert benchmark_scores == {
        "AggreFact-CNN": [(1, 0.91), (0, 0.08)],
        "RAGTruth": [(1, 0.73), (0, 0.22)],
    }
    assert training_scores["0"] == {
        "dataset": "AggreFact-CNN",
        "label": 1,
        "score": 0.91,
    }
    assert training_scores["3"] == {
        "dataset": "RAGTruth",
        "label": 0,
        "score": 0.22,
    }
