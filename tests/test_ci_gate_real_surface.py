# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — CI gate real-surface tests
"""Real CLI coverage for the Director-AI CI quality gate."""

from __future__ import annotations

import json
import os

# Real-surface tests intentionally invoke the local Python CLI.
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import cast

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]


def _ci_gate_env() -> dict[str, str]:
    """Return a lightweight production configuration for subprocess CLI runs."""
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "DIRECTOR_ADAPTIVE_THRESHOLD_ENABLED": "false",
            "DIRECTOR_COHERENCE_THRESHOLD": "0.6",
            "DIRECTOR_FORCE_CPU": "1",
            "DIRECTOR_HYBRID_RETRIEVAL": "false",
            "DIRECTOR_LLM_PROVIDER": "mock",
            "DIRECTOR_MODEL_FALLBACK_ENABLED": "false",
            "DIRECTOR_RERANKER_ENABLED": "false",
            "DIRECTOR_SCORER_BACKEND": "lite",
            "DIRECTOR_USE_NLI": "false",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(ROOT / "src"),
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def _write_cases(path: Path, rows: list[dict[str, str]]) -> None:
    """Write labelled CI-gate JSONL rows to ``path``."""
    path.write_text(
        "".join(f"{json.dumps(row, sort_keys=True)}\n" for row in rows),
        encoding="utf-8",
    )


def _run_ci_gate(
    dataset: Path,
    output: Path,
    *extra_args: str,
) -> subprocess.CompletedProcess[str]:
    """Run the public ``director-ai ci-gate`` command through Python's CLI module."""
    # Fixed local module invocation; shell remains false.
    return subprocess.run(  # nosec B603
        [
            sys.executable,
            "-m",
            "director_ai.cli",
            "ci-gate",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            *extra_args,
        ],
        cwd=ROOT,
        env=_ci_gate_env(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30.0,
    )


def _read_report(path: Path) -> dict[str, object]:
    """Read a CI-gate JSON report from ``path``."""
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def test_ci_gate_unit_guard_has_real_cli_companion() -> None:
    """The helper-heavy CI-gate unit guard should be backed by CLI coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_ci_gate.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_ci_gate_real_surface.py" in category


def test_ci_gate_cli_scores_jsonl_and_writes_report(tmp_path: Path) -> None:
    """The public CLI should score JSONL cases and emit a CI artefact report."""
    dataset = tmp_path / "cases.jsonl"
    output = tmp_path / "gate.json"
    _write_cases(
        dataset,
        [
            {
                "id": "grounded-capital",
                "prompt": "What is the capital of France?",
                "response": "Paris is the capital of France.",
                "expected": "approve",
            },
            {
                "id": "grounded-arithmetic",
                "prompt": "What is two plus two?",
                "response": "Two plus two is four.",
                "expected": "approve",
            },
        ],
    )

    result = _run_ci_gate(
        dataset,
        output,
        "--min-accuracy",
        "1.0",
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Director-AI CI gate: PASS" in result.stdout
    assert "report written" in result.stdout

    report = _read_report(output)
    outcomes = cast(list[dict[str, object]], report["outcomes"])
    assert report["passed"] is True
    assert report["total"] == 2
    assert report["correct"] == 2
    assert report["accuracy"] == 1.0
    assert report["catch_rate"] is None
    assert report["false_halt_rate"] == 0.0
    assert [item["case_id"] for item in outcomes] == [
        "grounded-capital",
        "grounded-arithmetic",
    ]
    assert {item["predicted"] for item in outcomes} == {"approve"}


def test_ci_gate_cli_returns_one_when_threshold_breaches(tmp_path: Path) -> None:
    """The public CLI should exit ``1`` and persist failures for breached gates."""
    dataset = tmp_path / "cases.jsonl"
    output = tmp_path / "gate.json"
    _write_cases(
        dataset,
        [
            {
                "id": "labelled-hallucination",
                "prompt": "What is two plus two?",
                "response": "Two plus two is four.",
                "expected": "reject",
            }
        ],
    )

    result = _run_ci_gate(dataset, output, "--min-accuracy", "1.0")

    assert result.returncode == 1, result.stdout + result.stderr
    assert "Director-AI CI gate: FAIL" in result.stdout
    assert "accuracy 0.0% < required 100.0%" in result.stdout

    report = _read_report(output)
    outcomes = cast(list[dict[str, object]], report["outcomes"])
    failures = cast(list[str], report["failures"])
    assert report["passed"] is False
    assert report["total"] == 1
    assert report["correct"] == 0
    assert failures == ["accuracy 0.0% < required 100.0%"]
    assert outcomes[0]["expected"] == "reject"
    assert outcomes[0]["predicted"] == "approve"
