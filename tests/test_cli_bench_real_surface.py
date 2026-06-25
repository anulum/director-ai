# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for benchmark and tuning CLI boundaries."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from director_ai.cli import main

BENCHMARK_FAMILY_HELP_CASES = [
    (
        "eval",
        ("--dataset", "--max-samples", "--output", "--quantize"),
        ("Running benchmarks", "benchmarks package not found"),
    ),
    (
        "bench",
        ("--dataset", "--seed", "--max-samples", "--output"),
        ("Running benchmarks", "benchmarks package not found"),
    ),
    (
        "tune",
        ("--dataset", "--profile", "--output"),
        ("missing dataset file", "file not found"),
    ),
    (
        "finetune",
        ("--eval", "--output", "--epochs", "--batch-size"),
        ("file not found", "Fine-tuning"),
    ),
]


@pytest.mark.parametrize(
    ("command", "expected_options", "forbidden_fragments"),
    BENCHMARK_FAMILY_HELP_CASES,
)
def test_benchmark_family_help_exits_without_running_work(
    command: str,
    expected_options: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> None:
    """Benchmark-family help must return through the public CLI entrypoint."""
    env = {
        **os.environ,
        "DIRECTOR_FORCE_CPU": "1",
    }

    result = subprocess.run(
        [sys.executable, "-m", "director_ai.cli", command, "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert f"Usage: director-ai {command}" in result.stdout
    for option in expected_options:
        assert option in result.stdout
    for fragment in forbidden_fragments:
        assert fragment not in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("command", "expected_options", "forbidden_fragments"),
    BENCHMARK_FAMILY_HELP_CASES,
)
def test_benchmark_family_dispatcher_help_has_no_side_effects(
    command: str,
    expected_options: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public dispatcher should print help before command side effects."""
    main([command, "--help"])

    captured = capsys.readouterr()
    assert f"Usage: director-ai {command}" in captured.out
    for option in expected_options:
        assert option in captured.out
    for fragment in forbidden_fragments:
        assert fragment not in captured.out
