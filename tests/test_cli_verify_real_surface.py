# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Verification CLI real-surface tests
"""Real subprocess coverage for verification and diagnostics CLI help surfaces."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from director_ai.cli import main

VERIFY_HELP_CASES = [
    (
        "doctor",
        ("runtime dependencies", "model revision"),
        ("dependency check", "NLI model ready"),
    ),
    (
        "license",
        ("status", "generate", "validate", "polar-env"),
        ("Tier:", "Unknown license subcommand"),
    ),
    (
        "verify-numeric",
        ("<text>", "numeric consistency"),
        ("Valid:", "Claims:"),
    ),
    (
        "verify-reasoning",
        ("<text>", "logical structure"),
        ("Chain valid:", "Steps:"),
    ),
    (
        "temporal-freshness",
        ("<text>", "temporal freshness"),
        ("Has temporal claims:", "Staleness risk:"),
    ),
    (
        "check-step",
        ("<goal>", "<action>"),
        ("Step:", "Budget:"),
    ),
    (
        "consensus",
        ("<model:response>", "colon-separated"),
        ("Models:", "Agreement:"),
    ),
    (
        "adversarial-test",
        ("[prompt]", "adversarial"),
        ("Patterns:", "COHERENCE FAILURE"),
    ),
    (
        "kpis",
        ("--input", "--format text|markdown|json"),
        ("Error: input bundle", "Guardrail KPIs"),
    ),
    (
        "forensics",
        ("--input", "--format text|markdown|json"),
        ("Error: input records", "Scorer-miss Forensics"),
    ),
    (
        "cost-report",
        ("--format text|json|html", "cost tracking"),
        ("Cost tracking is disabled", "No CostAnalyser"),
    ),
]


@pytest.mark.parametrize(
    ("command", "expected_fragments", "forbidden_fragments"),
    VERIFY_HELP_CASES,
)
def test_verify_command_help_exits_without_runtime_work(
    command: str,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> None:
    """Verification command help must return through the installed CLI boundary."""
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
    for fragment in expected_fragments:
        assert fragment in result.stdout
    for fragment in forbidden_fragments:
        assert fragment not in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("command", "expected_fragments", "forbidden_fragments"),
    VERIFY_HELP_CASES,
)
def test_verify_command_dispatcher_help_has_no_side_effects(
    command: str,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public dispatcher should print verification help before side effects."""
    main([command, "--help"])

    captured = capsys.readouterr()
    assert f"Usage: director-ai {command}" in captured.out
    for fragment in expected_fragments:
        assert fragment in captured.out
    for fragment in forbidden_fragments:
        assert fragment not in captured.out
    assert captured.err == ""
