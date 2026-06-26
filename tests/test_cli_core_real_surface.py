# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for core CLI command help surfaces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from director_ai.cli import main

CORE_HELP_CASES = [
    (
        "review",
        ("<prompt>", "<response>"),
        ("Approved:", "Coherence:"),
    ),
    (
        "process",
        ("<prompt>",),
        ("Output:", "Halted:", "Candidates:"),
    ),
    (
        "batch",
        ("<input.jsonl>", "--output"),
        ("file not found", "Total:", "Success:"),
    ),
    (
        "config",
        ("--profile",),
        ("mode:", "coherence_threshold:"),
    ),
    (
        "quickstart",
        ("--profile", "--no-compose", "--run"),
        ("Created director_guard", "already exists"),
    ),
]


@pytest.mark.parametrize(
    ("command", "expected_fragments", "forbidden_fragments"),
    CORE_HELP_CASES,
)
def test_core_command_help_exits_without_runtime_work(
    command: str,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> None:
    """Core command help must return through the installed CLI boundary."""
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
    CORE_HELP_CASES,
)
def test_core_command_dispatcher_help_has_no_side_effects(
    command: str,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public dispatcher should print core command help before side effects."""
    main([command, "--help"])

    captured = capsys.readouterr()
    assert f"Usage: director-ai {command}" in captured.out
    for fragment in expected_fragments:
        assert fragment in captured.out
    for fragment in forbidden_fragments:
        assert fragment not in captured.out
    assert captured.err == ""


def test_verify_audit_accepts_trailing_secret_flag_with_real_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A trailing --secret flag should not break real audit-log verification."""
    audit_log = tmp_path / "audit.jsonl"
    audit_log.write_text("", encoding="utf-8")

    main(["verify-audit", str(audit_log), "--secret"])

    captured = capsys.readouterr()
    assert "Audit chain VERIFIED" in captured.out
    assert captured.err == ""
