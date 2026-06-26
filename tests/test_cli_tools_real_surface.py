# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for interactive CLI tool help surfaces."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from director_ai.cli import main

TOOL_HELP_CASES = [
    (
        "kb-health",
        ("--min-docs", "--max-latency"),
        ("KB Health:", "Loading weights", "HF_TOKEN", "no CUDA device"),
    ),
    (
        "wizard",
        ("--cli", "--port", "--share", "--output"),
        ("Launching Gradio", "Config written", "Gradio not installed"),
    ),
    (
        "safety-dashboard",
        (
            "--text",
            "--events",
            "--feedback",
            "--halt-alert-threshold",
            "--false-positive-alert-threshold",
        ),
        ("Tenant halt rates", "Gradio not installed", "Launching"),
    ),
    (
        "compliance",
        ("report", "status", "drift", "--format"),
        ("EU AI Act Compliance Report", "No audit database found"),
    ),
]


@pytest.mark.parametrize(
    ("command", "expected_fragments", "forbidden_fragments"),
    TOOL_HELP_CASES,
)
def test_interactive_tool_help_exits_without_runtime_work(
    command: str,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
) -> None:
    """Tool command help must return through the installed CLI boundary."""
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
    TOOL_HELP_CASES,
)
def test_interactive_tool_dispatcher_help_has_no_side_effects(
    command: str,
    expected_fragments: tuple[str, ...],
    forbidden_fragments: tuple[str, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public dispatcher should print tool help before side effects."""
    main([command, "--help"])

    captured = capsys.readouterr()
    assert f"Usage: director-ai {command}" in captured.out
    for fragment in expected_fragments:
        assert fragment in captured.out
    for fragment in forbidden_fragments:
        assert fragment not in captured.out
    assert captured.err == ""


def test_safety_dashboard_text_mode_reads_events_feedback_and_thresholds(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Text dashboard should use real JSONL files through the CLI dispatcher."""
    events = tmp_path / "events.jsonl"
    events.write_text(
        '{"tenant_id":"tenant-a","policy_decision":"halt",'
        '"halt_reason":"contradiction","evidence_refs":["kb://physics"]}\n',
        encoding="utf-8",
    )
    feedback = tmp_path / "feedback.jsonl"
    feedback.write_text(
        '{"tenant_id":"tenant-a","false_positive":true,"source":"reviewer"}\n',
        encoding="utf-8",
    )

    main(
        [
            "safety-dashboard",
            "--text",
            "--events",
            str(events),
            "--feedback",
            str(feedback),
            "--halt-alert-threshold",
            "0.2",
            "--false-positive-alert-threshold",
            "0.1",
        ]
    )

    captured = capsys.readouterr()
    assert "Safety Operations" in captured.out
    assert "tenant-a" in captured.out
    assert "kb://physics" in captured.out
    assert "director-ai tune" in captured.out
    assert captured.err == ""


def test_kb_health_ignores_unknown_options_on_general_store(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unknown options should not block the real general-mode health path."""
    monkeypatch.setenv("DIRECTOR_MODE", "general")

    with pytest.raises(SystemExit) as exc_info:
        main(["kb-health", "--ignored", "--min-docs", "0"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "KB Health: HEALTHY" in captured.out
    assert captured.err == ""
