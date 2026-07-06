# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Real subprocess coverage for core CLI command help surfaces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import director_ai.cli as cli_module
from director_ai.cli import main
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

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


def test_top_level_help_lists_every_registered_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The top-level help output should not drift from the command registry."""
    main(["--help"])

    captured = capsys.readouterr()
    for command in cli_module._command_specs():
        assert f"  {command}" in captured.out
    assert captured.err == ""


def test_phase4_hardening_unit_guard_declares_real_surface_companions() -> None:
    """The phase4 hardening unit guard is backed by public workflow tests."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_phase4_hardening.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_actor_real_surface.py" in reason
    assert "tests/test_cli_core_real_surface.py" in reason
    assert "tests/test_config_real_surface.py" in reason
    assert "tests/test_server_real_surface.py" in reason


def test_batch_help_mentions_runtime_limits() -> None:
    """Batch help should expose the same limits enforced by the runtime."""
    env = {
        **os.environ,
        "DIRECTOR_FORCE_CPU": "1",
    }

    result = subprocess.run(
        [sys.executable, "-m", "director_ai.cli", "batch", "--help"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert "10K prompts" in result.stdout
    assert "100 MB file" in result.stdout
    assert "1 MB per line" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["nonpublic-cli-value-which-is-long-enough"], "Unknown command."),
        (
            ["nonpublic-cli-value-which-is-long-enough/invalid"],
            "Invalid command name.",
        ),
    ],
)
def test_rejected_commands_do_not_echo_secret_like_tokens(
    argv: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Rejected top-level commands must not disclose raw argv tokens."""
    with pytest.raises(SystemExit) as exc_info:
        main(argv)

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert message in captured.out
    assert "nonpublic-cli-value-which-is-long-enough" not in captured.out
    assert captured.err == ""


def test_evidence_success_writes_packet_without_exiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A passing evidence demo should write the packet and return zero."""
    import director_ai.core.evidence_packet as evidence_module
    from director_ai.guard import ProductionGuard

    seen: dict[str, object] = {}

    def fake_from_profile(profile: str) -> object:
        seen["profile"] = profile
        return object()

    def fake_packet(guard: object) -> dict[str, object]:
        seen["guard"] = guard
        return {
            "content": {
                "checks": {
                    "grounded_approved": True,
                    "hallucinated_blocked": True,
                },
            },
            "integrity": {"digest": "b" * 64},
        }

    monkeypatch.setattr(
        ProductionGuard, "from_profile", staticmethod(fake_from_profile)
    )
    monkeypatch.setattr(evidence_module, "build_evidence_packet", fake_packet)

    main(["evidence", "--profile", "fast", "--emit", str(tmp_path)])

    packet_path = tmp_path / "evidence_packet.json"
    assert seen["profile"] == "fast"
    assert packet_path.is_file()
    assert (
        json.loads(packet_path.read_text(encoding="utf-8"))["integrity"]["digest"]
        == "b" * 64
    )
    captured = capsys.readouterr()
    assert "Evidence packet written" in captured.out
    assert "Demo expectations not met" not in captured.out
    assert captured.err == ""


def test_verify_evidence_reports_missing_packet(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """verify-evidence should fail clearly when the packet path is absent."""
    missing = tmp_path / "missing-packet.json"

    with pytest.raises(SystemExit) as exc_info:
        main(["verify-evidence", str(missing)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert f"evidence packet not found at {missing}" in captured.out
    assert captured.err == ""


def test_verify_evidence_accepts_directory_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """verify-evidence should resolve directories to evidence_packet.json."""
    import director_ai.core.evidence_packet as evidence_module

    packet_path = tmp_path / "evidence_packet.json"
    packet_path.write_text('{"ok": true}', encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_verify(packet: object) -> tuple[bool, str]:
        seen["packet"] = packet
        return True, ""

    monkeypatch.setattr(evidence_module, "verify_evidence_packet", fake_verify)

    main(["verify-evidence", str(tmp_path)])

    captured = capsys.readouterr()
    assert seen["packet"] == {"ok": True}
    assert f"Evidence packet VERIFIED: {packet_path}" in captured.out
    assert captured.err == ""


def test_verify_evidence_invalid_packet_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """verify-evidence should preserve verifier failure reasons."""
    import director_ai.core.evidence_packet as evidence_module

    packet_path = tmp_path / "evidence_packet.json"
    packet_path.write_text('{"ok": false}', encoding="utf-8")

    def fake_verify(packet: object) -> tuple[bool, str]:
        return False, "bad digest"

    monkeypatch.setattr(evidence_module, "verify_evidence_packet", fake_verify)

    with pytest.raises(SystemExit) as exc_info:
        main(["verify-evidence", str(packet_path)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert f"Evidence packet INVALID (bad digest): {packet_path}" in captured.out
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


def test_verify_audit_accepts_real_log_without_secret(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """verify-audit should also verify logs without a secret flag."""
    audit_log = tmp_path / "audit.jsonl"
    audit_log.write_text("", encoding="utf-8")

    main(["verify-audit", str(audit_log)])

    captured = capsys.readouterr()
    assert "Audit chain VERIFIED" in captured.out
    assert captured.err == ""


def test_verify_audit_reports_missing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """verify-audit should fail before constructing an audit logger for misses."""
    missing = tmp_path / "missing-audit.jsonl"

    with pytest.raises(SystemExit) as exc_info:
        main(["verify-audit", str(missing)])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert f"audit log not found at {missing}" in captured.out
    assert captured.err == ""


def test_verify_audit_uses_secret_value_and_reports_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """verify-audit should pass explicit secrets into tamper verification."""
    import director_ai.core.safety.audit as audit_module

    audit_log = tmp_path / "audit.jsonl"
    audit_log.write_text("tampered\n", encoding="utf-8")
    seen: dict[str, object] = {}

    class FakeAuditLogger:
        """Audit logger double that records constructor and path inputs."""

        def __init__(self, hmac_secret: str | None = None) -> None:
            seen["secret"] = hmac_secret

        def verify_chain(self, path: Path) -> tuple[bool, int]:
            seen["path"] = path
            return False, 3

    monkeypatch.setattr(audit_module, "AuditLogger", FakeAuditLogger)

    with pytest.raises(SystemExit) as exc_info:
        main(["verify-audit", str(audit_log), "--secret", "custom-secret"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert seen == {"secret": "custom-secret", "path": audit_log}
    assert f"Audit chain TAMPERED at entry 3: {audit_log}" in captured.out
    assert captured.err == ""
