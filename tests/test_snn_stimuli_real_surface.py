# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real CLI-surface tests for canonical SNN stimulus records."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _repo_root() -> Path:
    """Return the checked-out repository root."""
    return Path(__file__).resolve().parents[1]


def _run_tool(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the SNN stimulus tool through its real Python CLI entry point."""
    return subprocess.run(
        [sys.executable, "tools/validate_snn_stimuli.py", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def test_validate_rejects_legacy_text_source_shape(tmp_path: Path) -> None:
    """The validator must reject pre-broadcast text/source records."""
    stimulus_dir = tmp_path / "snn_stimuli"
    stimulus_dir.mkdir()
    (stimulus_dir / "codex_1781901159.json").write_text(
        json.dumps(
            {
                "text": "Closed a previous DIRECTOR-AI hardening lane.",
                "source": "codex",
                "project": "DIRECTOR-AI",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_tool(_repo_root(), "validate", str(stimulus_dir))

    assert result.returncode == 1
    assert "missing keys: actor, content, timestamp" in result.stderr
    assert "unexpected keys: source, text" in result.stderr


def test_migrate_apply_rewrites_legacy_signal_record(tmp_path: Path) -> None:
    """The migrator should convert agent/repo/signal records in place."""
    stimulus_dir = tmp_path / "snn_stimuli"
    stimulus_dir.mkdir()
    stimulus_path = stimulus_dir / "codex_1782553887.json"
    stimulus_path.write_text(
        json.dumps(
            {
                "agent": "codex",
                "repo": "DIRECTOR-AI",
                "timestamp": "2026-06-27T11:51:00+02:00",
                "task": "middleware real-surface companion",
                "signals": [
                    "Converted middleware tests to a real ASGI companion.",
                    "Verified API-key and rate-limit interaction through HTTPX ASGI.",
                ],
                "verification": ["focused pytest passed", "strict mypy passed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    migrate = _run_tool(_repo_root(), "migrate", str(stimulus_dir), "--apply")
    validate = _run_tool(_repo_root(), "validate", str(stimulus_dir))

    assert migrate.returncode == 0, migrate.stderr
    assert validate.returncode == 0, validate.stderr
    payload = json.loads(stimulus_path.read_text(encoding="utf-8"))
    assert payload == {
        "content": (
            "middleware real-surface companion: Converted middleware tests to "
            "a real ASGI companion.; Verified API-key and rate-limit interaction "
            "through HTTPX ASGI.; verification: focused pytest passed; "
            "strict mypy passed"
        ),
        "project": "DIRECTOR-AI",
        "actor": "codex",
        "timestamp": "2026-06-27T11:51:00+02:00",
        "kind": "session_evidence",
    }


def test_write_creates_canonical_record_and_validate_accepts_it(
    tmp_path: Path,
) -> None:
    """The writer should create the canonical broadcast schema directly."""
    stimulus_dir = tmp_path / "snn_stimuli"
    output_path = stimulus_dir / "codex_1783211111.json"
    content = "Closed W001 memory-write discipline with validator evidence."

    write = _run_tool(
        _repo_root(),
        "write",
        "--stimulus-dir",
        str(stimulus_dir),
        "--output",
        str(output_path),
        "--project",
        "DIRECTOR-AI",
        "--actor",
        "codex",
        "--timestamp",
        "2026-07-05T14:21:00+02:00",
        "--content",
        content,
        "--kind",
        "session_evidence",
        "--source-ref",
        ".coordination/sessions/DIRECTOR-AI/codex_2026-07-05_w001.md",
        "--entity",
        "tools/validate_snn_stimuli.py",
        "--entity",
        "tests/test_snn_stimuli_real_surface.py",
    )
    validate = _run_tool(_repo_root(), "validate", str(stimulus_dir))

    assert write.returncode == 0, write.stderr
    assert validate.returncode == 0, validate.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == {
        "content": content,
        "project": "DIRECTOR-AI",
        "actor": "codex",
        "timestamp": "2026-07-05T14:21:00+02:00",
        "entities": [
            "tools/validate_snn_stimuli.py",
            "tests/test_snn_stimuli_real_surface.py",
        ],
        "kind": "session_evidence",
        "source_ref": ".coordination/sessions/DIRECTOR-AI/codex_2026-07-05_w001.md",
    }


def test_write_defaults_to_parseable_timestamp(tmp_path: Path) -> None:
    """The writer should supply a canonical timestamp when one is omitted."""
    stimulus_dir = tmp_path / "snn_stimuli"
    output_path = stimulus_dir / "codex_default_timestamp.json"

    write = _run_tool(
        _repo_root(),
        "write",
        "--stimulus-dir",
        str(stimulus_dir),
        "--output",
        str(output_path),
        "--project",
        "DIRECTOR-AI",
        "--actor",
        "codex",
        "--content",
        "Default timestamp path for canonical memory discipline.",
    )
    validate = _run_tool(_repo_root(), "validate", str(stimulus_dir))

    assert write.returncode == 0, write.stderr
    assert validate.returncode == 0, validate.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(payload["timestamp"], str)
    assert datetime.fromisoformat(payload["timestamp"]).tzinfo is not None
