# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real CLI-surface tests for the STUDIO manifest emitter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    """Return the checked-out repository root."""
    return Path(__file__).resolve().parents[2]


def _tool_env(repo_root: Path) -> dict[str, str]:
    """Return an environment that imports the repository source tree."""
    env = os.environ.copy()
    source_entries = [str(repo_root / "src"), str(repo_root)]
    existing = env.get("PYTHONPATH")
    if existing:
        source_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(source_entries)
    return env


def _run_tool(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the manifest emitter through its real Python CLI entry point."""
    return subprocess.run(
        [sys.executable, "tools/emit_studio_manifest.py", *args],
        cwd=repo_root,
        env=_tool_env(repo_root),
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_checks_committed_studio_manifest() -> None:
    """The checked-in STUDIO manifest should pass the production drift gate."""
    result = _run_tool(_repo_root(), "--check")

    assert result.returncode == 0, result.stderr or result.stdout


def test_cli_emits_and_checks_custom_artifact(tmp_path: Path) -> None:
    """The CLI should emit and verify a caller-selected manifest artifact."""
    artifact = tmp_path / "docs" / "_generated" / "studio_manifest.json"
    repo_root = _repo_root()

    emit = _run_tool(repo_root, "--artifact", str(artifact))
    check = _run_tool(repo_root, "--check", "--artifact", str(artifact))

    assert emit.returncode == 0, emit.stderr or emit.stdout
    assert check.returncode == 0, check.stderr or check.stdout
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["schema_a"]["studio"] == "director-ai"
    assert payload["schema_a"]["content_digest"].startswith("sha256:")
    assert payload["architecture_map"]["version"] == "architecture-map.v2"


def test_cli_reports_custom_artifact_drift(tmp_path: Path) -> None:
    """The CLI drift gate should reject a stale caller-selected artifact."""
    artifact = tmp_path / "studio_manifest.json"
    artifact.write_text(
        json.dumps({"studio": "director-ai", "verbs": []}) + "\n",
        encoding="utf-8",
    )

    result = _run_tool(_repo_root(), "--check", "--artifact", str(artifact))

    assert result.returncode == 1
    assert "stale" in result.stdout


def test_cli_reports_missing_custom_artifact(tmp_path: Path) -> None:
    """The CLI drift gate should report a missing caller-selected artifact."""
    artifact = tmp_path / "missing" / "studio_manifest.json"

    result = _run_tool(_repo_root(), "--check", "--artifact", str(artifact))

    assert result.returncode == 1
    assert "missing" in result.stdout
