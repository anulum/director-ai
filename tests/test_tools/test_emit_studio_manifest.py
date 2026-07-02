# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the studio schema-A manifest emit/check tool.

Covers deterministic rendering (sorted keys, trailing newline), the emit path
(writes the artifact), and the ``--check`` drift gate: green against a fresh
artifact, red when missing or stale, and version-stable (a studio_version-only
difference does not trip the check).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Protocol, cast

import pytest


class StudioManifestTool(Protocol):
    """Protocol for the loaded STUDIO manifest emitter module."""

    _ARTIFACT: Path

    def render(self) -> str:
        """Return the rendered STUDIO manifest JSON."""
        ...

    def main(self, argv: list[str] | None = None) -> int:
        """Run the STUDIO manifest CLI entry point."""
        ...


def _repo_root() -> Path:
    """Return the checked-out repository root."""
    return Path(__file__).resolve().parents[2]


def _load_tool() -> StudioManifestTool:
    """Import the emitter module for unit-level branch checks."""
    module = importlib.import_module("tools.emit_studio_manifest")
    return cast("StudioManifestTool", module)


def test_render_is_sorted_with_trailing_newline() -> None:
    """Rendering should be stable and preserve the trailing newline."""
    tool = _load_tool()
    rendered = tool.render()
    assert rendered.endswith("\n")
    payload = json.loads(rendered)
    # Sorted-key render: re-dumping with sort_keys reproduces it byte-for-byte.
    assert (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        == rendered
    )
    assert set(payload) == {"architecture_map", "schema_a"}
    assert payload["schema_a"]["studio"] == "director-ai"
    assert payload["schema_a"]["content_digest"].startswith("sha256:")
    assert payload["architecture_map"]["version"] == "architecture-map.v2"


def test_committed_artifact_is_current() -> None:
    """The committed artifact must match the producer (the CI drift gate)."""
    tool = _load_tool()
    assert tool.main(["--check"]) == 0


def test_emit_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The unit-level emit path should write the selected artifact."""
    tool = _load_tool()
    target = tmp_path / "_generated" / "studio_manifest.json"
    monkeypatch.setattr(tool, "_ARTIFACT", target)
    assert tool.main([]) == 0
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_a"]["studio"] == "director-ai"
    assert payload["architecture_map"]["version"] == "architecture-map.v2"


def test_check_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The unit-level check path should report a missing artifact."""
    tool = _load_tool()
    monkeypatch.setattr(tool, "_ARTIFACT", tmp_path / "absent.json")
    assert tool.main(["--check"]) == 1
    assert "missing" in capsys.readouterr().out


def test_check_reports_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The unit-level check path should report a stale artifact."""
    tool = _load_tool()
    stale = tmp_path / "studio_manifest.json"
    stale.write_text(
        json.dumps({"studio": "director-ai", "verbs": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(tool, "_ARTIFACT", stale)
    assert tool.main(["--check"]) == 1
    assert "stale" in capsys.readouterr().out


def test_check_ignores_studio_version_only_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A studio_version-only difference is not drift (env-stable check)."""
    tool = _load_tool()
    payload = json.loads(tool.render())
    payload["schema_a"]["studio_version"] = "0+source-tree-stamp"
    artifact = tmp_path / "studio_manifest.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(tool, "_ARTIFACT", artifact)
    assert tool.main(["--check"]) == 0
