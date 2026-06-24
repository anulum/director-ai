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

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "emit_studio_manifest.py"
    spec = importlib.util.spec_from_file_location("emit_studio_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_is_sorted_with_trailing_newline() -> None:
    tool = _load_tool()
    rendered = tool.render()
    assert rendered.endswith("\n")
    payload = json.loads(rendered)
    # Sorted-key render: re-dumping with sort_keys reproduces it byte-for-byte.
    assert (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        == rendered
    )
    assert payload["studio"] == "director-ai"
    assert payload["content_digest"].startswith("sha256:")


def test_committed_artifact_is_current() -> None:
    """The committed artifact must match the producer (the CI drift gate)."""
    tool = _load_tool()
    assert tool.main(["--check"]) == 0


def test_emit_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _load_tool()
    target = tmp_path / "_generated" / "studio_manifest.json"
    monkeypatch.setattr(tool, "_ARTIFACT", target)
    assert tool.main([]) == 0
    assert target.exists()
    assert json.loads(target.read_text())["studio"] == "director-ai"


def test_check_reports_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    monkeypatch.setattr(tool, "_ARTIFACT", tmp_path / "absent.json")
    assert tool.main(["--check"]) == 1
    assert "missing" in capsys.readouterr().out


def test_check_reports_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = _load_tool()
    stale = tmp_path / "studio_manifest.json"
    stale.write_text(json.dumps({"studio": "director-ai", "verbs": []}))
    monkeypatch.setattr(tool, "_ARTIFACT", stale)
    assert tool.main(["--check"]) == 1
    assert "stale" in capsys.readouterr().out


def test_check_ignores_studio_version_only_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A studio_version-only difference is not drift (env-stable check)."""
    tool = _load_tool()
    payload = json.loads(tool.render())
    payload["studio_version"] = "0+source-tree-stamp"
    artifact = tmp_path / "studio_manifest.json"
    artifact.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(tool, "_ARTIFACT", artifact)
    assert tool.main(["--check"]) == 0
