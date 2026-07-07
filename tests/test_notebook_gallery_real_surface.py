# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - notebook gallery real-surface tests
"""Real-surface coverage for the notebook gallery validator CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_notebook_gallery.py"


def _run_validator(root: Path) -> subprocess.CompletedProcess[str]:
    """Run the notebook gallery validator through its public CLI."""
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )


def _write_manifest(root: Path, notebook_path: str) -> None:
    """Write a minimal notebook gallery manifest under ``root``."""
    manifest = root / "notebooks" / "gallery.toml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        f"""
[[notebook]]
id = "listed"
path = "{notebook_path}"
title = "Listed"
track = "Foundations"
audience = "Evaluator"
duration_minutes = 5
use_case = "Baseline walkthrough."
extras = []
""".strip(),
        encoding="utf-8",
    )


def test_notebook_gallery_unit_guard_declares_real_surface_companion() -> None:
    """The notebook gallery guard should name its CLI companion surface."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_notebook_gallery.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_notebook_gallery_real_surface.py" in reason


def test_notebook_gallery_cli_accepts_repository_gallery() -> None:
    """The production CLI should accept the current repository gallery."""
    result = _run_validator(ROOT)

    assert result.returncode == 0
    assert result.stdout.strip() == "notebook_gallery_ok"
    assert result.stderr == ""


def test_notebook_gallery_cli_reports_missing_manifest(tmp_path: Path) -> None:
    """The production CLI should report a missing gallery manifest."""
    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "notebooks/gallery.toml: missing notebook gallery manifest" in result.stderr


def test_notebook_gallery_cli_rejects_path_escape(tmp_path: Path) -> None:
    """The production CLI should reject notebook paths escaping the repo."""
    _write_manifest(tmp_path, "../outside.ipynb")
    (tmp_path / "docs-site").mkdir()
    (tmp_path / "docs-site" / "notebook-gallery.md").write_text(
        "<!-- notebook-gallery:generated from notebooks/gallery.toml -->\n",
        encoding="utf-8",
    )

    result = _run_validator(tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "notebook[0]: path must stay inside repository" in result.stderr
