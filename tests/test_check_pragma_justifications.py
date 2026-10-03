# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — pragma-justification gate tests

"""The justified-only ``pragma: no cover`` gate must catch bare markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.check_pragma_justifications import find_bare_pragmas, main

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_module(root: Path, body: str) -> None:
    """Create the UTF-8 source module scanned by the public pragma checker.

    Parameters
    ----------
    root : Path
        Temporary root below which the src/pkg directory is created.
    body : str
        Complete source text containing the marker cases under examination.
    """
    package = root / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "mod.py").write_text(body, encoding="utf-8")


def test_bare_marker_is_reported_with_location(tmp_path: Path) -> None:
    """Locate an unjustified marker in a real source file at its original line."""
    _write_module(tmp_path, "def f():\n    return 1  # pragma: no cover\n")

    offenders = find_bare_pragmas(tmp_path)

    assert len(offenders) == 1
    path, line_number, line = offenders[0]
    assert path.name == "mod.py"
    assert line_number == 2
    assert "pragma: no cover" in line


def test_dash_justified_markers_pass(tmp_path: Path) -> None:
    """Accept documented em-dash, hyphen and double-hyphen justifications."""
    _write_module(
        tmp_path,
        "A = 1  # pragma: no cover — defensive: unreachable\n"
        "B = 2  # pragma: no cover - extras-gated\n"
        "C = 3  # pragma: no cover -- hardware-gated\n",
    )

    assert find_bare_pragmas(tmp_path) == []


def test_trailing_whitespace_is_still_bare(tmp_path: Path) -> None:
    """Reject a marker followed only by spaces."""
    _write_module(tmp_path, "A = 1  # pragma: no cover   \n")

    assert len(find_bare_pragmas(tmp_path)) == 1


def test_dash_without_text_is_still_bare(tmp_path: Path) -> None:
    """Reject a justification separator without any explanatory text."""
    _write_module(tmp_path, "A = 1  # pragma: no cover —\n")

    assert len(find_bare_pragmas(tmp_path)) == 1


def test_main_exit_codes_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exercise failure output and subsequent success through the public CLI."""
    _write_module(tmp_path, "A = 1  # pragma: no cover\n")

    assert main(["--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "mod.py:1" in out

    (tmp_path / "src" / "pkg" / "mod.py").write_text(
        "A = 1  # pragma: no cover — justified\n",
        encoding="utf-8",
    )
    assert main(["--root", str(tmp_path)]) == 0


def test_repository_source_tree_is_justified_only() -> None:
    """The live src/ tree must hold the justified-only invariant."""
    assert find_bare_pragmas(_REPO_ROOT) == []
