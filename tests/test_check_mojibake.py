# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Mojibake Source Guard Tests
"""Multi-angle tests for the cp1250/cp1252 mojibake source guard (WCH-5)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.check_mojibake import MOJIBAKE_SIGNATURES, find_mojibake, main
from tools.preflight import GATES

# Legitimate non-ASCII the guard must NOT flag: em dash, en dash, box-drawing
# horizontal, rightwards arrow, copyright, S-caron. Built from escapes so this
# test module carries none of the artefact byte sequences it exercises.
_LEGIT_UNICODE = "— – ─ → © Š"
_MISSED_MOJIBAKE = (
    "\u00e2\u2020\u2019",  # rightwards arrow
    "\u00e2\u2030\u0088",  # approximately equal
    "\u00e2\u201d\u20ac",  # box rule, cp1250 path
    "\u00e2\u0022\u20ac",  # box rule, cp1252 path
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestMojibakeGuard:
    """Real-surface and synthetic coverage for the mojibake guard."""

    def test_real_source_tree_is_mojibake_free(self) -> None:
        """Real-surface regression guard: the live tree must stay clean."""
        assert find_mojibake(_REPO_ROOT) == []

    def test_legitimate_unicode_is_not_flagged(self, tmp_path: Path) -> None:
        """Legitimate UTF-8 punctuation and Slovak letters must be accepted."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "clean.py").write_text(
            f"# dashes and rules: {_LEGIT_UNICODE}\nX = 1\n", encoding="utf-8"
        )
        assert find_mojibake(tmp_path) == []

    @pytest.mark.parametrize("signature", MOJIBAKE_SIGNATURES)
    def test_every_signature_is_detected(self, tmp_path: Path, signature: str) -> None:
        """Every configured corruption signature must produce one finding."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text(f"# corrupted {signature} here\n", encoding="utf-8")
        hits = find_mojibake(tmp_path)
        assert len(hits) == 1
        path, lineno, matched = hits[0]
        assert path.name == "bad.py"
        assert lineno == 1
        assert matched == signature

    @pytest.mark.parametrize("signature", _MISSED_MOJIBAKE)
    def test_common_arrow_and_rule_mojibake_is_detected(
        self,
        tmp_path: Path,
        signature: str,
    ) -> None:
        """Previously missed arrow and box-rule mojibake must be rejected."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text(f"# corrupted {signature} here\n", encoding="utf-8")
        hits = find_mojibake(tmp_path)
        assert len(hits) == 1
        assert hits[0][2] == signature

    def test_one_finding_per_corrupted_line(self, tmp_path: Path) -> None:
        """Two distinct artefacts on one line should report one finding."""
        src = tmp_path / "src"
        src.mkdir()
        line = f"x = '{MOJIBAKE_SIGNATURES[0]}{MOJIBAKE_SIGNATURES[1]}'\n"
        (src / "bad.py").write_text(line, encoding="utf-8")
        assert len(find_mojibake(tmp_path)) == 1

    def test_scan_skips_absent_directories(self, tmp_path: Path) -> None:
        """Missing configured scan directories should be skipped cleanly."""
        (tmp_path / "tools").mkdir()
        assert find_mojibake(tmp_path) == []

    def test_main_returns_one_and_reports_hits(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The CLI should print findings and fail when corruption is present."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text(f"# {MOJIBAKE_SIGNATURES[10]}\n", encoding="utf-8")
        rc = main(["--root", str(tmp_path)])
        captured = capsys.readouterr()
        assert rc == 1
        assert "mojibake artefact" in captured.out
        assert "artefact(s) found" in captured.err

    def test_main_returns_zero_on_clean_root(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The CLI should stay silent and pass for a clean repository root."""
        (tmp_path / "src").mkdir()
        rc = main(["--root", str(tmp_path)])
        assert rc == 0
        assert capsys.readouterr().out == ""

    def test_preflight_includes_mojibake_guard(self) -> None:
        """Local preflight should run the same mojibake guard wired in CI."""
        gates = {name: command for name, command in GATES}

        assert gates["mojibake-guard"] == [
            sys.executable,
            "tools/check_mojibake.py",
            "--root",
            ".",
        ]
