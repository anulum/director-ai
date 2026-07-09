# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Capability-coverage Matrix Tests

"""Real-surface tests for the capability-matrix matrix + ratchet (WCC-1).

Covers the real repository checkout (the committed matrix must be current and
pass its own ratchet) and a miniature repository fixture exercising every
ratchet failure mode: stale outputs, unwired symbols, new gaps outside the
baseline, stale baseline entries, and unknown baseline entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.capability_matrix import build_matrix, check, main, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def real_matrix():
    """Build the real-repository matrix once for this module."""
    return build_matrix(REPO_ROOT)


class TestRealRepository:
    def test_committed_matrix_is_current_and_ratchet_passes(self):
        assert check(REPO_ROOT) == []

    def test_matrix_covers_every_export_and_hook(self, real_matrix):
        counts = real_matrix["counts"]
        assert counts["public_exports"] >= 226
        assert counts["experimental_hooks"] >= 23
        assert counts["unwired"] == 0
        assert len(real_matrix["rows"]) == (
            counts["public_exports"] + counts["experimental_hooks"]
        )

    def test_core_symbols_are_fully_covered(self, real_matrix):
        rows = {row["name"]: row for row in real_matrix["rows"]}
        for name in ("CoherenceScorer", "CoherenceAgent", "GroundTruthStore"):
            row = rows[name]
            assert row["wired"] is True
            assert row["tested"] is True
            assert row["documented"] is True


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mini_repo(tmp_path: Path) -> Path:
    """Create a miniature repository exercising every matrix state."""
    repo = tmp_path / "repo"
    _write(
        repo / "src/director_ai/__init__.py",
        "_LAZY_IMPORTS: dict[str, tuple[str, str]] = {\n"
        '    "Alpha": (".core", "Alpha"),\n'
        '    "Beta": (".missing_mod", "Beta"),\n'
        "}\n"
        "__all__ = sorted(_LAZY_IMPORTS)\n",
    )
    _write(repo / "src/director_ai/core/__init__.py", "class Alpha: ...\n")
    _write(
        repo / "src/director_ai/experimental/__init__.py",
        'EXPERIMENTAL_HOOKS: dict[str, str] = {\n    "hookx": "director_ai.core",\n}\n',
    )
    _write(repo / "tests/test_alpha.py", "from director_ai import Alpha, hookx\n")
    _write(repo / "docs-site/guide.md", "# Guide\n\nUse `Alpha` for scoring.\n")
    _write(repo / "benchmarks/bench_alpha.py", "# benchmarks Alpha\n")
    return repo


def _passing_baseline(repo: Path) -> None:
    _write(
        repo / "tools/capability_matrix_baseline.toml",
        'untested = ["Beta"]\nundocumented = ["Beta", "hookx"]\n',
    )


class TestMiniRepositoryMatrix:
    def test_rows_reflect_wiring_tests_docs_and_benchmarks(self, tmp_path):
        repo = _mini_repo(tmp_path)
        matrix = build_matrix(repo)
        rows = {row["name"]: row for row in matrix["rows"]}

        assert rows["Alpha"] == {
            "name": "Alpha",
            "kind": "public_export",
            "module": ".core",
            "wired": True,
            "tested": True,
            "documented": True,
            "benchmarked": True,
        }
        assert rows["Beta"]["wired"] is False
        assert rows["Beta"]["tested"] is False
        assert rows["hookx"]["kind"] == "experimental_hook"
        assert rows["hookx"]["wired"] is True
        assert rows["hookx"]["tested"] is True
        assert rows["hookx"]["documented"] is False

    def test_generate_writes_deterministic_outputs(self, tmp_path):
        repo = _mini_repo(tmp_path)
        assert main(["--repo", str(repo)]) == 0
        json_path = repo / "docs/_generated/capability_matrix.json"
        md_path = repo / "docs/_generated/capability_matrix.md"

        matrix = json.loads(json_path.read_text(encoding="utf-8"))
        assert matrix["schema_version"] == "capability-matrix.v1"
        assert matrix["gaps"]["unwired"] == ["Beta"]
        assert md_path.read_text(encoding="utf-8") == render_markdown(matrix)

        before = json_path.read_text(encoding="utf-8")
        assert main(["--repo", str(repo)]) == 0
        assert json_path.read_text(encoding="utf-8") == before  # deterministic


class TestRatchet:
    def test_missing_outputs_fail_as_stale(self, tmp_path):
        repo = _mini_repo(tmp_path)
        errors = check(repo)
        assert any("stale generated matrix" in e for e in errors)

    def test_unwired_symbol_always_fails(self, tmp_path):
        repo = _mini_repo(tmp_path)
        main(["--repo", str(repo)])
        _passing_baseline(repo)
        errors = check(repo)
        assert any("unwired public symbols" in e and "Beta" in e for e in errors)

    def test_baselined_gaps_pass_and_main_check_exit_codes_agree(self, tmp_path):
        repo = _mini_repo(tmp_path)
        # Wire Beta so only baselined test/doc gaps remain.
        _write(repo / "src/director_ai/missing_mod.py", "class Beta: ...\n")
        main(["--repo", str(repo)])
        _passing_baseline(repo)

        assert check(repo) == []
        assert main(["--repo", str(repo), "--check"]) == 0

    def test_new_gap_outside_baseline_fails(self, tmp_path):
        repo = _mini_repo(tmp_path)
        _write(repo / "src/director_ai/missing_mod.py", "class Beta: ...\n")
        main(["--repo", str(repo)])
        _write(
            repo / "tools/capability_matrix_baseline.toml",
            'untested = ["Beta"]\nundocumented = ["Beta"]\n',  # hookx missing
        )
        errors = check(repo)
        assert any("new undocumented" in e and "hookx" in e for e in errors)

    def test_stale_baseline_entry_fails(self, tmp_path):
        repo = _mini_repo(tmp_path)
        _write(repo / "src/director_ai/missing_mod.py", "class Beta: ...\n")
        main(["--repo", str(repo)])
        _write(
            repo / "tools/capability_matrix_baseline.toml",
            'untested = ["Beta", "Alpha"]\nundocumented = ["Beta", "hookx"]\n',
        )
        errors = check(repo)
        assert any("stale baseline entries" in e and "Alpha" in e for e in errors)

    def test_unknown_baseline_entry_fails(self, tmp_path):
        repo = _mini_repo(tmp_path)
        _write(repo / "src/director_ai/missing_mod.py", "class Beta: ...\n")
        main(["--repo", str(repo)])
        _write(
            repo / "tools/capability_matrix_baseline.toml",
            'untested = ["Beta", "Ghost"]\nundocumented = ["Beta", "hookx"]\n',
        )
        errors = check(repo)
        assert any("unknown baseline entries" in e and "Ghost" in e for e in errors)

    def test_main_check_reports_failures_with_exit_one(self, tmp_path, capsys):
        repo = _mini_repo(tmp_path)  # no outputs generated yet
        assert main(["--repo", str(repo), "--check"]) == 1
        assert "capability-matrix:" in capsys.readouterr().err
