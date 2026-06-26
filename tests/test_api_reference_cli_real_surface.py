# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - API reference CLI real-surface tests
"""Real subprocess coverage for the API reference validator CLI."""

from __future__ import annotations

import runpy
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_api_reference.py"


def _load_validator() -> Callable[[Path], list[str]]:
    namespace = runpy.run_path(str(VALIDATOR))
    return cast(Callable[[Path], list[str]], namespace["validate_api_reference"])


def _run_validator_entrypoint(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str, str]:
    monkeypatch.setattr(sys, "argv", [str(VALIDATOR), *argv])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(VALIDATOR), run_name="__main__")
    code = exit_info.value.code
    assert isinstance(code, int)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_api_reference_validator_cli_accepts_explicit_root_option() -> None:
    """The validator CLI should verify the live API index via ``--root``."""
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--root", str(ROOT)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 0
    assert result.stdout == "api_reference_ok\n"
    assert result.stderr == ""


def test_api_reference_validator_cli_reports_stale_index_rows(
    tmp_path: Path,
) -> None:
    """The validator CLI should reject stale markdown links and symbols."""
    api_dir = tmp_path / "docs-site" / "api"
    api_dir.mkdir(parents=True)
    (api_dir / "guard.md").write_text("# Guard\n\n", encoding="utf-8")
    (api_dir / "index.md").write_text(
        "# API Reference\n\n"
        "| Symbol | Module | Purpose |\n"
        "|--------|--------|---------|\n"
        "| [`guard()`](missing.md) | `director_ai` | stale link |\n"
        "| [`definitely_missing()`](guard.md) | `director_ai` | stale symbol |\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=8,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert (
        "docs-site/api/index.md:5: missing markdown target missing.md" in result.stderr
    )
    assert (
        "docs-site/api/index.md:6: director_ai does not expose definitely_missing"
        in result.stderr
    )


def test_api_reference_validator_entrypoint_supports_root_forms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The script entrypoint should support positional and option root forms."""
    api_dir = tmp_path / "docs-site" / "api"
    api_dir.mkdir(parents=True)
    (api_dir / "index.md").write_text(
        "# API Reference\n\n"
        "| Symbol | Module | Purpose |\n"
        "|--------|--------|---------|\n",
        encoding="utf-8",
    )

    code, stdout, stderr = _run_validator_entrypoint(
        [str(tmp_path)],
        monkeypatch,
        capsys,
    )
    assert code == 0
    assert stdout == "api_reference_ok\n"
    assert stderr == ""

    missing_root = tmp_path / "missing-root"
    missing_root.mkdir()
    code, stdout, stderr = _run_validator_entrypoint(
        ["--root", str(missing_root)],
        monkeypatch,
        capsys,
    )
    assert code == 1
    assert stdout == ""
    assert stderr == "docs-site/api/index.md: missing API reference index\n"


def test_api_reference_validator_checks_markdown_edge_cases(
    tmp_path: Path,
) -> None:
    """The validator should reject stale anchors, escapes, and imports."""
    validate_api_reference = _load_validator()
    api_dir = tmp_path / "docs-site" / "api"
    api_dir.mkdir(parents=True)
    (api_dir / "target.md").write_text(
        "# Guard\n\n"
        "# Duplicate\n\n"
        "# Duplicate\n\n"
        "# Explicit {: #explicit-anchor}\n\n"
        "# Legacy {#legacy-anchor}\n\n"
        "# !!!\n",
        encoding="utf-8",
    )
    (api_dir / "index.md").write_text(
        "# API Reference\n\n"
        "| Symbol | Module | Purpose |\n"
        "|--------|--------|---------|\n"
        "| [`external()`](https://example.com/a#b) | `external.module` | external |\n"
        "| [`empty`](#) | `external.module` | empty link |\n"
        "| [`escape`](../../../../outside.md) | `external.module` | escape |\n"
        "| [`missing_anchor`](target.md#missing-anchor) | `external.module` | anchor |\n"
        "| [`duplicate`](target.md#duplicate_1) | `external.module` | duplicate |\n"
        "| [`explicit`](target.md#explicit-anchor) | `external.module` | explicit |\n"
        "| [`legacy`](target.md#legacy-anchor) | `external.module` | legacy |\n"
        "| singleton |\n"
        "| [`safe`](target.md#guard) | `external.module` | safe |\n"
        "| [`x`](target.md) | `director_ai.__missing_api_reference_module__` | import |\n",
        encoding="utf-8",
    )

    assert validate_api_reference(tmp_path) == [
        "docs-site/api/index.md:7: markdown target escapes repository ../../../../outside.md",
        "docs-site/api/index.md:8: missing anchor #missing-anchor in target.md",
        "docs-site/api/index.md:14: cannot import director_ai.__missing_api_reference_module__: No module named 'director_ai.__missing_api_reference_module__'",
    ]
