# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual strict compiler and source-bound typing snapshots
"""Exercise actual strict typing snapshots, source custody and original advisories."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_ratchet_typing import capture_python_typing


def commit_typing_sources(root: Path) -> None:
    """Commit exact fixture source bytes in its isolated local Git repository.

    Parameters
    ----------
    root : Path
        Actual selected test repository.
    """
    subprocess.run(
        ["git", "add", "."], cwd=root, check=True, capture_output=True, timeout=10
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Quality Test",
            "-c",
            "user.email=quality@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-m",
            "Record actual typing source",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    )


@pytest.fixture
def typing_repository(tmp_path: Path) -> Path:
    """Create actual committed Python source and its original strict configuration.

    Parameters
    ----------
    tmp_path : Path
        Runner-owned Samsung temporary directory.

    Returns
    -------
    Path
        Real Git repository with genuine native annotation debt.
    """
    root = tmp_path / "typing-repository"
    root.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    )
    for name in ("src", "tools", "requirements", "packages"):
        (root / name).mkdir()
    (root / "pyproject.toml").write_text(
        '[tool.mypy]\nstrict=true\npython_version="3.11"\n'
        'explicit_package_bases=true\nmypy_path="src"\n'
    )
    (root / "requirements/ci-dev.txt").write_text("mypy==1.20.2\n")
    (root / "requirements/ci-types.txt").write_text("types-requests==2.33.0.20260906\n")
    (root / "src/legacy.py").write_text("def missing(value):\n    return value\n")
    commit_typing_sources(root)
    return root


def test_native_typing_binds_original_zero_based_column(
    typing_repository: Path,
) -> None:
    """Bind the native annotation error to actual source bytes and a one-based column.

    Parameters
    ----------
    typing_repository : Path
        Actual committed fixture supplied to MyPy.
    """
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert snapshot.complete and snapshot.profile == "python-types"
    assert snapshot.sources == {
        "src/legacy.py": hashlib.sha256(
            (typing_repository / "src/legacy.py").read_bytes()
        ).hexdigest()
    }
    assert len(snapshot.findings) == 1
    finding = snapshot.findings[0]
    assert (finding.path, finding.line, finding.column, finding.rule) == (
        "src/legacy.py",
        1,
        1,
        "no-untyped-def",
    )


def test_native_typing_retains_each_independent_setup(typing_repository: Path) -> None:
    """Analyse every real setup source without a duplicate logical-module collision.

    Parameters
    ----------
    typing_repository : Path
        Actual Git-visible source repository.
    """
    for name in ("setup.py", "packages/pro/setup.py", "packages/full/setup.py"):
        path = typing_repository / name
        path.parent.mkdir(exist_ok=True)
        path.write_text('quantity: int = "wrong"\n')
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert set(snapshot.sources) == {
        "src/legacy.py",
        "setup.py",
        "packages/pro/setup.py",
        "packages/full/setup.py",
    }
    assert {f.path for f in snapshot.findings} == set(snapshot.sources)


def test_native_typing_clean_source_and_stub(typing_repository: Path) -> None:
    """Complete actual native analysis of clean maintained Python and stub source.

    Parameters
    ----------
    typing_repository : Path
        Actual repository with replaced clean legacy source.
    """
    (typing_repository / "src/legacy.py").write_text(
        "def length(value: str) -> int:\n    return len(value)\n"
    )
    (typing_repository / "src/declared.pyi").write_text(
        "def size(value: str) -> int: ...\n"
    )
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert not snapshot.findings and len(snapshot.sources) == 2


def test_native_typing_removes_maintained_module_masking(
    typing_repository: Path,
) -> None:
    """Expose genuine errors despite original per-module masking and relaxed strict flags.

    Parameters
    ----------
    typing_repository : Path
        Actual selected native source repository.
    """
    with (typing_repository / "pyproject.toml").open("a") as stream:
        stream.write(
            '[[tool.mypy.overrides]]\nmodule="legacy"\nignore_errors=true\n'
            'disallow_untyped_defs=false\nfollow_imports="skip"\n'
            'disable_error_code=["no-untyped-def"]\n'
        )
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert len(snapshot.findings) == 1
    assert snapshot.findings[0].rule == "no-untyped-def"


@pytest.mark.parametrize(
    "configuration",
    (
        "",
        "[tool.mypy]\nstrict=false\n",
        '[tool.mypy]\nstrict=true\noverrides="invalid"\n',
        "[tool.mypy]\nstrict=true\n[[tool.mypy.overrides]]\nmodule=[]\n",
        '[tool.mypy]\nstrict=true\n[[tool.mypy.overrides]]\nmodule="bad,name"\n',
        "[tool.mypy]\nstrict=true\nplugins=7\n",
        '[tool.mypy]\nstrict=true\n[[tool.mypy.overrides]]\nmodule="legacy"\n'
        '[[tool.mypy.overrides]]\nmodule="legacy"\n',
    ),
)
def test_native_typing_refuses_invalid_original_configuration(
    typing_repository: Path,
    configuration: str,
) -> None:
    """Reject malformed or relaxed original configuration through the native surface.

    Parameters
    ----------
    typing_repository : Path
        Actual selected repository.
    configuration : str
        Explicit invalid original configuration bytes.
    """
    (typing_repository / "pyproject.toml").write_text(configuration)
    with pytest.raises(ValueError):
        capture_python_typing(typing_repository, sys.executable)


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("inf"), float("nan")))
def test_native_typing_refuses_invalid_deadline(
    typing_repository: Path, timeout: float
) -> None:
    """Reject unbounded or nonpositive native analysis budgets.

    Parameters
    ----------
    typing_repository : Path
        Actual selected repository.
    timeout : float
        Explicit invalid native process budget.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        capture_python_typing(typing_repository, sys.executable, timeout=timeout)


def test_native_typing_refuses_actual_syntax_failure(typing_repository: Path) -> None:
    """Refuse incomplete native analysis when maintained source cannot parse.

    Parameters
    ----------
    typing_repository : Path
        Actual selected repository.
    """
    (typing_repository / "src/broken.py").write_text("def broken(:\n")
    with pytest.raises(ValueError, match="completely analyse"):
        capture_python_typing(typing_repository, sys.executable)


def test_native_typing_actual_process_timeout(typing_repository: Path) -> None:
    """Terminate and reap actual native work when its finite budget is exhausted.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    """
    with pytest.raises((ValueError, subprocess.TimeoutExpired)):
        capture_python_typing(typing_repository, sys.executable, timeout=0.000001)


def test_native_typing_absent_interpreter(typing_repository: Path) -> None:
    """Refuse unavailable tooling without fabricating a clean source snapshot.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    """
    with pytest.raises(FileNotFoundError):
        capture_python_typing(
            typing_repository, str(typing_repository / "absent-python")
        )


def test_native_typing_retains_strict_mode_under_a_module_override(
    typing_repository: Path,
) -> None:
    """Keep native annotation checks enabled under an original relaxed override.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    """
    with (typing_repository / "pyproject.toml").open("a") as stream:
        stream.write('[[tool.mypy.overrides]]\nmodule="legacy"\nstrict=false\n')
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert snapshot.findings[0].rule == "no-untyped-def"


def test_native_typing_only_setup_remains_analyzed(typing_repository: Path) -> None:
    """Analyse the actual packaging source even when the main cohort is absent.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    """
    (typing_repository / "src/legacy.py").unlink()
    (typing_repository / "setup.py").write_text('quantity: int = "wrong"\n')
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert len(snapshot.sources) == 1 and snapshot.findings[0].path == "setup.py"


def test_native_typing_refuses_notebook_only_scope(typing_repository: Path) -> None:
    """Refuse a notebook-only repository rather than claiming to type-check its JSON.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    """
    (typing_repository / "src/legacy.py").unlink()
    (typing_repository / "notebooks").mkdir()
    (typing_repository / "notebooks/source.ipynb").write_text(
        json.dumps({"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []})
    )
    with pytest.raises(ValueError, match="typing source inventory is empty"):
        capture_python_typing(typing_repository, sys.executable)


def test_native_typing_distinguishes_actual_notes_from_errors(
    typing_repository: Path,
) -> None:
    """Keep a genuine compiler information note from becoming annotation debt.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    """
    (typing_repository / "src/legacy.py").write_text("reveal_type(1)\n")
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert snapshot.complete and not snapshot.findings


def test_native_typing_preserves_only_original_generated_module_boundary(
    typing_repository: Path,
) -> None:
    """Follow a real generated import while retaining its exact generator-owned exclusion.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    """
    generated = typing_repository / "src/director_ai/director_pb2.py"
    generated.parent.mkdir()
    generated.write_text('VALUE: int = "generated annotation defect"\n')
    (generated.parent / "__init__.py").write_text('"""Actual fixture package."""\n')
    (typing_repository / "src/legacy.py").write_text(
        "from director_ai.director_pb2 import VALUE\nRESULT: int = VALUE\n"
    )
    with (typing_repository / "pyproject.toml").open("a") as stream:
        stream.write(
            '[[tool.mypy.overrides]]\nmodule="director_ai.director_pb2"\n'
            "ignore_errors=true\n"
        )
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert snapshot.complete and not snapshot.findings
    assert set(snapshot.sources) == {"src/legacy.py", "src/director_ai/__init__.py"}


@pytest.mark.parametrize(
    "source", ("VALUE: int = 1\n", "VALUE: int = 1 # type: ignore[assignment]\n")
)
def test_native_typing_retains_actual_unused_override_notice_and_unknown_column(
    typing_repository: Path,
    source: str,
) -> None:
    """Retain actual unused-section information under both native success and error.

    Parameters
    ----------
    typing_repository : Path
        Actual selected source repository.
    source : str
        Genuine clean source or a genuine unused-ignore native diagnostic.
    """
    (typing_repository / "src/legacy.py").write_text(source)
    with (typing_repository / "pyproject.toml").open("a") as stream:
        stream.write(
            '[[tool.mypy.overrides]]\nmodule=["never_imported", "other_unused.*"]\n'
            "ignore_missing_imports=true\n"
        )
    snapshot = capture_python_typing(typing_repository, sys.executable)
    assert snapshot.complete
    assert snapshot.advisories == (
        "mypy-unused-override:never_imported",
        "mypy-unused-override:other_unused.*",
    )
    if "ignore" in source:
        assert len(snapshot.findings) == 1
        assert snapshot.findings[0].rule == "unused-ignore"
        assert snapshot.findings[0].column is None
        assert snapshot.findings[0].line == 1
    else:
        assert not snapshot.findings
