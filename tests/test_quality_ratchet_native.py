# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — real source inventory and native notebook documentation tests
"""Verify native Ruff findings retain real source bytes, notebooks and scope boundaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.quality_ratchet_inventory import (
    GENERATED_PYTHON,
    python_sources,
    source_path,
)
from tools.quality_ratchet_native import (
    DOCUMENTATION_FLAGS,
    capture_python_documentation,
    documentation_identity,
    parse_documentation_result,
)


@pytest.fixture
def notebook_repository(tmp_path: Path) -> Path:
    """Build an actual Git-visible notebook and native configuration fixture.

    Parameters
    ----------
    tmp_path : Path
        Native runner's isolated temporary root.

    Returns
    -------
    Path
        Real Git repository with exact original notebook bytes.
    """
    root = tmp_path / "notebook-quality"
    root.mkdir()
    subprocess.run(
        ["git", "init"], cwd=root, check=True, capture_output=True, timeout=10
    )
    for directory in ("notebooks", "requirements"):
        (root / directory).mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname="notebook-test"\nversion="0.0.0"\n'
    )
    (root / "requirements/ci-lint.txt").write_text("ruff==0.16.10\n")
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"language_info": {"name": "python"}},
        "cells": [
            {
                "id": "original-cell",
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": ["def missing_documentation():\n", "    return 1\n"],
            }
        ],
    }
    (root / "notebooks/source.ipynb").write_text(json.dumps(notebook))
    return root


def test_native_notebook_findings_bind_actual_cell_and_original_bytes(
    notebook_repository: Path,
) -> None:
    """Keep the native notebook cell and byte digest rather than projecting to Python.

    Parameters
    ----------
    notebook_repository : Path
        Actual native notebook source repository.
    """
    snapshot = capture_python_documentation(notebook_repository, sys.executable)
    assert snapshot.complete
    assert snapshot.profile == "python-docs"
    assert snapshot.sources == {
        "notebooks/source.ipynb": hashlib.sha256(
            (notebook_repository / "notebooks/source.ipynb").read_bytes(),
        ).hexdigest(),
    }
    assert len(snapshot.findings) == 1
    finding = snapshot.findings[0]
    assert (finding.path, finding.cell, finding.rule, finding.line) == (
        "notebooks/source.ipynb",
        1,
        "D103",
        1,
    )


def test_inventory_refuses_nested_symbolic_source(notebook_repository: Path) -> None:
    """Reject symbolic source aliases even when their target is valid maintained code.

    Parameters
    ----------
    notebook_repository : Path
        Actual native source repository with an original notebook.
    """
    (notebook_repository / "src").mkdir()
    (notebook_repository / "src/alias.py").symlink_to("../notebooks/source.ipynb")
    with pytest.raises(ValueError, match="Symbolic"):
        python_sources(notebook_repository)


def test_inventory_refuses_nested_repository_as_its_root(
    notebook_repository: Path,
) -> None:
    """Require exact source repository ownership instead of scanning a selected subtree.

    Parameters
    ----------
    notebook_repository : Path
        Actual Git repository containing the nested selected path.
    """
    with pytest.raises(ValueError, match="actual Git root"):
        python_sources(notebook_repository / "notebooks")


def test_native_syntax_refusal_cannot_be_labelled_a_complete_snapshot(
    notebook_repository: Path,
) -> None:
    """Refuse real notebook syntax errors instead of accepting a smaller diagnostic set.

    Parameters
    ----------
    notebook_repository : Path
        Actual native notebook source repository.
    """
    (notebook_repository / "src").mkdir()
    (notebook_repository / "src/broken.py").write_text("def invalid(:\n")
    with pytest.raises(ValueError, match="completely parse"):
        capture_python_documentation(notebook_repository, sys.executable)


def test_native_missing_interpreter_refuses(notebook_repository: Path) -> None:
    """Reject unavailable native tooling instead of returning an empty snapshot.

    Parameters
    ----------
    notebook_repository : Path
        Actual selected native source repository.
    """
    with pytest.raises(FileNotFoundError):
        capture_python_documentation(
            notebook_repository, str(notebook_repository / "absent-python")
        )


def test_native_actual_process_deadline_refuses(notebook_repository: Path) -> None:
    """Enforce a real subprocess deadline with process termination and wait.

    Parameters
    ----------
    notebook_repository : Path
        Actual selected native source repository.
    """
    with pytest.raises(subprocess.TimeoutExpired):
        capture_python_documentation(
            notebook_repository, sys.executable, timeout=0.0000001
        )


def test_inventory_retains_pending_deletions_and_original_generator_boundaries(
    notebook_repository: Path,
) -> None:
    """Include maintained setup source while excluding only original generated paths.

    Parameters
    ----------
    notebook_repository : Path
        Actual repository with Git-visible maintained and generated source.
    """
    root = notebook_repository
    (root / "setup.py").write_text('"""Maintain the actual packaging source."""\n')
    generated = root / GENERATED_PYTHON[0]
    generated.parent.mkdir(parents=True)
    generated.write_bytes(b"original generated bytes\n")
    subprocess.run(
        ["git", "add", "."], cwd=root, check=True, capture_output=True, timeout=10
    )
    (root / "notebooks/source.ipynb").unlink()
    assert python_sources(root) == ("setup.py",)
    (root / "setup.py").unlink()
    with pytest.raises(ValueError, match="inventory is empty"):
        python_sources(root)


def test_regular_source_lookup_refuses_a_directory(notebook_repository: Path) -> None:
    """Reject a directory carrying a file-shaped path instead of supplying empty bytes.

    Parameters
    ----------
    notebook_repository : Path
        Actual selected repository for public source lookup.
    """
    directory = notebook_repository / "src/directory.py"
    directory.mkdir(parents=True)
    with pytest.raises(ValueError, match="not a regular file"):
        source_path(notebook_repository, "src/directory.py")


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("inf"), float("nan")))
def test_documentation_refuses_invalid_deadline(
    notebook_repository: Path,
    timeout: float,
) -> None:
    """Refuse an unbounded or invalid budget before starting native work.

    Parameters
    ----------
    notebook_repository : Path
        Actual selected source repository.
    timeout : float
        Invalid process budget.
    """
    with pytest.raises(ValueError, match="positive and finite"):
        capture_python_documentation(
            notebook_repository, sys.executable, timeout=timeout
        )


def test_documentation_identity_refuses_another_version(
    notebook_repository: Path,
) -> None:
    """Refuse a version outside the actual pinned profile at its consumed boundary.

    Parameters
    ----------
    notebook_repository : Path
        Actual source repository containing original configuration bytes.
    """
    with pytest.raises(ValueError, match="version differs"):
        documentation_identity(notebook_repository, "ruff 0.0.0")


@pytest.fixture
def original_documentation_report(
    notebook_repository: Path,
) -> subprocess.CompletedProcess[str]:
    """Capture the actual original native notebook diagnostic without interception.

    Parameters
    ----------
    notebook_repository : Path
        Actual source repository with one undocumented notebook function.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Complete actual native report used by the public parser controls.
    """
    report = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "ruff",
            *DOCUMENTATION_FLAGS,
            "--",
            "notebooks/source.ipynb",
        ],
        cwd=notebook_repository,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert report.returncode == 1 and not report.stderr
    assert len(json.loads(report.stdout)) == 1
    return report


@pytest.mark.parametrize(
    "field,value",
    (
        ("filename", 42),
        ("code", 42),
        ("code", "invalid-syntax"),
        ("location", {}),
        ("location", {"row": True, "column": 1}),
        ("message", ""),
        ("filename", "src/absent.py"),
    ),
)
def test_documentation_member_controls_refuse(
    notebook_repository: Path,
    original_documentation_report: subprocess.CompletedProcess[str],
    field: str,
    value: object,
) -> None:
    """Refuse malformed diagnostic records through the parser consumed by capture.

    Parameters
    ----------
    notebook_repository : Path
        Original native repository.
    original_documentation_report : subprocess.CompletedProcess[str]
        Actual native report before explicit negative-control corruption.
    field : str
        Native diagnostic member to corrupt.
    value : object
        Invalid member value; it never qualifies native execution success.
    """
    payload = json.loads(original_documentation_report.stdout)
    payload[0][field] = value
    control = subprocess.CompletedProcess(
        original_documentation_report.args, 1, json.dumps(payload), ""
    )
    with pytest.raises(ValueError):
        parse_documentation_result(
            notebook_repository, control, {"notebooks/source.ipynb": "a" * 64}, "b" * 64
        )


@pytest.mark.parametrize(
    "returncode,selector,stderr",
    (
        (2, "original", ""),
        (1, "original", "unexpected stderr"),
        (0, "original", ""),
        (1, "empty", ""),
        (0, "object", ""),
        (1, "missing", ""),
        (1, "duplicate", ""),
    ),
)
def test_documentation_report_controls_refuse(
    notebook_repository: Path,
    original_documentation_report: subprocess.CompletedProcess[str],
    returncode: int,
    selector: str,
    stderr: str,
) -> None:
    """Refuse incomplete processes, missing members and nonoriginal JSON structures.

    Parameters
    ----------
    notebook_repository : Path
        Original native source repository.
    original_documentation_report : subprocess.CompletedProcess[str]
        Actual complete diagnostic report.
    returncode : int
        Negative-control status.
    selector : str
        Explicit corruption selected from actual native bytes.
    stderr : str
        Negative-control stderr.
    """
    original = original_documentation_report.stdout
    payload = json.loads(original)
    del payload[0]["code"]
    controls = {
        "original": original,
        "empty": "[]",
        "object": "{}",
        "missing": json.dumps(payload),
        "duplicate": original.replace('"code":', '"code": null, "code":', 1),
    }
    control = subprocess.CompletedProcess(
        original_documentation_report.args, returncode, controls[selector], stderr
    )
    with pytest.raises(ValueError):
        parse_documentation_result(
            notebook_repository, control, {"notebooks/source.ipynb": "a" * 64}, "b" * 64
        )


def test_documentation_clean_source_qualifies(notebook_repository: Path) -> None:
    """Accept actual zero-status native analysis with no documentation findings.

    Parameters
    ----------
    notebook_repository : Path
        Actual notebook source repository.
    """
    (notebook_repository / "notebooks/source.ipynb").unlink()
    (notebook_repository / "setup.py").write_text('"""Actual packaging source."""\n')
    snapshot = capture_python_documentation(notebook_repository, sys.executable)
    assert snapshot.complete and not snapshot.findings
