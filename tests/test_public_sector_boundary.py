# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — public sector boundary verifier tests

"""Exercise public sector-file boundaries against actual temporary Git state."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.verify_public_sector_boundary import (
    evaluate_public_files,
    evaluate_staged_additions,
)


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run a Git command in the isolated repository and retain both streams.

    Parameters
    ----------
    command : list[str]
        Argument vector passed directly to the subprocess.
    cwd : Path
        Temporary repository in which the command runs.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Unchecked exit status and captured textual output.
    """
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    """Initialise a temporary Git repository with local test author details.

    Parameters
    ----------
    tmp_path : Path
        Pytest directory allocated for this test case.

    Returns
    -------
    Path
        Repository whose successful initialisation is asserted.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _run(["git", "init"], repo).returncode == 0
    assert (
        _run(["git", "config", "user.email", "test@example.invalid"], repo).returncode
        == 0
    )
    assert _run(["git", "config", "user.name", "Test User"], repo).returncode == 0
    return repo


def _write(repo: Path, relative_path: str, content: str) -> Path:
    """Write UTF-8 content beneath the isolated repository.

    Parameters
    ----------
    repo : Path
        Temporary repository root.
    relative_path : str
        Fixture path relative to that root.
    content : str
        Complete text written to the fixture file.

    Returns
    -------
    Path
        Written fixture path, after creating its parent directories.
    """
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_staged_sector_pack_path_is_reported_once_per_file(tmp_path: Path) -> None:
    """Report one forbidden pack-path finding from actual staged Git additions."""
    repo = _repo(tmp_path)
    private_pack = "src/director_ai/core/customer_model_factory/insurance_pack.py"
    _write(
        repo,
        private_pack,
        "def build_private_pack():\n    return {'ready': True}\n",
    )
    assert _run(["git", "add", private_pack], repo).returncode == 0

    findings = evaluate_staged_additions(repo)

    assert [(finding.path, finding.token) for finding in findings] == [
        (private_pack, "proprietary sector pack module path")
    ]


def test_public_sector_boundary_allows_generic_open_core_files(
    tmp_path: Path,
) -> None:
    """Allow tracked generic evidence code and the public metadata schema."""
    repo = _repo(tmp_path)
    _write(
        repo,
        "src/director_ai/core/customer_model_factory/evidence_pack.py",
        "class EvidencePack:\n    pass\n",
    )
    _write(
        repo,
        "schemas/customer-model-factory-sector-metadata.schema.json",
        "{}\n",
    )
    assert _run(["git", "add", "."], repo).returncode == 0

    assert evaluate_public_files(repo) == ()
