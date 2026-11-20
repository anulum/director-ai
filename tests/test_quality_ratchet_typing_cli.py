# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — actual strict compiler and Git typing ratchet acceptance
"""Exercise actual typing CLI adoption after its owning entry point is integrated."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def commit_typing_sources(root: Path) -> None:
    """Commit exact fixture source bytes in its isolated local Git repository.

    Parameters
    ----------
    root : Path
        Actual selected test repository.
    """
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True)
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
    )


def typing_cli(
    root: Path, command: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Invoke the public quality CLI with its actual native Python typing profile.

    Parameters
    ----------
    root : Path
        Actual selected source repository.
    command : str
        Capture, record or immutable comparison operation.
    arguments : str
        Additional explicit arguments.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Original native CLI status and output.
    """
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.check_quality_ratchet",
            command,
            "--profile",
            "python-types",
            "--root",
            str(root),
            *arguments,
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
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
    )
    for name in ("src", "tools", "requirements", "packages"):
        (root / name).mkdir()
    (root / "pyproject.toml").write_text(
        '[tool.mypy]\nstrict=true\npython_version="3.11"\n'
        'explicit_package_bases=true\nmypy_path="src"\n'
    )
    (root / "requirements/ci-dev.txt").write_text("mypy==1.20.2\n")
    (root / "requirements/ci-types.txt").write_text("types-requests==2.33.0.20260906\n")
    protobuf_lock = (
        Path(__file__).resolve().parents[1] / "requirements/quality-protobuf.txt"
    )
    (root / "requirements/quality-protobuf.txt").write_bytes(protobuf_lock.read_bytes())
    (root / "src/legacy.py").write_text("def missing(value):\n    return value\n")
    commit_typing_sources(root)
    return root


def test_typing_cli_original_adoption_and_new_debt_refusal(
    typing_repository: Path,
) -> None:
    """Accept genuine original debt and refuse a newly introduced annotation error.

    Parameters
    ----------
    typing_repository : Path
        Actual committed repository before baseline adoption.
    """
    baseline = typing_repository / "tools/quality_ratchet_baseline.json"
    recorded = typing_cli(typing_repository, "record", "--output", str(baseline))
    assert recorded.returncode == 0, recorded.stderr
    commit_typing_sources(typing_repository)
    checked = typing_cli(typing_repository, "check")
    assert checked.returncode == 0, checked.stderr
    (typing_repository / "src/new.py").write_text('quantity: int = "wrong"\n')
    refused = typing_cli(typing_repository, "check")
    assert refused.returncode == 1, refused.stderr
    assert "New or modified source has native findings: src/new.py" in refused.stdout


def test_typing_cli_dirty_origin_cannot_adopt(typing_repository: Path) -> None:
    """Refuse annotation allowances for bytes absent from the original Git commit.

    Parameters
    ----------
    typing_repository : Path
        Original committed fixture before adoption.
    """
    (typing_repository / "src/legacy.py").write_text('quantity: int = "wrong"\n')
    output = typing_repository / "tools/quality_ratchet_baseline.json"
    result = typing_cli(typing_repository, "record", "--output", str(output))
    assert result.returncode == 2 and not output.exists()
    assert "new or modified source bytes" in result.stderr


def test_typing_cli_refuses_protobuf_lock_identity_drift(
    typing_repository: Path,
) -> None:
    """Refuse an unchanged source set when the native Protobuf lock bytes drift.

    Parameters
    ----------
    typing_repository : Path
        Actual committed source and qualified Protobuf dependency lock.
    """
    baseline = typing_repository / "tools/quality_ratchet_baseline.json"
    recorded = typing_cli(typing_repository, "record", "--output", str(baseline))
    assert recorded.returncode == 0, recorded.stderr
    commit_typing_sources(typing_repository)
    lock = typing_repository / "requirements/quality-protobuf.txt"
    lock.write_bytes(lock.read_bytes() + b"\n# Unreviewed lock revision\n")
    checked = typing_cli(typing_repository, "check")
    assert checked.returncode == 1, checked.stdout + checked.stderr
    assert (
        "Native quality tool, configuration or scope identity changed" in checked.stdout
    )
