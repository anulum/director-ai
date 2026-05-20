# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Repository backup verification tests

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.verify_repository_backup import (
    BackupVerificationError,
    verify_repository_backup,
)


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _make_bundle(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "source"
    repo.mkdir()
    _run_git(repo, "init", "-b", "main")
    _run_git(repo, "config", "user.email", "test@example.invalid")
    _run_git(repo, "config", "user.name", "Backup Test")
    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\n", encoding="utf-8"
    )
    _run_git(repo, "add", "pyproject.toml")
    _run_git(repo, "commit", "-m", "initial fixture")
    head = _run_git(repo, "rev-parse", "HEAD")
    bundle = tmp_path / "fixture.bundle"
    _run_git(repo, "bundle", "create", str(bundle), "--all")
    return bundle, head


def test_verify_repository_backup_restores_bundle_and_checks_expected_head(
    tmp_path: Path,
) -> None:
    bundle, expected_head = _make_bundle(tmp_path)

    result = verify_repository_backup(
        bundle,
        expected_head=expected_head,
        restore_parent=tmp_path / "restore",
        keep_restore=True,
    )

    assert result.ok is True
    assert result.actual_head == expected_head
    assert result.expected_head == expected_head
    assert result.main_ref == expected_head
    assert result.restore_path is not None
    assert (result.restore_path / "pyproject.toml").is_file()
    assert result.fsck_returncode == 0
    assert result.bundle_verify_returncode == 0


def test_verify_repository_backup_rejects_unexpected_head(tmp_path: Path) -> None:
    bundle, _expected_head = _make_bundle(tmp_path)

    try:
        verify_repository_backup(
            bundle,
            expected_head="0" * 40,
            restore_parent=tmp_path / "restore",
        )
    except BackupVerificationError as exc:
        assert "expected HEAD" in str(exc)
        assert exc.result is not None
        assert exc.result.ok is False
    else:
        raise AssertionError("expected BackupVerificationError")


def test_verify_repository_backup_cli_outputs_json(tmp_path: Path) -> None:
    bundle, expected_head = _make_bundle(tmp_path)

    completed = subprocess.run(
        [
            "python",
            "tools/verify_repository_backup.py",
            str(bundle),
            "--expected-head",
            expected_head,
            "--restore-parent",
            str(tmp_path / "restore"),
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["ok"] is True
    assert payload["actual_head"] == expected_head
    assert payload["main_ref"] == expected_head
    assert payload["restore_removed"] is True
