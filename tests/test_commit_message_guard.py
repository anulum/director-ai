# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI -- commit message attribution guard tests

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "commit-msg"
REQUIRED_TRAILER = "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".githooks").mkdir()
    shutil.copy2(HOOK, repo / ".githooks" / "commit-msg")
    return repo


def _message(repo: Path, content: str) -> Path:
    message_path = repo / "COMMIT_EDITMSG"
    message_path.write_text(content, encoding="utf-8")
    return message_path


def _hook(repo: Path, message_path: Path) -> subprocess.CompletedProcess[str]:
    return _run(["sh", ".githooks/commit-msg", str(message_path)], repo)


def test_commit_message_requires_arcane_sapience_trailer(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    message_path = _message(repo, "Add feature\n")

    result = _hook(repo, message_path)

    assert result.returncode == 1
    assert "exactly one required attribution trailer" in result.stderr


def test_commit_message_rejects_missing_message_file(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    result = _hook(repo, repo / "missing-message")

    assert result.returncode == 1
    assert "commit message file is missing" in result.stderr


def test_commit_message_rejects_duplicate_coauthored_trailers(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    message_path = _message(
        repo,
        "Add feature\n\n"
        f"{REQUIRED_TRAILER}\n"
        "Co-Authored-By: Another Contributor <other@example.invalid>\n",
    )

    result = _hook(repo, message_path)

    assert result.returncode == 1
    assert "exactly one Co-Authored-By trailer" in result.stderr


def test_commit_message_rejects_blocked_agent_identity(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    blocked_identity = "Co" + "dex"
    message_path = _message(
        repo,
        f"Add feature from {blocked_identity}\n\n{REQUIRED_TRAILER}\n",
    )

    result = _hook(repo, message_path)

    assert result.returncode == 1
    assert "blocked implementation-agent identity text" in result.stderr


def test_commit_message_accepts_single_required_trailer(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    message_path = _message(repo, f"Add feature\n\n{REQUIRED_TRAILER}\n")

    result = _hook(repo, message_path)

    assert result.returncode == 0
    assert result.stderr == ""
