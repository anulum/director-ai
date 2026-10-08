# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Native repository history restore ownership tests

"""Exercise restore ownership and actual Git failures through public API and CLI."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tools.verify_repository_backup import (
    BackupVerificationError,
    verify_repository_backup,
)


def _run_git(repo: Path, *args: str) -> str:
    """Run native Git in the exclusively owned fixture repository."""
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _make_bundle(tmp_path: Path) -> tuple[Path, str]:
    """Create a real main-branch commit and its complete native history bundle."""
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
    """Keep an owned, fully verified checkout with native HEAD and fsck evidence."""
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
    """Remove the owned clone after an actual expected-commit mismatch."""
    bundle, _expected_head = _make_bundle(tmp_path)

    with pytest.raises(BackupVerificationError, match="expected HEAD") as caught:
        verify_repository_backup(
            bundle,
            expected_head="0" * 40,
            restore_parent=tmp_path / "restore",
        )
    result = caught.value.result
    assert result is not None
    assert result.ok is False
    assert result.restore_removed is True
    assert not (tmp_path / "restore" / "fixture").exists()


def test_verify_repository_backup_cli_outputs_json(tmp_path: Path) -> None:
    """Emit truthful JSON only after the owned CLI checkout is removed."""
    bundle, expected_head = _make_bundle(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
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


def _cli(
    bundle: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Invoke the actual verifier CLI under the current qualified interpreter."""
    return subprocess.run(
        [sys.executable, "-B", "tools/verify_repository_backup.py", str(bundle), *args],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _clone_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """Configure a real fixture-local Git checkout hook without mocking commands."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    hook = hooks / "post-checkout"
    hook.write_text("#!/usr/bin/env bash\nset -eu\n" + body + "\n")
    hook.chmod(0o700)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(hooks))


@pytest.mark.parametrize(
    "kind", ["directory", "file", "symlink", "dangling", "empty", "parent", "ancestor"]
)
def test_existing_destination_is_preserved(tmp_path: Path, kind: str) -> None:
    """Refuse every pre-existing path kind before clone without changing its bytes."""
    bundle, head = _make_bundle(tmp_path)
    special = {"empty": ".bundle", "parent": "..bundle", "ancestor": "...bundle"}
    if kind in special:
        unusual = bundle.with_name(special[kind])
        bundle.rename(unusual)
        bundle = unusual
    parent = tmp_path / "restore"
    parent.mkdir()
    destination = parent / {"empty": "", "parent": ".", "ancestor": ".."}.get(
        kind, "fixture"
    )
    target = tmp_path / "caller-data"
    target.mkdir()
    marker = target / "sentinel"
    marker.write_bytes(b"caller-owned data\n")
    marker.chmod(0o440)
    if kind in {"directory", "empty", "parent", "ancestor"}:
        if kind == "directory":
            destination.mkdir()
        marker = destination / "sentinel"
        marker.write_bytes(b"caller-owned data\n")
        marker.chmod(0o440)
    elif kind == "file":
        destination.write_bytes(b"caller-owned data\n")
        destination.chmod(0o440)
        marker = destination
    elif kind == "symlink":
        destination.symlink_to(target, target_is_directory=True)
    else:
        destination.symlink_to(tmp_path / "missing")
    original = destination.lstat()
    marker_original = marker.stat()

    completed = _cli(
        bundle, "--expected-head", head, "--restore-parent", str(parent), "--json"
    )
    report = json.loads(completed.stdout)

    assert completed.returncode == 1
    assert "restore path already exists" in report["error"]
    assert report["restore_removed"] is False
    assert [item["argv"][1] for item in report["commands"]] == ["bundle"]
    current = destination.lstat()
    assert (current.st_dev, current.st_ino, current.st_mode) == (
        original.st_dev,
        original.st_ino,
        original.st_mode,
    )
    assert marker.read_bytes() == b"caller-owned data\n"
    assert marker.stat().st_ino == marker_original.st_ino
    assert stat.S_IMODE(marker.stat().st_mode) == 0o440
    if kind in {"symlink", "dangling"}:
        assert destination.is_symlink()


@pytest.mark.parametrize("keep", [False, True])
def test_private_parent_retention_matches_request(tmp_path: Path, keep: bool) -> None:
    """Keep or remove a real default temporary checkout exactly as requested."""
    bundle, head = _make_bundle(tmp_path)
    result = verify_repository_backup(bundle, expected_head=head, keep_restore=keep)
    assert result.restore_path is not None
    path = result.restore_path
    assert path.parent.parent == Path(tempfile.gettempdir()).resolve()
    assert result.restore_removed is (not keep)
    assert path.exists() is keep
    assert path.parent.exists() is keep
    if keep:
        try:
            assert _run_git(path, "rev-parse", "HEAD") == head
            assert (
                path / "pyproject.toml"
            ).read_text() == "[project]\nname='fixture'\n"
        finally:
            shutil.rmtree(path.parent)


@pytest.mark.parametrize("keep", [False, True])
@pytest.mark.parametrize("private_parent", [False, True])
def test_failed_head_check_reports_final_cleanup(
    tmp_path: Path, keep: bool, private_parent: bool
) -> None:
    """Preserve native failure evidence and honour retention after a wrong HEAD."""
    bundle, head = _make_bundle(tmp_path)
    parent = tmp_path / "restore"
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(
            bundle,
            expected_head="0" * 40,
            restore_parent=None if private_parent else parent,
            keep_restore=keep,
        )
    result = caught.value.result
    assert result is not None
    assert result.restore_path is not None
    path = result.restore_path
    try:
        assert result.actual_head == head
        assert result.fsck_returncode == 0
        assert len(result.commands) == 5
        assert result.restore_removed is (not keep)
        assert path.exists() is keep
        assert path.parent.is_dir() is (keep or not private_parent)
    finally:
        if keep and private_parent:
            shutil.rmtree(path.parent)


@pytest.mark.parametrize("keep", [False, True])
def test_native_clone_failure_honours_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: bool
) -> None:
    """Capture a real failing Git checkout hook and retain only the requested clone."""
    bundle, head = _make_bundle(tmp_path)
    _clone_hook(tmp_path, monkeypatch, "exit 7")
    parent = tmp_path / "restore"
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(
            bundle, expected_head=head, restore_parent=parent, keep_restore=keep
        )
    result = caught.value.result
    assert result is not None
    assert result.error == "git clone failed"
    assert len(result.commands) == 2
    assert result.commands[-1].returncode != 0
    assert result.commands[-1].argv[1] == "clone"
    assert result.actual_head == ""
    assert result.fsck_returncode == -1
    assert result.restore_removed is (not keep)
    assert (parent / "fixture").exists() is keep


@pytest.mark.parametrize("replacement", ["directory", "symlink", "file", "absent"])
def test_replaced_restore_is_never_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    """Reject an actual checkout-path replacement and preserve its foreign payload."""
    bundle, head = _make_bundle(tmp_path)
    body = 'mv "$PWD" "${PWD}.original"\n'
    if replacement == "directory":
        body += 'mkdir "$PWD"\nprintf caller > "$PWD/sentinel"'
    elif replacement == "symlink":
        body += 'mkdir "${PWD}.foreign"\nprintf caller > "${PWD}.foreign/sentinel"\nln -s "${PWD}.foreign" "$PWD"'
    elif replacement == "file":
        body += 'printf caller > "$PWD"'
    _clone_hook(tmp_path, monkeypatch, body)
    parent = tmp_path / "restore"
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(bundle, expected_head=head, restore_parent=parent)
    result = caught.value.result
    assert result is not None
    assert result.ok is False
    assert "restore cleanup failed" in result.error
    assert result.restore_removed is False
    path = parent / "fixture"
    original = parent / "fixture.original"
    assert _run_git(original, "rev-parse", "HEAD") == head
    if replacement == "directory":
        assert (path / "sentinel").read_bytes() == b"caller"
    elif replacement == "symlink":
        assert path.is_symlink()
        assert (path / "sentinel").read_bytes() == b"caller"
    elif replacement == "file":
        assert path.read_bytes() == b"caller"
    else:
        assert not path.exists()


@pytest.mark.parametrize("keep", [False, True])
def test_native_unresolved_head_preserves_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep: bool
) -> None:
    """Reject native HEAD fallback text and retain the actual failure and clone policy."""
    bundle, head = _make_bundle(tmp_path)
    _clone_hook(tmp_path, monkeypatch, "git symbolic-ref HEAD refs/heads/missing")
    parent = tmp_path / "restore"
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(
            bundle, expected_head=head, restore_parent=parent, keep_restore=keep
        )
    result = caught.value.result
    assert result is not None
    assert result.error == "restored HEAD cannot be resolved"
    assert result.actual_head == "" and result.main_ref == ""
    assert result.fsck_returncode == -1
    assert len(result.commands) == 3
    assert result.commands[-1].argv[1:] == ("rev-parse", "HEAD")
    assert result.commands[-1].returncode != 0
    assert result.commands[-1].stdout == "HEAD"
    assert result.commands[-1].stderr
    assert result.restore_removed is (not keep)
    assert (parent / "fixture").exists() is keep


def test_native_missing_main_ref_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a detached native checkout with its main ref removed by Git itself."""
    bundle, head = _make_bundle(tmp_path)
    _clone_hook(
        tmp_path,
        monkeypatch,
        'git update-ref --no-deref HEAD "$2"\ngit update-ref -d refs/heads/main',
    )
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(
            bundle, expected_head=head, restore_parent=tmp_path / "restore"
        )
    result = caught.value.result
    assert result is not None
    assert result.actual_head == head
    assert result.error == "restored main ref cannot be resolved"
    assert result.commands[-1].argv[1] == "show-ref"
    assert result.commands[-1].returncode != 0
    assert result.restore_removed is True


def test_native_corrupt_object_fails_fsck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain actual fsck failure diagnostics after a real checkout object is corrupt."""
    bundle, head = _make_bundle(tmp_path)
    _clone_hook(
        tmp_path,
        monkeypatch,
        "mkdir -p .git/objects/aa\nprintf corrupt > .git/objects/aa/00000000000000000000000000000000000000",
    )
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(
            bundle, expected_head=head, restore_parent=tmp_path / "restore"
        )
    result = caught.value.result
    assert result is not None
    assert result.error == "git fsck failed"
    assert result.fsck_returncode != 0
    assert result.commands[-1].argv[1:4] == ("fsck", "--full", "--strict")
    assert result.commands[-1].stderr
    assert result.restore_removed is True


def test_native_cleanup_permission_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report a real read-only checkout cleanup failure instead of claiming success."""
    bundle, head = _make_bundle(tmp_path)
    _clone_hook(tmp_path, monkeypatch, 'chmod 500 "$PWD"')
    path = tmp_path / "restore" / "fixture"
    try:
        with pytest.raises(BackupVerificationError) as caught:
            verify_repository_backup(
                bundle, expected_head=head, restore_parent=path.parent
            )
        result = caught.value.result
        assert result is not None
        assert result.actual_head == head
        assert result.fsck_returncode == 0
        assert result.restore_removed is False
        assert "restore cleanup failed" in result.error
        assert path.is_dir()
    finally:
        path.chmod(0o700)


def test_private_parent_foreign_entry_survives_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remove only the owned clone and refuse to delete unrelated parent entries."""
    bundle, head = _make_bundle(tmp_path)
    _clone_hook(tmp_path, monkeypatch, "printf caller > ../sentinel")
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(bundle, expected_head=head)
    result = caught.value.result
    assert result is not None
    assert result.restore_path is not None
    parent = result.restore_path.parent
    try:
        assert result.actual_head == head
        assert result.fsck_returncode == 0
        assert result.restore_removed is True
        assert not result.restore_path.exists()
        assert "temporary parent cleanup failed" in result.error
        assert (parent / "sentinel").read_bytes() == b"caller"
    finally:
        shutil.rmtree(parent)


@pytest.mark.parametrize("keep", [False, True])
def test_invalid_bundle_leaves_no_default_parent(tmp_path: Path, keep: bool) -> None:
    """Refuse native invalid-bundle evidence and remove the unused owned parent."""
    invalid = tmp_path / "invalid.bundle"
    invalid.write_bytes(b"not a git bundle\n")
    parent = Path(tempfile.gettempdir())
    before = set(parent.glob("director-ai-restore-*"))
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(invalid, keep_restore=keep)
    result = caught.value.result
    assert result is not None
    assert result.error == "git bundle verify failed"
    assert result.bundle_verify_returncode != 0
    assert result.restore_path is None
    assert result.restore_removed is False
    assert set(parent.glob("director-ai-restore-*")) == before


def test_missing_native_git_returns_cli_refusal(tmp_path: Path) -> None:
    """Report real native-tool absence and preserve the caller's parent sentinel."""
    bundle, _head = _make_bundle(tmp_path)
    parent = tmp_path / "restore"
    parent.mkdir()
    marker = parent / "sentinel"
    marker.write_bytes(b"caller")
    env = os.environ.copy()
    empty = tmp_path / "empty-path"
    empty.mkdir()
    env["PATH"] = str(empty)
    completed = _cli(bundle, "--restore-parent", str(parent), "--json", env=env)
    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert report["ok"] is False
    assert report["commands"] == []
    assert report["restore_removed"] is False
    assert marker.read_bytes() == b"caller"


@pytest.mark.parametrize("as_json", [False, True])
def test_missing_bundle_cli_has_no_restore_effect(
    tmp_path: Path, as_json: bool
) -> None:
    """Refuse a missing input before creating a restore directory in either mode."""
    parent = tmp_path / "restore"
    args = ["--restore-parent", str(parent)]
    if as_json:
        args.append("--json")
    completed = _cli(tmp_path / "missing.bundle", *args)
    assert completed.returncode == 1
    assert "bundle does not exist" in completed.stdout
    assert not parent.exists()


def test_markdown_success_and_failure_reports(tmp_path: Path) -> None:
    """Render real native success and invalid-bundle cleanup outcomes as Markdown."""
    bundle, _head = _make_bundle(tmp_path)
    parent = tmp_path / "restore"
    success = _cli(bundle, "--restore-parent", str(parent))
    assert success.returncode == 0
    assert "ok: true" in success.stdout
    assert "restore_removed: true" in success.stdout
    invalid = tmp_path / "invalid.bundle"
    invalid.write_bytes(b"invalid")
    failure = _cli(invalid, "--restore-parent", str(parent))
    assert failure.returncode == 1
    assert "ok: false" in failure.stdout
    assert "restore_removed: false" in failure.stdout
    assert "git bundle verify failed" in failure.stdout


def test_safe_bundle_basename_is_used_by_public_api(tmp_path: Path) -> None:
    """Restore a native bundle with punctuation using its safe deterministic name."""
    bundle, head = _make_bundle(tmp_path)
    unusual = bundle.with_name("fixture ?.bundle")
    bundle.rename(unusual)
    result = verify_repository_backup(
        unusual, expected_head=head, restore_parent=tmp_path / "restore"
    )
    assert result.restore_path is not None
    assert result.restore_path.name == "fixture__"
    assert result.actual_head == head
    assert result.restore_removed is True


@pytest.mark.parametrize("via_cli", [False, True])
def test_existing_parent_file_is_preserved(tmp_path: Path, via_cli: bool) -> None:
    """Refuse a parent that is a real caller-owned file without modifying it."""
    bundle, head = _make_bundle(tmp_path)
    parent = tmp_path / "parent-file"
    parent.write_bytes(b"caller parent contents")
    parent.chmod(0o440)
    before = parent.stat()
    if via_cli:
        completed = _cli(bundle, "--restore-parent", str(parent), "--json")
        assert completed.returncode == 1
        payload = json.loads(completed.stdout)
        assert payload["ok"] is False
        assert payload["restore_removed"] is False
        assert len(payload["commands"]) == 1
    else:
        with pytest.raises(BackupVerificationError) as caught:
            verify_repository_backup(bundle, expected_head=head, restore_parent=parent)
        result = caught.value.result
        assert result is not None
        assert result.ok is False
        assert result.restore_removed is False
        assert len(result.commands) == 1
    assert parent.read_bytes() == b"caller parent contents"
    assert (parent.stat().st_ino, parent.stat().st_mode) == (
        before.st_ino,
        before.st_mode,
    )
