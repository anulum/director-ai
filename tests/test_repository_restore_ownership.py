# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Native public restore custody race regressions

"""Schedule real separate-process replacements across public restore operations."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import get_ident as ident

import pytest

from tools.verify_repository_backup import (
    BackupVerificationError,
    verify_repository_backup,
)


def _pin(path: Path) -> dict[str, object]:
    """Record real entry identity without following a caller's symbolic link."""
    value = path.lstat()
    return {"device": value.st_dev, "inode": value.st_ino, "mode": value.st_mode}


def _actor(request_fd: int, reply_fd: int, record: Path) -> None:
    """Perform only requested mutations in the explicitly owned test fixture."""
    events: list[dict[str, object]] = []
    with os.fdopen(request_fd) as requests:
        for line in requests:
            request: dict[str, str] = json.loads(line)
            mutations = [request]
            extra = request.get("extra")
            if extra is not None:
                mutations.append(
                    {"path": extra, "action": "replace", "marker": request["marker"]}
                )
            for mutation in mutations:
                path = Path(mutation["path"])
                action = mutation["action"]
                before = _pin(path) if path.exists() else None
                if action in {"replace", "empty"}:
                    path.rename(path.with_name(path.name + ".original"))
                path.mkdir(mode=0o700)
                if action != "empty":
                    marker = path / "DO_NOT_REMOVE"
                    marker.write_bytes(mutation["marker"].encode())
                    marker.chmod(0o440)
                events.append(
                    {"request": mutation, "original": before, "replacement": _pin(path)}
                )
            record.write_text(json.dumps(events, indent=2))
            os.write(reply_fd, b"1")
    os.close(reply_fd)


def _verifier(fixture: Path, phase: str, expected_head: str) -> int:
    """Call the actual public API while standard audit events rendezvous an actor."""
    request_r, request_w = os.pipe()
    reply_r, reply_w = os.pipe()
    record = fixture / "actor.json"
    launch = "import runpy; runpy.run_path('tests/test_repository_restore_ownership.py', run_name='__main__')"
    actor = subprocess.Popen(
        [
            sys.executable,
            "-B",
            "-c",
            launch,
            "actor",
            str(request_r),
            str(reply_w),
            str(record),
        ],
        pass_fds=(request_r, reply_w),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(request_r)
    os.close(reply_w)
    parent = fixture / "restore"
    fired = 0
    checkout_fd: int | None = None
    audit: list[dict[str, object]] = []

    def rendezvous(event: str, args: tuple[object, ...]) -> None:
        """Pause an actual filesystem operation until the independent actor finishes."""
        nonlocal fired, checkout_fd
        path: Path | None = None
        action = "replace"
        extra: Path | None = None
        if event == "subprocess.Popen":
            argv = args[1]
            if isinstance(argv, list) and len(argv) > 1 and argv[1] == "clone":
                checkout_fd = int(str(argv[-1]).rsplit("/", 1)[-1])
        if (
            event == "shutil.rmtree"
            and phase in {"checkout", "reoccupy"}
            and fired == 0
        ):
            candidate = Path(str(args[0]))
            assert candidate == parent / "fixture"
            path = candidate
        elif event == "os.rmdir" and phase == "parent" and fired == 0:
            candidate = Path(str(args[0]))
            if candidate.name.startswith("director-ai-restore-"):
                path = candidate
                action = "empty"
        elif (
            event == "os.rename"
            and phase in {"publish", "unused", "closeworkspace"}
            and fired == 0
        ):
            assert str(args[0]) == "checkout" and str(args[1]) == "fixture"
            path = parent / "fixture"
            action = "install"
            if phase == "unused":
                extra = Path(os.readlink(f"/proc/self/fd/{args[2]}")) / "checkout"
        elif event == "open" and phase == "entry" and str(args[0]) == ".git":
            assert fired == 0 and checkout_fd is not None
            path = Path(os.readlink(f"/proc/self/fd/{checkout_fd}")) / ".git"
        elif event == "os.rmdir" and phase == "closeworkspace" and fired == 1:
            assert str(args[0]) == "checkout"
            path = Path(os.readlink(f"/proc/self/fd/{args[1]}"))
        elif event == "os.rename" and phase == "reoccupy" and fired == 1:
            if str(args[0]) == "checkout" and str(args[1]) == "fixture":
                path = parent / "fixture"
                action = "install"
                audit.append(
                    {
                        "captured_path": str(
                            Path(os.readlink(f"/proc/self/fd/{args[2]}")) / "checkout"
                        )
                    }
                )
        elif (
            event == "subprocess.Popen"
            and phase
            in {"clone", "ancestor", "workspace", "authority", "head", "main", "fsck"}
            and fired == 0
        ):
            argv = args[1]
            command = {"head": "rev-parse", "main": "show-ref", "fsck": "fsck"}.get(
                phase, "clone"
            )
            if isinstance(argv, list) and len(argv) > 1 and argv[1] == command:
                if phase in {"workspace", "authority"}:
                    root = parent / f".director-ai-restore-custody-{os.geteuid()}"
                    path = root if phase == "authority" else next(root.iterdir())
                else:
                    path = parent if phase == "ancestor" else parent / "fixture"
        if path is None:
            return
        fired += 1
        before = _pin(path) if path.exists() else None
        audit.append({"event": event, "path": str(path), "before": before})
        message = {
            "path": str(path),
            "action": action,
            "marker": f"caller replacement {fired}\n",
        }
        if extra is not None:
            message["extra"] = str(extra)
        os.write(request_w, json.dumps(message).encode() + b"\n")
        assert select.select([reply_r], [], [], 20)[0]
        assert os.read(reply_r, 1) == b"1"

    executor = ThreadPoolExecutor(max_workers=1)
    executor.submit(str).result()
    call = executor.submit
    act = rendezvous
    tid = ident()
    live = True
    register = sys.addaudithook
    register(lambda e, a: call(act, e, a).result() if live and ident() == tid else None)
    status = 0
    try:
        try:
            result = verify_repository_backup(
                fixture / "fixture.bundle",
                expected_head=expected_head,
                restore_parent=None if phase == "parent" else parent,
            )
        except BackupVerificationError as exc:
            assert exc.result is not None
            result = exc.result
            status = 1
        print(json.dumps(result.to_json_dict()))
    finally:
        live = False
        executor.shutdown(wait=True)
        os.close(request_w)
        stdout, stderr = actor.communicate(timeout=30)
        os.close(reply_r)
        assert actor.returncode == 0, (stdout, stderr)
        (fixture / "audit.json").write_text(
            json.dumps(
                {"events": audit, "fired": fired, "actor_returncode": actor.returncode},
                indent=2,
            )
        )
    return status


def _git(path: Path, *args: str) -> str:
    """Run native Git and retain its actual successful output in the owned fixture."""
    run = subprocess.run(
        ["git", *args], cwd=path, capture_output=True, text=True, check=True
    )
    return run.stdout.strip()


def _run_case(
    tmp_path: Path, phase: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Build a real history bundle and execute the public race in a fresh process."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-b", "main")
    _git(source, "config", "user.email", "fixture@example.invalid")
    _git(source, "config", "user.name", "Owned Restore Race Fixture")
    (source / "README").write_text("Owned restore race fixture\n")
    _git(source, "add", "README")
    _git(source, "commit", "-m", "Owned restore race fixture")
    head = _git(source, "rev-parse", "HEAD")
    _git(source, "bundle", "create", str(tmp_path / "fixture.bundle"), "--all")
    temp = tmp_path / "temp"
    temp.mkdir()
    env = os.environ.copy()
    env.update(TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp))
    launch = "import runpy; runpy.run_path('tests/test_repository_restore_ownership.py', run_name='__main__')"
    run = subprocess.run(
        [sys.executable, "-B", "-c", launch, "verify", str(tmp_path), phase, head],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return run, head


@pytest.mark.parametrize(
    "phase",
    [
        "checkout",
        "parent",
        "publish",
        "clone",
        "ancestor",
        "reoccupy",
        "workspace",
        "authority",
        "entry",
        "unused",
        "closeworkspace",
        "head",
        "main",
        "fsck",
    ],
)
def test_public_restore_preserves_actual_concurrent_replacements(
    tmp_path: Path, phase: str
) -> None:
    """Preserve actual caller entries across allocation, clone, capture and recovery."""
    completed, head = _run_case(tmp_path, phase)
    assert completed.returncode == 1, completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is False
    events = json.loads((tmp_path / "actor.json").read_text())
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["fired"] == (2 if phase in {"reoccupy", "closeworkspace"} else 1)
    assert audit["actor_returncode"] == 0
    last = events[-1]
    replacement = Path(last["request"]["path"])
    if phase == "entry":
        replacement = Path(result["restore_path"]) / ".git"
    assert _pin(replacement) == last["replacement"]
    if phase == "parent":
        assert list(replacement.iterdir()) == []
        assert result["restore_removed"] is True
    else:
        marker = replacement / "DO_NOT_REMOVE"
        assert marker.read_bytes() == last["request"]["marker"].encode()
        assert marker.stat().st_mode & 0o777 == 0o440
        assert result["restore_removed"] is False
    if phase in {"checkout", "reoccupy"}:
        original = replacement.with_name(replacement.name + ".original")
        assert _git(original, "rev-parse", "HEAD") == head
        assert len(result["commands"]) == 5
        assert all(item["returncode"] == 0 for item in result["commands"])
    elif phase == "clone":
        original = replacement.with_name(replacement.name + ".original")
        assert _git(original, "rev-parse", "HEAD") == head
        assert result["commands"][-1]["argv"][1] == "clone"
        assert result["commands"][-1]["returncode"] == 0
    elif phase == "ancestor":
        original = replacement.with_name(replacement.name + ".original") / "fixture"
        assert _git(original, "rev-parse", "HEAD") == head
    if phase in {"head", "main", "fsck"}:
        original = replacement.with_name(replacement.name + ".original")
        assert _git(original, "rev-parse", "HEAD") == head
        assert len(result["commands"]) == {"head": 3, "main": 4, "fsck": 5}[phase]
        assert all(item["returncode"] == 0 for item in result["commands"])
        assert result["actual_head"] == head
        if phase in {"main", "fsck"}:
            assert result["main_ref"] == head
        assert result["fsck_returncode"] == (0 if phase == "fsck" else -1)
    if phase == "entry":
        checkout = Path(result["restore_path"])
        assert (
            _git(
                checkout,
                "--git-dir",
                str(checkout / ".git.original"),
                "rev-parse",
                "HEAD",
            )
            == head
        )
        assert "checkout entry changed while opening" in result["error"]
        assert len(result["commands"]) == 5
        assert all(item["returncode"] == 0 for item in result["commands"])
    if phase in {"unused", "closeworkspace"}:
        first = events[0]
        public = Path(first["request"]["path"])
        assert _pin(public) == first["replacement"]
        assert (public / "DO_NOT_REMOVE").read_bytes() == first["request"][
            "marker"
        ].encode()
        assert len(events) == 2 and len(result["commands"]) == 1
        assert (
            "unused allocation changed"
            if phase == "unused"
            else "private workspace identity changed"
        ) in result["error"]
    if phase in {"workspace", "authority"}:
        assert "private custody identity changed" in result["error"]
        assert _git(tmp_path / "restore" / "fixture", "rev-parse", "HEAD") == head
        assert len(result["commands"]) == 5
    if phase == "reoccupy":
        captured = Path(audit["events"][1]["captured_path"])
        first = events[0]
        assert _pin(captured) == first["replacement"]
        assert (captured / "DO_NOT_REMOVE").read_bytes() == first["request"][
            "marker"
        ].encode()
        assert "preserved at" in result["error"]


@pytest.mark.parametrize("kind", ["permissions", "symlink"])
def test_public_restore_refuses_unsafe_custody(tmp_path: Path, kind: str) -> None:
    """Preserve a caller's entry when account-private custody cannot be established."""
    bundle_source = tmp_path / "source"
    bundle_source.mkdir()
    _git(bundle_source, "init", "-b", "main")
    _git(bundle_source, "config", "user.email", "fixture@example.invalid")
    _git(bundle_source, "config", "user.name", "Owned Custody Fixture")
    (bundle_source / "README").write_text("Owned custody fixture")
    _git(bundle_source, "add", "README")
    _git(bundle_source, "commit", "-m", "Owned custody fixture")
    bundle = tmp_path / "fixture.bundle"
    _git(bundle_source, "bundle", "create", str(bundle), "--all")
    parent = tmp_path / "restore"
    parent.mkdir()
    root = parent / f".director-ai-restore-custody-{os.geteuid()}"
    target = root if kind == "permissions" else tmp_path / "caller-custody"
    target.mkdir(mode=0o755)
    target.chmod(0o755)
    assert target.stat().st_mode & 0o777 == 0o755
    marker = target / "DO_NOT_REMOVE"
    marker.write_bytes(b"caller custody contents")
    marker.chmod(0o440)
    if kind == "symlink":
        root.symlink_to(target, target_is_directory=True)
    before = _pin(root)
    with pytest.raises(BackupVerificationError) as caught:
        verify_repository_backup(bundle, restore_parent=parent)
    result = caught.value.result
    assert result is not None and result.ok is False
    assert result.restore_removed is False
    assert len(result.commands) == 1
    assert marker.read_bytes() == b"caller custody contents"
    assert _pin(root) == before
    assert marker.stat().st_mode & 0o777 == 0o440


if __name__ == "__main__":
    if sys.argv[1] == "actor":
        _actor(int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(_verifier(Path(sys.argv[2]), sys.argv[3], sys.argv[4]))
