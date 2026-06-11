# SPDX-License-Identifier: Apache-2.0
# Commercial licence available
# Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Repository backup restore verifier

"""Verify a git bundle backup by restoring it into a fresh checkout."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    """Captured command result for audit reporting."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class BackupVerificationResult:
    """Repository backup restore verification result."""

    ok: bool
    bundle_path: Path
    restore_path: Path | None
    actual_head: str
    expected_head: str | None
    main_ref: str
    bundle_verify_returncode: int
    fsck_returncode: int
    restore_removed: bool
    commands: tuple[CommandResult, ...]
    error: str = ""

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable result payload."""

        payload = asdict(self)
        payload["bundle_path"] = str(self.bundle_path)
        payload["restore_path"] = str(self.restore_path) if self.restore_path else None
        return payload

    def to_markdown(self) -> str:
        """Return a compact human-readable verification report."""

        rows = [
            "# Repository Backup Verification",
            "",
            f"ok: {str(self.ok).lower()}",
            f"bundle: `{self.bundle_path}`",
            f"restore_path: `{self.restore_path}`"
            if self.restore_path
            else "restore_path:",
            f"actual_head: `{self.actual_head}`",
            f"expected_head: `{self.expected_head}`"
            if self.expected_head
            else "expected_head:",
            f"main_ref: `{self.main_ref}`",
            f"bundle_verify_returncode: {self.bundle_verify_returncode}",
            f"fsck_returncode: {self.fsck_returncode}",
            f"restore_removed: {str(self.restore_removed).lower()}",
        ]
        if self.error:
            rows.append(f"error: {self.error}")
        rows.append("")
        return "\n".join(rows)


class BackupVerificationError(RuntimeError):
    """Raised when repository backup verification fails."""

    def __init__(self, message: str, result: BackupVerificationResult | None = None):
        super().__init__(message)
        self.result = result


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> CommandResult:
    completed = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    result = CommandResult(
        argv=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )
    if check and completed.returncode != 0:
        raise BackupVerificationError(
            f"command failed with exit code {completed.returncode}: {' '.join(argv)}"
        )
    return result


def _safe_restore_name(bundle_path: Path) -> str:
    stem = bundle_path.name.replace(".bundle", "")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)


def verify_repository_backup(
    bundle_path: Path,
    *,
    expected_head: str | None = None,
    restore_parent: Path | None = None,
    keep_restore: bool = False,
) -> BackupVerificationResult:
    """Restore and verify a git bundle backup.

    The verifier performs `git bundle verify`, clones the bundle into a fresh
    directory, validates `HEAD`, checks `refs/heads/main`, and runs
    `git fsck --full --strict`. By default the temporary restore directory is
    removed after verification.
    """

    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.is_file():
        raise BackupVerificationError(f"bundle does not exist: {bundle_path}")

    owned_temp: tempfile.TemporaryDirectory[str] | None = None
    restore_removed = False
    restore_path: Path | None = None
    commands: list[CommandResult] = []
    actual_head = ""
    main_ref = ""
    bundle_verify_returncode = -1
    fsck_returncode = -1
    error = ""

    if restore_parent is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="director-ai-restore-")
        restore_parent = Path(owned_temp.name)
    else:
        restore_parent = restore_parent.expanduser().resolve()
        restore_parent.mkdir(parents=True, exist_ok=True)

    try:
        bundle_verify = _run(["git", "bundle", "verify", str(bundle_path)], check=False)
        commands.append(bundle_verify)
        bundle_verify_returncode = bundle_verify.returncode
        if bundle_verify.returncode != 0:
            error = "git bundle verify failed"
            raise BackupVerificationError(error)

        restore_path = restore_parent / _safe_restore_name(bundle_path)
        if restore_path.exists():
            raise BackupVerificationError(
                f"restore path already exists: {restore_path}"
            )

        clone = _run(["git", "clone", str(bundle_path), str(restore_path)])
        commands.append(clone)

        head_result = _run(["git", "rev-parse", "HEAD"], cwd=restore_path)
        commands.append(head_result)
        actual_head = head_result.stdout

        main_result = _run(
            ["git", "show-ref", "--verify", "refs/heads/main"],
            cwd=restore_path,
        )
        commands.append(main_result)
        main_ref = main_result.stdout.split()[0]

        fsck = _run(
            ["git", "fsck", "--full", "--strict"], cwd=restore_path, check=False
        )
        commands.append(fsck)
        fsck_returncode = fsck.returncode
        if fsck.returncode != 0:
            error = "git fsck failed"
            raise BackupVerificationError(error)

        if expected_head is not None and actual_head != expected_head:
            error = f"expected HEAD {expected_head}, restored {actual_head}"
            raise BackupVerificationError(error)

        result = BackupVerificationResult(
            ok=True,
            bundle_path=bundle_path,
            restore_path=restore_path,
            actual_head=actual_head,
            expected_head=expected_head,
            main_ref=main_ref,
            bundle_verify_returncode=bundle_verify_returncode,
            fsck_returncode=fsck_returncode,
            restore_removed=False,
            commands=tuple(commands),
        )
    except BackupVerificationError as exc:
        result = BackupVerificationResult(
            ok=False,
            bundle_path=bundle_path,
            restore_path=restore_path,
            actual_head=actual_head,
            expected_head=expected_head,
            main_ref=main_ref,
            bundle_verify_returncode=bundle_verify_returncode,
            fsck_returncode=fsck_returncode,
            restore_removed=False,
            commands=tuple(commands),
            error=error or str(exc),
        )
        raise BackupVerificationError(str(exc), result) from exc
    finally:
        if not keep_restore:
            if restore_path is not None and restore_path.exists():
                shutil.rmtree(restore_path)
                restore_removed = True
            if owned_temp is not None:
                owned_temp.cleanup()

    if restore_removed:
        result = BackupVerificationResult(
            ok=result.ok,
            bundle_path=result.bundle_path,
            restore_path=result.restore_path,
            actual_head=result.actual_head,
            expected_head=result.expected_head,
            main_ref=result.main_ref,
            bundle_verify_returncode=result.bundle_verify_returncode,
            fsck_returncode=result.fsck_returncode,
            restore_removed=True,
            commands=result.commands,
            error=result.error,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    """Run repository backup restore verification from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Git bundle backup to verify.")
    parser.add_argument("--expected-head", default=None)
    parser.add_argument("--restore-parent", type=Path, default=None)
    parser.add_argument("--keep-restore", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        result = verify_repository_backup(
            args.bundle,
            expected_head=args.expected_head,
            restore_parent=args.restore_parent,
            keep_restore=args.keep_restore,
        )
    except BackupVerificationError as exc:
        result = exc.result
        if result is None:
            print(str(exc))
            return 1
        if args.as_json:
            print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
        else:
            print(result.to_markdown())
        return 1

    if args.as_json:
        print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(result.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
