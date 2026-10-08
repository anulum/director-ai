# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Repository history restore verifier

"""Verify a git bundle backup by restoring it into a fresh checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.repository_restore_ownership import (
    RestoreCustody,
    RestoreDirectory,
    RestoreOwnershipError,
)


@dataclass(frozen=True)
class CommandResult:
    """Captured native command result and its actual directory context.

    Attributes
    ----------
    argv : tuple of str
        Exact argument vector passed to the native process.
    returncode : int
        Actual process exit code.
    stdout, stderr : str
        Captured native output and diagnostics.
    cwd : str or None, optional
        Actual working-directory argument or the inherited current directory.
    directory_fds : tuple of tuple of int, optional
        Inherited descriptor, device and inode identities used by the command.
    """

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    cwd: str | None = None
    directory_fds: tuple[tuple[int, int, int], ...] = ()


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

    def __init__(
        self, message: str, result: BackupVerificationResult | None = None
    ) -> None:
        """Initialise a refusal with its optional completed verification report."""
        super().__init__(message)
        self.result = result


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    pass_fds: tuple[int, ...] = (),
) -> CommandResult:
    """Capture native Git evidence together with the actual held-directory context."""
    directory_fds = tuple(
        (fd, metadata.st_dev, metadata.st_ino)
        for fd in pass_fds
        for metadata in [os.fstat(fd)]
    )
    completed = subprocess.run(
        argv,
        cwd=cwd,
        pass_fds=pass_fds,
        check=False,
        capture_output=True,
        text=True,
    )
    result = CommandResult(
        argv=tuple(argv),
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        cwd=str(cwd) if cwd is not None else os.getcwd(),
        directory_fds=directory_fds,
    )
    return result


def _safe_restore_name(bundle_path: Path) -> str:
    """Derive the existing deterministic restore basename from a bundle name."""
    stem = bundle_path.name.replace(".bundle", "")
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)


def _record_command(
    argv: list[str], commands: list[CommandResult], *, cwd: Path | None = None
) -> CommandResult:
    """Retain the native result before rejecting an unsuccessful Git command."""
    result = _run(argv, cwd=cwd)
    commands.append(result)
    return result


def _require_success(result: CommandResult, message: str) -> None:
    """Reject a failed recorded command with a stable verification reason."""
    if result.returncode != 0:
        raise BackupVerificationError(message)


def _record_checkout_command(
    argv: list[str],
    commands: list[CommandResult],
    custody: RestoreCustody,
    directory: RestoreDirectory,
    *,
    clone: bool = False,
) -> CommandResult:
    """Record native Git evidence after validating its held checkout binding."""
    custody.require_public(directory)
    result = _run(
        argv,
        cwd=None if clone else directory.descriptor_path,
        pass_fds=(directory.fd,),
    )
    commands.append(result)
    return result


def verify_repository_backup(
    bundle_path: Path,
    *,
    expected_head: str | None = None,
    restore_parent: Path | None = None,
    keep_restore: bool = False,
) -> BackupVerificationResult:
    """Restore a Git history bundle without removing caller-owned destinations.

    Parameters
    ----------
    bundle_path : pathlib.Path
        Existing history bundle to check with native Git.
    expected_head : str or None, optional
        Exact commit expected in the restored checkout, when supplied.
    restore_parent : pathlib.Path or None, optional
        Parent of the deterministic bundle-derived restore directory. An
        existing child is refused and preserved. If omitted, create a private
        temporary parent under the configured temporary directory.
    keep_restore : bool, optional
        Preserve the owned checkout and private temporary parent, including
        after verification fails. The caller is responsible for their removal.

    Returns
    -------
    BackupVerificationResult
        Native bundle, HEAD, main-ref and fsck evidence after cleanup completes.

    Raises
    ------
    BackupVerificationError
        The bundle, restoration, expected identity or owned cleanup fails.
        Its result records actual commands and the final cleanup outcome.

    Notes
    -----
    The original directory is created and held inside account-private custody
    before atomic publication without overwriting an existing destination.
    Native Git operates through the inherited original directory descriptor.
    Cleanup captures and verifies the entry before traversing the held object.
    A supplied parent is never removed. Linux descriptors, procfs and native
    no-overwrite rename are required; there is no unsafe cleanup fallback.
    Bundle verification proves history restoration, not an entire backup.
    """
    bundle_path = bundle_path.expanduser().resolve()
    if not bundle_path.is_file():
        raise BackupVerificationError(f"bundle does not exist: {bundle_path}")

    custody: RestoreCustody | None = None
    owned_parent: RestoreDirectory | None = None
    checkout: RestoreDirectory | None = None
    restore_removed = False
    restore_path: Path | None = None
    commands: list[CommandResult] = []
    actual_head = ""
    main_ref = ""
    bundle_verify_returncode = -1
    fsck_returncode = -1
    errors: list[str] = []

    try:
        bundle_verify = _record_command(
            ["git", "bundle", "verify", str(bundle_path)], commands
        )
        bundle_verify_returncode = bundle_verify.returncode
        _require_success(bundle_verify, "git bundle verify failed")

        custody = RestoreCustody(restore_parent)
        if restore_parent is None:
            owned_parent = custody.create_parent()
            restore_parent = owned_parent.path
            parent_fd = owned_parent.fd
        else:
            restore_parent = custody.base
            parent_fd = custody.base_fd

        restore_path = restore_parent / _safe_restore_name(bundle_path)
        checkout = custody.create_checkout(
            parent_fd, restore_parent, _safe_restore_name(bundle_path)
        )

        clone = _record_checkout_command(
            ["git", "clone", str(bundle_path), str(checkout.descriptor_path)],
            commands,
            custody,
            checkout,
            clone=True,
        )
        custody.require_public(checkout)
        _require_success(clone, "git clone failed")

        head_result = _record_checkout_command(
            ["git", "rev-parse", "HEAD"], commands, custody, checkout
        )
        if head_result.returncode == 0:
            actual_head = head_result.stdout
        custody.require_public(checkout)
        _require_success(head_result, "restored HEAD cannot be resolved")

        main_result = _record_checkout_command(
            ["git", "show-ref", "--verify", "refs/heads/main"],
            commands,
            custody,
            checkout,
        )
        if main_result.returncode == 0:
            main_ref = main_result.stdout.split()[0]
        custody.require_public(checkout)
        _require_success(main_result, "restored main ref cannot be resolved")

        fsck = _record_checkout_command(
            ["git", "fsck", "--full", "--strict"], commands, custody, checkout
        )
        fsck_returncode = fsck.returncode
        custody.require_public(checkout)
        _require_success(fsck, "git fsck failed")

        if expected_head is not None and actual_head != expected_head:
            raise BackupVerificationError(
                f"expected HEAD {expected_head}, restored {actual_head}"
            )
    except (
        BackupVerificationError,
        RestoreOwnershipError,
        OSError,
        AttributeError,
    ) as exc:
        errors.append(str(exc))
    finally:
        if custody is not None:
            if not keep_restore:
                if checkout is not None:
                    try:
                        custody.remove(checkout)
                        restore_removed = True
                    except (RestoreOwnershipError, OSError) as exc:
                        errors.append(f"restore cleanup failed: {exc}")
                if owned_parent is not None:
                    try:
                        custody.remove(owned_parent, empty_only=True)
                    except (RestoreOwnershipError, OSError) as exc:
                        errors.append(f"temporary parent cleanup failed: {exc}")
            errors.extend(custody.close())

    result = BackupVerificationResult(
        ok=not errors,
        bundle_path=bundle_path,
        restore_path=restore_path,
        actual_head=actual_head,
        expected_head=expected_head,
        main_ref=main_ref,
        bundle_verify_returncode=bundle_verify_returncode,
        fsck_returncode=fsck_returncode,
        restore_removed=restore_removed,
        commands=tuple(commands),
        error="; ".join(errors),
    )
    if errors:
        raise BackupVerificationError(result.error, result)
    return result


def main(argv: list[str] | None = None) -> int:
    """Verify history restoration and return a CLI success or refusal status.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments; None uses the process argument vector.

    Returns
    -------
    int
        Zero for verified restoration, one for a reported verification failure.
    """
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
        failed_result = exc.result
        if failed_result is None:
            print(str(exc))
            return 1
        if args.as_json:
            print(json.dumps(failed_result.to_json_dict(), indent=2, sort_keys=True))
        else:
            print(failed_result.to_markdown())
        return 1

    if args.as_json:
        print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
    else:
        print(result.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
