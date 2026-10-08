# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Descriptor-bound repository restore directory custody

"""Keep restore allocation and deletion bound to held directory objects."""

from __future__ import annotations

import ctypes
import os
import stat
import sys
import tempfile
import uuid
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


class RestoreOwnershipError(RuntimeError):
    """Refuse restoration when directory identity or private custody is uncertain."""


@dataclass
class RestoreDirectory:
    """Held invocation-created directory and its currently published name.

    Attributes
    ----------
    fd : int
        Original directory descriptor, held until the invocation finishes.
    parent_fd : int
        Descriptor of the parent in which the directory is currently named.
    path : pathlib.Path
        Published path, or the initial account-private allocation path.
    private_name : str
        Reserved name used inside this invocation's private custody directory.
    published : bool
        Whether atomic publication to the requested name succeeded.
    """

    fd: int
    parent_fd: int
    path: Path
    private_name: str
    published: bool = False

    @property
    def descriptor_path(self) -> Path:
        """Return the live original object path for inherited native Git access."""
        return Path(f"/proc/self/fd/{self.fd}")


def _same_object(metadata: os.stat_result, fd: int) -> bool:
    """Compare an entry with its held object, whose inode cannot be recycled."""
    held = os.fstat(fd)
    return (metadata.st_dev, metadata.st_ino) == (held.st_dev, held.st_ino)


def _rename_without_replace(
    source_fd: int, source: str, target_fd: int, target: str
) -> None:
    """Atomically move an entry without ever overwriting an existing target."""
    function = ctypes.CDLL(None, use_errno=True).renameat2
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    sys.audit("os.rename", source, target, source_fd, target_fd)
    if function(source_fd, os.fsencode(source), target_fd, os.fsencode(target), 1):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), source, None, target)


def _delete_contents(fd: int) -> None:
    """Traverse only the held checkout object using native descriptor operations."""
    with os.scandir(fd) as entries:
        for entry in entries:
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                )
                try:
                    if not _same_object(metadata, child):
                        raise RestoreOwnershipError(
                            "checkout entry changed while opening; cleanup refused"
                        )
                    _delete_contents(child)
                finally:
                    os.close(child)
                os.rmdir(entry.name, dir_fd=fd)
            else:
                os.unlink(entry.name, dir_fd=fd)


class RestoreCustody:
    """Own a private workspace and hold all directory descriptors to completion.

    Parameters
    ----------
    restore_parent : pathlib.Path or None
        Supplied parent, which is borrowed and never deleted. None uses the
        configured temporary directory as the publication base.

    Notes
    -----
    Linux directory descriptors, procfs and renameat2(RENAME_NOREPLACE) are
    required; unsupported operations fail without a pathname-based fallback.
    The account-private custody root must belong to the effective user and
    have mode0700. Processes authorised as that same account are trusted
    inside this private root; public names remain safe under replacement.
    The small root persists, so its own removal never reopens an untrusted
    final-name interval. Only invocation-created child entries are removed.
    """

    def __init__(self, restore_parent: Path | None) -> None:
        """Open private custody before allocating or publishing any checkout."""
        self._stack = ExitStack()
        self._directories: list[RestoreDirectory] = []
        try:
            self.base = (
                Path(tempfile.gettempdir()).resolve()
                if restore_parent is None
                else restore_parent.expanduser().resolve()
            )
            self.base.mkdir(parents=True, exist_ok=True)
            self.base_fd = self._hold(self.base)
            self.root = self.base / f".director-ai-restore-custody-{os.geteuid()}"
            self.root.mkdir(mode=0o700, exist_ok=True)
            self.root_fd = self._hold(self.root)
            self._require_private(self.root_fd)
            self.workspace = self.root / uuid.uuid4().hex
            os.mkdir(self.workspace.name, mode=0o700, dir_fd=self.root_fd)
            self.workspace_fd = self._hold(self.workspace.name, parent_fd=self.root_fd)
            self._require_private(self.workspace_fd)
        except (OSError, AttributeError, RestoreOwnershipError):
            self._stack.close()
            raise

    def _hold(self, path: Path | str, *, parent_fd: int | None = None) -> int:
        """Open and retain a non-symlink directory until every operation finishes."""
        fd = os.open(
            path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        self._stack.callback(os.close, fd)
        return fd

    @staticmethod
    def _require_private(fd: int) -> None:
        """Enforce the account-owned0700 custody condition before private mutation."""
        metadata = os.fstat(fd)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
            raise RestoreOwnershipError("restore custody is not account-private0700")

    def _require_custody(self) -> None:
        """Refuse moved or replaced private authority before mutating its entries."""
        self._require_private(self.root_fd)
        self._require_private(self.workspace_fd)
        root = self.root.lstat()
        workspace = os.stat(
            self.workspace.name, dir_fd=self.root_fd, follow_symlinks=False
        )
        if not _same_object(root, self.root_fd) or not _same_object(
            workspace, self.workspace_fd
        ):
            raise RestoreOwnershipError("private custody identity changed; refused")

    def _allocate(self, name: str) -> RestoreDirectory:
        """Create and open the original object inside protected private custody."""
        self._require_custody()
        os.mkdir(name, mode=0o700, dir_fd=self.workspace_fd)
        fd = self._hold(name, parent_fd=self.workspace_fd)
        directory = RestoreDirectory(fd, self.workspace_fd, self.workspace / name, name)
        self._directories.append(directory)
        return directory

    def _publish(
        self, directory: RestoreDirectory, parent_fd: int, path: Path
    ) -> RestoreDirectory:
        """Publish the already-held original object with atomic no-overwrite rename."""
        try:
            _rename_without_replace(
                self.workspace_fd, directory.private_name, parent_fd, path.name
            )
        except FileExistsError as exc:
            raise RestoreOwnershipError(f"restore path already exists: {path}") from exc
        directory.parent_fd = parent_fd
        directory.path = path
        directory.published = True
        return directory

    def create_parent(self) -> RestoreDirectory:
        """Publish a held private default parent directly under the temporary base."""
        directory = self._allocate("parent")
        path = self.base / f"director-ai-restore-{uuid.uuid4().hex}"
        return self._publish(directory, self.base_fd, path)

    def create_checkout(
        self, parent_fd: int, parent: Path, name: str
    ) -> RestoreDirectory:
        """Reserve the requested name only after holding its privately-created object."""
        path = parent / name
        if name in {"", ".", ".."}:
            raise RestoreOwnershipError(f"restore path already exists: {path}")
        return self._publish(self._allocate("checkout"), parent_fd, path)

    @staticmethod
    def _require_parent(directory: RestoreDirectory) -> None:
        """Refuse a changed parent or ancestor instead of operating via a moved alias."""
        if not _same_object(directory.path.parent.lstat(), directory.parent_fd):
            raise RestoreOwnershipError("restore parent changed; operation refused")

    def require_public(self, directory: RestoreDirectory) -> None:
        """Verify the public name still refers to the already-held original object."""
        self._require_parent(directory)
        metadata = os.stat(
            directory.path.name, dir_fd=directory.parent_fd, follow_symlinks=False
        )
        if not _same_object(metadata, directory.fd):
            raise RestoreOwnershipError("restore directory ownership changed; refused")

    def _return_captured(self, directory: RestoreDirectory) -> None:
        """Restore a captured entry without replacing anything at its original name."""
        try:
            _rename_without_replace(
                self.workspace_fd,
                directory.private_name,
                directory.parent_fd,
                directory.path.name,
            )
        except OSError as exc:
            actual = Path(os.readlink(f"/proc/self/fd/{self.workspace_fd}"))
            raise RestoreOwnershipError(
                f"entry preserved at {actual / directory.private_name}; "
                "original name unavailable; no replacement overwritten"
            ) from exc

    def remove(self, directory: RestoreDirectory, *, empty_only: bool = False) -> None:
        """Capture and validate ownership before deleting within private custody.

        Parameters
        ----------
        directory : RestoreDirectory
            Invocation-created object whose original descriptor is still held.
        empty_only : bool, optional
            Refuse a nonempty default parent rather than deleting any contents.

        Raises
        ------
        RestoreOwnershipError
            Public identity, custody or recovery is uncertain. Entries with
            uncertain ownership are preserved; owned contents may be removed.
        OSError
            Native capture, traversal or removal fails; captured contents are
            returned without overwriting a concurrently acquired public name.
        """
        self._require_custody()
        self.require_public(directory)
        sys.audit(
            "os.rmdir" if empty_only else "shutil.rmtree",
            directory.path,
            directory.parent_fd,
        )
        _rename_without_replace(
            directory.parent_fd,
            directory.path.name,
            self.workspace_fd,
            directory.private_name,
        )
        try:
            self._require_parent(directory)
            metadata = os.stat(
                directory.private_name, dir_fd=self.workspace_fd, follow_symlinks=False
            )
            if not _same_object(metadata, directory.fd):
                raise RestoreOwnershipError("captured entry ownership changed; refused")
            if not empty_only:
                _delete_contents(directory.fd)
            os.rmdir(directory.private_name, dir_fd=self.workspace_fd)
        except (OSError, RestoreOwnershipError):
            self._return_captured(directory)
            raise

    def close(self) -> list[str]:
        """Remove only unused private allocations and close every held descriptor."""
        errors: list[str] = []
        try:
            self._require_custody()
            for directory in self._directories:
                if not directory.published:
                    metadata = os.stat(
                        directory.private_name,
                        dir_fd=self.workspace_fd,
                        follow_symlinks=False,
                    )
                    if not _same_object(metadata, directory.fd):
                        raise RestoreOwnershipError(
                            "unused allocation changed; refused"
                        )
                    os.rmdir(directory.private_name, dir_fd=self.workspace_fd)
            metadata = os.stat(
                self.workspace.name, dir_fd=self.root_fd, follow_symlinks=False
            )
            if not _same_object(metadata, self.workspace_fd):
                raise RestoreOwnershipError(
                    "private workspace identity changed; refused"
                )
            os.rmdir(self.workspace.name, dir_fd=self.root_fd)
        except (OSError, RestoreOwnershipError) as exc:
            errors.append(f"private custody cleanup failed: {exc}")
        finally:
            self._stack.close()
        return errors
