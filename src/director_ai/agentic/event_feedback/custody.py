# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — Private immutable review custody

"""Bind original custody and containing entries to one descriptor-relative walk."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .codec import FeedbackInvalidError, canonical


def private_bytes(path: Path, directory_fd: int | None = None) -> bytes:
    """Read an owner-only regular file without following its final symlink.

    Parameters
    ----------
    path : Path
        Trusted operator-selected credential or evidence file.
    directory_fd : int or None
        Verified directory handle for a relative immutable-collision read.

    Returns
    -------
    bytes
        Complete contents, limited by the protocol's 1 MiB admission.

    Raises
    ------
    FeedbackInvalidError
        If custody, permissions, file identity or admission size is invalid.
    """
    try:
        fd = os.open(
            path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd
        )
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
            ):
                raise FeedbackInvalidError()
            raw = stream.read(1048577)
            if len(raw) > 1048576:
                raise FeedbackInvalidError()
            return raw
    except (OSError, AttributeError, ValueError) as exc:
        raise FeedbackInvalidError() from exc


def _single_name(name: str) -> None:
    """Refuse traversal outside a descriptor's original custody."""
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise FeedbackInvalidError()


@dataclass(frozen=True)
class EvidenceRoot:
    """Bind a pre-existing operator root across process restarts.

    Attributes
    ----------
    path : Path
        Absolute registered private root; the producer never creates it.
    device : int
        Original filesystem device from the private operator profile.
    inode : int
        Original inode from the same profile; never refreshed automatically.
    """

    path: Path
    device: int
    inode: int

    def __post_init__(self) -> None:
        """Refuse an unbound, relative or structurally invalid root identity."""
        if (
            not self.path.is_absolute()
            or self.path == Path("/")
            or type(self.device) is not int
            or self.device < 0
            or type(self.inode) is not int
            or self.inode <= 0
        ):
            raise FeedbackInvalidError()


@dataclass(frozen=True)
class PinnedDirectory:
    """Keep checked parent/child relationships open with the original leaf.

    Attributes
    ----------
    path : Path
        Custody reference; operations use descriptors rather than this path.
    descriptor : int
        Verified private leaf handle, held throughout the context.
    bindings : tuple
        Parent descriptors, child names and their original device/inode identities.
    sync_descriptors : tuple
        Leaf and containing descriptors on its filesystem, in sync order.
    """

    path: Path
    descriptor: int
    bindings: tuple[tuple[int, str, int, int, bool], ...]
    sync_descriptors: tuple[int, ...]

    def verify(self) -> None:
        """Refuse renamed, replaced, symbolic or newly public custody ancestry."""
        try:
            for parent, name, device, inode, private in self.bindings:
                current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (
                    not stat.S_ISDIR(current.st_mode)
                    or (
                        current.st_dev,
                        current.st_ino,
                    )
                    != (device, inode)
                    or (
                        private
                        and (current.st_uid != os.getuid() or current.st_mode & 0o077)
                    )
                ):
                    raise FeedbackInvalidError()
            leaf = os.fstat(self.descriptor)
            if leaf.st_uid != os.getuid() or leaf.st_mode & 0o077:
                raise FeedbackInvalidError()
        except OSError as exc:
            raise FeedbackInvalidError() from exc

    def synchronise(self) -> None:
        """Sync the SAME leaf/parent handles and recheck their identities."""
        self.verify()
        for descriptor in self.sync_descriptors:
            os.fsync(descriptor)
        self.verify()

    def exists(self, name: str) -> bool:
        """Check a single original name relative to the held custody handle.

        Parameters
        ----------
        name : str
            Caller-selected single file name.

        Returns
        -------
        bool
            Whether any entry already reserves this original name.
        """
        _single_name(name)
        self.verify()
        try:
            os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    @contextmanager
    def child(self, name: str) -> Iterator[PinnedDirectory]:
        """Hold a private transport child relative to this SAME parent handle.

        Parameters
        ----------
        name : str
            Single caller-selected child name, without traversal.

        Yields
        ------
        PinnedDirectory
            Verified child and the inherited original ancestry handles.

        Raises
        ------
        FeedbackInvalidError
            If creation, private custody or any ancestor identity check fails.
        """
        _single_name(name)
        self.verify()
        try:
            with suppress(FileExistsError):
                os.mkdir(name, 0o700, dir_fd=self.descriptor)
            with ExitStack() as descriptors:
                fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=self.descriptor,
                )
                descriptors.callback(os.close, fd)
                identity = os.fstat(fd)
                child = PinnedDirectory(
                    self.path / name,
                    fd,
                    (
                        *self.bindings,
                        (self.descriptor, name, identity.st_dev, identity.st_ino, True),
                    ),
                    (fd, *self.sync_descriptors),
                )
                child.synchronise()
                yield child
        except (OSError, AttributeError) as exc:
            raise FeedbackInvalidError() from exc

    def reserve(self, name: str, raw: bytes) -> None:
        """Durably reserve one invocation in the SAME original directory.

        Parameters
        ----------
        name : str
            Single immutable reservation file name.
        raw : bytes
            Complete original invocation identity.
        """
        _single_name(name)
        self.verify()
        fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self.descriptor,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        self.synchronise()

    def retain(self, name: str, raw: bytes) -> None:
        """Publish bytes or verify identical existing bytes through this handle.

        Parameters
        ----------
        name : str
            Single original file name; never a request-selected path.
        raw : bytes
            Complete original bytes, without in-place updates.
        """
        _single_name(name)
        self.verify()
        temporary = ".review-" + uuid4().hex
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=self.descriptor,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self.verify()
            try:
                os.link(
                    temporary,
                    name,
                    src_dir_fd=self.descriptor,
                    dst_dir_fd=self.descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError:
                if private_bytes(Path(name), self.descriptor) != raw:
                    raise FeedbackInvalidError() from None
            self.synchronise()
        finally:
            os.unlink(temporary, dir_fd=self.descriptor)


@contextmanager
def durable_directory(
    path: Path, root: EvidenceRoot | None = None, *, create: bool = True
) -> Iterator[PinnedDirectory]:
    """Walk from a stable root handle without following any path component.

    Parameters
    ----------
    path : Path
        Operator-selected private leaf; dot-parent components are refused.
    root : EvidenceRoot or None
        Registered root binding for producer operations. Anonymous file retention
        may establish a fresh private directory without a root pin.
    create : bool
        Whether to create missing custody and its identity record. Recovery uses
        False and requires the existing original directory and root record.

    Yields
    ------
    PinnedDirectory
        Same leaf/ancestry handles for originals, reservation, sync and contact.

    Raises
    ------
    FeedbackInvalidError
        If ancestry changes, a root is replaced or any custody operation fails.
    """
    absolute = path.absolute()
    if ".." in absolute.parts or (
        root is not None and absolute != root.path and absolute.parent != root.path
    ):
        raise FeedbackInvalidError()
    try:
        with ExitStack() as descriptors:
            parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            descriptors.callback(os.close, parent)
            handles = [parent]
            bindings: list[tuple[int, str, int, int, bool]] = []
            current = Path("/")
            for name in absolute.parts[1:]:
                current = current / name
                if create and (root is None or root.path in current.parents):
                    with suppress(FileExistsError):
                        os.mkdir(name, 0o700, dir_fd=parent)
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                descriptors.callback(os.close, child)
                identity = os.fstat(child)
                if (
                    root is not None
                    and current == root.path
                    and (
                        (identity.st_dev, identity.st_ino) != (root.device, root.inode)
                        or identity.st_uid != os.getuid()
                        or identity.st_mode & 0o077
                    )
                ):
                    raise FeedbackInvalidError()
                bindings.append(
                    (
                        parent,
                        name,
                        identity.st_dev,
                        identity.st_ino,
                        identity.st_uid == os.getuid() and not identity.st_mode & 0o077,
                    )
                )
                handles.append(child)
                parent = child
            device = os.fstat(parent).st_dev
            custody = PinnedDirectory(
                absolute,
                parent,
                tuple(bindings),
                tuple(fd for fd in reversed(handles) if os.fstat(fd).st_dev == device),
            )
            custody.verify()
            if root is not None and absolute != root.path:
                parent_device = os.fstat(handles[-2]).st_dev
                parent_custody = PinnedDirectory(
                    root.path,
                    handles[-2],
                    tuple(bindings[:-1]),
                    tuple(
                        fd
                        for fd in reversed(handles[:-1])
                        if os.fstat(fd).st_dev == parent_device
                    ),
                )
                identity = os.fstat(parent)
                marker = ".review-directory-" + absolute.name
                expected = canonical(
                    {"device": identity.st_dev, "inode": identity.st_ino}
                )
                if create:
                    parent_custody.retain(marker, expected)
                elif private_bytes(Path(marker), parent_custody.descriptor) != expected:
                    raise FeedbackInvalidError()
            if create:
                custody.synchronise()
            yield custody
    except (OSError, AttributeError, ValueError) as exc:
        raise FeedbackInvalidError() from exc


def retain(path: Path, raw: bytes) -> None:
    """Publish original bytes atomically or verify an identical original.

    Parameters
    ----------
    path : Path
        Private operator-selected original destination.
    raw : bytes
        Complete original content; never an in-place update.

    Raises
    ------
    FeedbackInvalidError
        If existing bytes disagree, ancestry changes or publication is uncertain.
    """
    with durable_directory(path.parent) as custody:
        custody.retain(path.name, raw)
