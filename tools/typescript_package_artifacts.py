# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — complete npm package member comparison
"""Compare an actual npm archive with explicitly supplied member bytes and modes.

This byte-comparison API does not run generators or attest publication. Its
caller owns the source-bound native recipe that produces the expected members.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import os
import stat
import tarfile
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from tools.quality_ratchet_contracts import repository_path, sha256_digest


@dataclass(frozen=True)
class PackageContents:
    """Retain the actual archive identity and every compared regular member.

    Attributes
    ----------
    archive_sha256 : str
        SHA-256 of the original compressed archive, without rewriting it.
    members : dict[str, tuple[str, int]]
        Original archive names mapped to their exact content digest and mode.
    """

    archive_sha256: str
    members: dict[str, tuple[str, int]]


def _remaining(deadline: float) -> None:
    """Keep archive reads and final custody inside the caller's shared budget."""
    if time.monotonic() >= deadline:
        raise ValueError("Native npm package comparison exhausted its deadline")


def _digest(stream: IO[bytes], deadline: float) -> str:
    """Hash complete original bytes in bounded chunks, without extraction."""
    result = hashlib.sha256()
    while chunk := stream.read(65536):
        _remaining(deadline)
        result.update(chunk)
    return result.hexdigest()


def inspect_npm_package(
    archive: Path,
    expected: Mapping[str, tuple[str, int]],
    *,
    deadline: float,
) -> PackageContents:
    """Compare all actual tar members with an explicit native-producer manifest.

    Parameters
    ----------
    archive : Path
        Exact physical regular gzip-compressed npm archive. No path is extracted.
    expected : Mapping[str, tuple[str, int]]
        Complete original ``package/...`` names with expected SHA-256 and mode.
        The caller must derive these from the real source/compiler/packer recipe;
        a caller-editable manifest is not generator or publication authority.
    deadline : float
        Positive finite absolute monotonic deadline shared with the native producer.

    Returns
    -------
    PackageContents
        Original compressed archive identity and complete matching member set.

    Raises
    ------
    ValueError, OSError
        If expected members, actual names, bytes, modes, format or custody differ,
        or the deadline expires. Links and duplicate members always refuse.
    """
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("Native npm package deadline must be positive and finite")
    _remaining(deadline)
    required = {
        "package/package.json",
        "package/README.md",
        "package/LICENSE",
        "package/LICENSE.license",
        "package/dist/index.js",
        "package/dist/index.d.ts",
    }
    manifest: dict[str, tuple[str, int]] = {}
    for name, (digest, mode) in expected.items():
        name = repository_path(name)
        if (
            not name.startswith("package/")
            or type(mode) is not int
            or not 0 <= mode <= 0o777
        ):
            raise ValueError("Invalid original npm package member identity")
        manifest[name] = (sha256_digest(digest), mode)
    if not required <= manifest.keys():
        raise ValueError("Original npm package manifest lacks public package members")
    archive = archive.absolute()
    if archive.resolve(strict=True) != archive:
        raise ValueError("Original npm package archive is a symbolic alias")
    original = archive.lstat()
    if not stat.S_ISREG(original.st_mode):
        raise ValueError("Original npm package archive is not regular")
    actual: dict[str, tuple[str, int]] = {}
    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            original.st_dev,
            original.st_ino,
            original.st_size,
            original.st_mtime_ns,
        ):
            raise ValueError("Original npm package changed before capture")
        digest = _digest(stream, deadline)
        stream.seek(0)
        try:
            with gzip.GzipFile(fileobj=stream, mode="rb") as decoded:
                with tarfile.open(fileobj=decoded, mode="r:") as package:
                    for member in package:
                        _remaining(deadline)
                        if (
                            not member.isfile()
                            or member.name not in manifest
                            or member.name in actual
                        ):
                            raise ValueError(
                                "Actual npm package has extra, repeated or nonregular members"
                            )
                        content = package.extractfile(member)
                        assert content is not None  # Regular members have byte streams.
                        with content:
                            value = (_digest(content, deadline), member.mode)
                        if value != manifest[member.name]:
                            raise ValueError(
                                "Actual npm package member bytes or mode differ"
                            )
                        actual[member.name] = value
                padding = 0
                while chunk := decoded.read(65536):
                    _remaining(deadline)
                    if chunk.strip(b"\0"):
                        raise ValueError(
                            "Actual npm package has unparsed trailing payload"
                        )
                    padding += len(chunk)
                if padding < 512:
                    raise ValueError("Actual npm package end markers are incomplete")
        except (tarfile.TarError, gzip.BadGzipFile, EOFError, zlib.error) as error:
            raise ValueError(
                "Original npm package is not a complete valid archive"
            ) from error
        if actual != manifest:
            raise ValueError("Actual npm package member inventory is incomplete")
        stream.seek(0)
        if _digest(stream, deadline) != digest:
            raise ValueError("Original npm package bytes changed during comparison")
    final = archive.lstat()
    if archive.resolve(strict=True) != archive or (
        final.st_dev,
        final.st_ino,
        final.st_size,
        final.st_mtime_ns,
    ) != (original.st_dev, original.st_ino, original.st_size, original.st_mtime_ns):
        raise ValueError("Original npm package custody changed during comparison")
    _remaining(deadline)
    return PackageContents(digest, dict(sorted(actual.items())))
