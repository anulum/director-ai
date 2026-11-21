# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Weight Archive Reader

"""A hardened reader for ``safetensors_tar`` weight archives.

A ``safetensors_tar`` is an uncompressed POSIX ustar archive holding one or
more ``*.safetensors`` shards and at most one ``config.json`` and one
``tokenizer.json``. The reader refuses:

* any header that is not POSIX ustar (pax ``x``/``g`` and GNU ``L``/``K``
  headers included), a bad header checksum, or a non-empty ``prefix``;
* any member that is not a regular file, or whose name has a ``/``, is
  ``..``, starts with ``.``, repeats, or is not one of the allowed names;
* non-zero padding between a member's data and its 512-byte boundary;
* an archive whose length is not a multiple of 512, that lacks the two
  end-of-archive blocks, or that has a non-zero byte after them or more than
  10,240 bytes from the first end block to the end.

The shards are parsed by :mod:`.safetensors_parser`, and their tensors form
one model-wide :class:`~.tensors.TensorSetIdentity`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .errors import MalformedWeightsError
from .safetensors_parser import read_safetensors_descriptors
from .tensors import TensorDescriptor, TensorSetIdentity, tensor_set_identity

_BLOCK = 512
_MAX_END_REGION = 10_240
_ZERO_BLOCK = bytes(_BLOCK)
_OCTAL = re.compile(rb"[0-7]+")
_SIDE_FILES = frozenset({"config.json", "tokenizer.json"})


@dataclass(frozen=True, slots=True)
class ArchiveMember:
    """One regular file inside a weight archive.

    Attributes
    ----------
    name : str
        Single-component member name.
    data_offset : int
        Byte offset of the member's data inside the archive.
    size : int
        Byte length of the member's data.
    """

    name: str
    data_offset: int
    size: int


def _octal(field: bytes, what: str) -> int:
    """Parse a NUL- or space-terminated octal header field."""
    digits = field.rstrip(b"\x00 ")
    if _OCTAL.fullmatch(digits) is None:
        raise MalformedWeightsError(f"weight archive {what} field is not octal")
    return int(digits, 8)


def _member_name(header: bytes) -> str:
    """Return a member's validated single-component name."""
    raw = header[:100].split(b"\x00", 1)[0]
    try:
        name = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MalformedWeightsError("weight archive member name is not UTF-8") from exc
    if not name or "/" in name or name.startswith("."):
        raise MalformedWeightsError(
            f"weight archive member name {name!r} is not allowed"
        )
    if name not in _SIDE_FILES and not name.endswith(".safetensors"):
        raise MalformedWeightsError(
            f"weight archive member {name!r} is not a weight shard"
        )
    return name


def _check_header(header: bytes) -> None:
    """Refuse a header that is not a valid POSIX ustar regular-file header."""
    if header[257:263] != b"ustar\x00" or header[263:265] != b"00":
        raise MalformedWeightsError("weight archive header is not POSIX ustar")
    expected = sum(header[:148]) + 8 * 0x20 + sum(header[156:])
    if _octal(header[148:156], "checksum") != expected:
        raise MalformedWeightsError("weight archive header checksum does not match")
    if header[156:157] not in (b"0", b"\x00"):
        raise MalformedWeightsError("weight archive member is not a regular file")
    if any(header[345:500]):
        raise MalformedWeightsError("weight archive header uses the ustar prefix field")


def _check_end(handle: BinaryIO, end_offset: int, length: int) -> None:
    """Refuse anything but zero padding after the end-of-archive blocks."""
    if length - end_offset < 2 * _BLOCK or length - end_offset > _MAX_END_REGION:
        raise MalformedWeightsError("weight archive end region has the wrong length")
    handle.seek(end_offset)
    if any(handle.read(length - end_offset)):
        raise MalformedWeightsError("weight archive has data after its end blocks")


def list_archive_members(handle: BinaryIO, length: int) -> tuple[ArchiveMember, ...]:
    """Return the members of a ``safetensors_tar`` after checking its framing.

    Parameters
    ----------
    handle : BinaryIO
        An open, seekable binary file holding the archive.
    length : int
        Byte length of the archive.

    Returns
    -------
    tuple[ArchiveMember, ...]
        Members in archive order.

    Raises
    ------
    MalformedWeightsError
        If any archive rule is broken.
    """
    if length % _BLOCK:
        raise MalformedWeightsError("weight archive length is not a multiple of 512")
    members: list[ArchiveMember] = []
    offset = 0
    while True:
        if offset + _BLOCK > length:
            raise MalformedWeightsError("weight archive has no end-of-archive blocks")
        handle.seek(offset)
        header = handle.read(_BLOCK)
        if header == _ZERO_BLOCK:
            _check_end(handle, offset, length)
            break
        _check_header(header)
        name = _member_name(header)
        if any(member.name == name for member in members):
            raise MalformedWeightsError(f"weight archive repeats member {name!r}")
        size = _octal(header[124:136], "size")
        data_offset = offset + _BLOCK
        offset = data_offset + -(-size // _BLOCK) * _BLOCK
        if offset > length:
            raise MalformedWeightsError(f"weight archive member {name!r} is truncated")
        handle.seek(data_offset + size)
        if any(handle.read(offset - data_offset - size)):
            raise MalformedWeightsError(
                f"weight archive member {name!r} has non-zero padding"
            )
        members.append(ArchiveMember(name, data_offset, size))
    if not any(member.name.endswith(".safetensors") for member in members):
        raise MalformedWeightsError("weight archive holds no safetensors shard")
    return tuple(members)


def read_archive_tensor_descriptors(
    path: str | os.PathLike[str], *, forbid_negative_zero: bool = False
) -> tuple[TensorDescriptor, ...]:
    """Return every tensor descriptor of a ``safetensors_tar``'s shards.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The archive file.
    forbid_negative_zero : bool
        Whether a negative zero in a floating-point tensor is refused too.
        NaN and infinities are always refused.

    Returns
    -------
    tuple[TensorDescriptor, ...]
        The descriptors of every ``.safetensors`` member, in archive order.

    Raises
    ------
    MalformedWeightsError
        If the archive or any shard breaks the hardened rules.
    """
    with Path(path).open("rb") as handle:
        length = os.fstat(handle.fileno()).st_size
        descriptors: list[TensorDescriptor] = []
        for member in list_archive_members(handle, length):
            if member.name.endswith(".safetensors"):
                descriptors.extend(
                    read_safetensors_descriptors(
                        handle,
                        offset=member.data_offset,
                        length=member.size,
                        source=member.name,
                        forbid_negative_zero=forbid_negative_zero,
                    )
                )
    return tuple(descriptors)


def compute_archive_tensor_set_identity(
    path: str | os.PathLike[str], *, forbid_negative_zero: bool = False
) -> TensorSetIdentity:
    """Return the model-wide tensor-set identity of a ``safetensors_tar``.

    Parameters
    ----------
    path : str | os.PathLike[str]
        The archive file.
    forbid_negative_zero : bool
        Whether a negative zero in a floating-point tensor is refused too.
        NaN and infinities are always refused.

    Returns
    -------
    TensorSetIdentity
        The identity over every shard's tensors, equal to the identity of
        the same tensors in one safetensors file.

    Raises
    ------
    MalformedWeightsError
        If the archive or any shard breaks the hardened rules, or two shards
        share a tensor name.
    """
    return tensor_set_identity(
        read_archive_tensor_descriptors(path, forbid_negative_zero=forbid_negative_zero)
    )
