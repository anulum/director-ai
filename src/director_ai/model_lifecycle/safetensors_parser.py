# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Hardened Safetensors Parser

"""A hardened, standard-library reader for safetensors weight files.

The reader never imports ``safetensors`` or ``torch``. It enforces:

* a header length ``n`` with ``0 < n <= 100 MiB`` that fits the file;
* a strict UTF-8 JSON header object with no duplicate member names;
* an optional ``__metadata__`` object of string values;
* for each tensor, a dtype from the closed set, a shape of non-negative
  integers, and ``0 <= start <= end <=`` the data length with
  ``end - start`` equal to the shape's element count times the dtype width;
* slices that partition the data section exactly: no overlap, no gap and no
  trailing bytes.

Tensor bytes are hashed in 1 MiB chunks from the open file.
"""

from __future__ import annotations

import hashlib
import os
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO, cast

from .digests import strict_json_loads
from .errors import (
    MalformedWeightsError,
    NonCanonicalJsonError,
    UnsupportedSafetensorsDtypeError,
)
from .float_scan import scan_float_chunk
from .tensors import (
    DTYPE_WIDTHS,
    TensorDescriptor,
    TensorDtype,
    TensorSetIdentity,
    tensor_set_identity,
)

MAX_HEADER_BYTES = 100 * 1024 * 1024
"""Largest safetensors header the reader accepts."""

_CHUNK = 1 << 20
_ENTRY_KEYS = frozenset({"dtype", "shape", "data_offsets"})


def _read_exact(handle: BinaryIO, size: int, source: str) -> bytes:
    """Read exactly ``size`` bytes or refuse a truncated file."""
    data = handle.read(size)
    if len(data) != size:
        raise MalformedWeightsError(f"{source} is truncated")
    return data


def _parse_header(raw: bytes, source: str) -> dict[str, object]:
    """Decode the header JSON object strictly."""
    try:
        header = strict_json_loads(raw)
    except NonCanonicalJsonError as exc:
        raise MalformedWeightsError(f"{source} header: {exc}") from exc
    if not isinstance(header, dict):
        raise MalformedWeightsError(f"{source} header must be a JSON object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(value, str) for value in metadata.values()
    ):
        raise MalformedWeightsError(
            f"{source} __metadata__ must map strings to strings"
        )
    if not header:
        raise MalformedWeightsError(f"{source} holds no tensors")
    return header


def _entry(
    name: str, entry: object, data_length: int, source: str
) -> tuple[TensorDtype, tuple[int, ...], int, int]:
    """Validate one tensor entry and return its dtype, shape and slice."""
    if not name:
        raise MalformedWeightsError(f"{source} has a tensor with an empty name")
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        raise MalformedWeightsError(
            f"{source} tensor {name!r} needs exactly {sorted(_ENTRY_KEYS)}"
        )
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    if not isinstance(dtype, str) or dtype not in DTYPE_WIDTHS:
        raise UnsupportedSafetensorsDtypeError(
            f"{source} tensor {name!r} has dtype {dtype!r}"
        )
    if not isinstance(shape, list) or not all(
        type(dim) is int and dim >= 0 for dim in shape
    ):
        raise MalformedWeightsError(
            f"{source} tensor {name!r} shape must be non-negative integers"
        )
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(type(bound) is int for bound in offsets)
        or not 0 <= offsets[0] <= offsets[1] <= data_length
    ):
        raise MalformedWeightsError(
            f"{source} tensor {name!r} has invalid data_offsets"
        )
    start, end = cast(list[int], offsets)
    dims = tuple(cast(list[int], shape))
    expected = DTYPE_WIDTHS[dtype]
    for dim in dims:
        expected *= dim
    if end - start != expected:
        raise MalformedWeightsError(
            f"{source} tensor {name!r} length does not match its shape"
        )
    return cast(TensorDtype, dtype), dims, start, end


def read_safetensors_descriptors(
    handle: BinaryIO,
    *,
    offset: int,
    length: int,
    source: str,
    forbid_negative_zero: bool = False,
) -> tuple[TensorDescriptor, ...]:
    """Parse and hash one safetensors file stored at ``offset`` in ``handle``.

    Parameters
    ----------
    handle : BinaryIO
        An open, seekable binary file.
    offset : int
        Byte offset of the safetensors file inside ``handle``.
    length : int
        Byte length of the safetensors file.
    source : str
        Name used in refusal messages.
    forbid_negative_zero : bool
        Whether a negative zero in a floating-point tensor is refused too.
        NaN and infinities are always refused.

    Returns
    -------
    tuple[TensorDescriptor, ...]
        One descriptor per tensor, in header order.

    Raises
    ------
    MalformedWeightsError
        If any hardened-parser rule is broken.
    UnsupportedSafetensorsDtypeError
        If a tensor's dtype is outside the closed set.
    """
    handle.seek(offset)
    (header_length,) = struct.unpack("<Q", _read_exact(handle, 8, source))
    if not 0 < header_length <= MAX_HEADER_BYTES or 8 + header_length > length:
        raise MalformedWeightsError(f"{source} has an invalid header length")
    header = _parse_header(_read_exact(handle, header_length, source), source)
    data_start, data_length = offset + 8 + header_length, length - 8 - header_length
    slices = [
        (name, *_entry(name, entry, data_length, source))
        for name, entry in header.items()
    ]
    position = 0
    for _, _, _, start, end in sorted(slices, key=lambda item: (item[3], item[4])):
        if start != position:
            raise MalformedWeightsError(
                f"{source} tensor slices overlap or leave a gap"
            )
        position = end
    if position != data_length:
        raise MalformedWeightsError(f"{source} has bytes after its last tensor")
    descriptors = []
    for name, dtype, shape, start, end in slices:
        digest = hashlib.sha256()
        handle.seek(data_start + start)
        remaining = end - start
        while remaining:
            chunk = _read_exact(handle, min(_CHUNK, remaining), source)
            scan_float_chunk(
                dtype,
                chunk,
                forbid_negative_zero=forbid_negative_zero,
                source=f"{source} tensor {name!r}",
            )
            digest.update(chunk)
            remaining -= len(chunk)
        descriptors.append(TensorDescriptor(name, dtype, shape, digest.hexdigest()))
    return tuple(descriptors)


def compute_tensor_set_identity(
    paths: Sequence[str | os.PathLike[str]],
    *,
    forbid_negative_zero: bool = False,
) -> TensorSetIdentity:
    """Return the tensor-set identity of one safetensors file or its shards.

    Parameters
    ----------
    paths : Sequence[str | os.PathLike[str]]
        The model's safetensors files, in any order.
    forbid_negative_zero : bool
        Whether a negative zero in a floating-point tensor is refused too.
        NaN and infinities are always refused.

    Returns
    -------
    TensorSetIdentity
        The identity, equal for one file and any shard split of it.

    Raises
    ------
    MalformedWeightsError
        If no path is given or any file breaks the hardened parser.
    """
    if not paths:
        raise MalformedWeightsError("a tensor set needs at least one safetensors file")
    descriptors: list[TensorDescriptor] = []
    for path in paths:
        with Path(path).open("rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            descriptors.extend(
                read_safetensors_descriptors(
                    handle,
                    offset=0,
                    length=size,
                    source=Path(path).name,
                    forbid_negative_zero=forbid_negative_zero,
                )
            )
    return tensor_set_identity(descriptors)


def compute_tensor_set_digest(
    paths: Sequence[str | os.PathLike[str]], *, forbid_negative_zero: bool = False
) -> str:
    """Return the ``tensor_set_digest`` of safetensors files.

    Parameters
    ----------
    paths : Sequence[str | os.PathLike[str]]
        The model's safetensors files, in any order.
    forbid_negative_zero : bool
        Whether a negative zero in a floating-point tensor is refused too.
        NaN and infinities are always refused.

    Returns
    -------
    str
        The SHA-256 hex digest of the sorted descriptors.
    """
    return compute_tensor_set_identity(
        paths, forbid_negative_zero=forbid_negative_zero
    ).tensor_set_digest
