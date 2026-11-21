# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Tensor Float Scanner

"""Refuse NaN, infinities and, where a family forbids it, negative zeros.

The scan runs on the raw little-endian bytes of each floating-point tensor
while they are hashed, one chunk at a time. Each chunk is read as one large
integer of equal-width lanes, and a lane test on that integer tells whether
any element matches a bit pattern. That keeps the scan in the standard
library and fast enough for large checkpoints.

* NaN and infinity: the exponent field is all ones (``F8_E4M3`` has no
  infinity; its NaN is ``0x7F`` or ``0xFF``).
* Negative zero: the sign bit alone is set.
"""

from __future__ import annotations

from .errors import NegativeZeroTensorError, NonFiniteTensorError

_EXPONENT_MASKS: dict[str, tuple[int, int]] = {
    "F64": (8, 0x7FF0000000000000),
    "F32": (4, 0x7F800000),
    "F16": (2, 0x7C00),
    "BF16": (2, 0x7F80),
    "F8_E5M2": (1, 0x7C),
    "F8_E4M3": (1, 0x7F),
}
FLOAT_DTYPES: frozenset[str] = frozenset(_EXPONENT_MASKS)
"""The floating-point dtypes the scanner checks."""


def _repeat(value: int, width: int, lanes: int) -> int:
    """Return ``value`` repeated in ``lanes`` little-endian lanes of ``width`` bytes."""
    return int.from_bytes(value.to_bytes(width, "little") * lanes, "little")


def any_lane_matches(chunk: bytes, width: int, mask: int, value: int) -> bool:
    """Return whether any ``width``-byte lane satisfies ``lane & mask == value``.

    Parameters
    ----------
    chunk : bytes
        Little-endian lanes; its length is a multiple of ``width``.
    width : int
        Lane width in bytes.
    mask, value : int
        The bit pattern to look for.

    Returns
    -------
    bool
        ``True`` when at least one lane matches.
    """
    lanes = len(chunk) // width
    if not lanes:
        return False
    word = int.from_bytes(chunk, "little")
    zeroed = (word & _repeat(mask, width, lanes)) ^ _repeat(value, width, lanes)
    low = _repeat(1, width, lanes)
    high = _repeat(1 << (8 * width - 1), width, lanes)
    return ((zeroed - low) & ~zeroed & high) != 0


def scan_float_chunk(
    dtype: str, chunk: bytes, *, forbid_negative_zero: bool, source: str
) -> None:
    """Refuse a chunk of a floating-point tensor that holds a forbidden value.

    Parameters
    ----------
    dtype : str
        The tensor's dtype; non-float dtypes are not scanned.
    chunk : bytes
        A whole number of the tensor's elements.
    forbid_negative_zero : bool
        Whether a negative zero is refused too.
    source : str
        Name used in refusal messages.

    Raises
    ------
    NonFiniteTensorError
        If an element is a NaN or an infinity.
    NegativeZeroTensorError
        If ``forbid_negative_zero`` is set and an element is a negative zero.
    """
    if dtype not in _EXPONENT_MASKS:
        return
    width, exponent = _EXPONENT_MASKS[dtype]
    if any_lane_matches(chunk, width, exponent, exponent):
        raise NonFiniteTensorError(
            f"{source} holds a NaN or an infinity in a {dtype} tensor"
        )
    sign = 1 << (8 * width - 1)
    if forbid_negative_zero and any_lane_matches(
        chunk, width, (1 << (8 * width)) - 1, sign
    ):
        raise NegativeZeroTensorError(
            f"{source} holds a negative zero in a {dtype} tensor"
        )
