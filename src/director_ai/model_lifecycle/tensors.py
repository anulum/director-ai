# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Tensor Identity

"""Tensor descriptors and the model-wide tensor-set identity.

A tensor set's identity covers each tensor's name, dtype, shape and the
SHA-256 of its raw little-endian bytes. Descriptors are sorted by the UTF-8
bytes of their names, so one file and any split of it into shards give the
same :attr:`TensorSetIdentity.tensor_set_digest`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from ._checks import require_sha256
from .digests import record_digest
from .errors import MalformedWeightsError

TensorDtype = Literal[
    "F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2",
    "I64", "I32", "I16", "I8", "U64", "U32", "U16", "U8", "BOOL",
]  # fmt: skip

DTYPE_WIDTHS: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1,
}  # fmt: skip
"""Bytes per element of every dtype in the closed safetensors set."""

SAFE_TENSORS_DTYPES: tuple[str, ...] = tuple(DTYPE_WIDTHS)
"""The closed dtype set; any other dtype is refused by name."""

TENSOR_SET_IDENTITY_SCHEMA: Literal["tensor-set-identity.v1"] = "tensor-set-identity.v1"


@dataclass(frozen=True, slots=True)
class TensorDescriptor:
    """One tensor's name, dtype, shape and data digest.

    Attributes
    ----------
    name : str
        Tensor name from the safetensors header.
    dtype : TensorDtype
        Element type, one of :data:`SAFE_TENSORS_DTYPES`.
    shape : tuple[int, ...]
        Dimensions; each is a non-negative integer.
    data_sha256 : str
        SHA-256 of the tensor's raw little-endian bytes.
    """

    name: str
    dtype: TensorDtype
    shape: tuple[int, ...]
    data_sha256: str

    @property
    def byte_length(self) -> int:
        """Return the number of data bytes the shape and dtype imply."""
        return math.prod(self.shape) * DTYPE_WIDTHS[self.dtype]


@dataclass(frozen=True, slots=True)
class TensorSetIdentity:
    """The identity of a model's whole tensor set.

    The canonical JSON document uses the top-level key ``schema``, the same
    name as this attribute.

    Attributes
    ----------
    schema : Literal["tensor-set-identity.v1"]
        Record schema.
    tensor_count : int
        Number of tensors across all shards.
    total_byte_length : int
        Sum of the tensors' data lengths.
    tensor_set_digest : str
        SHA-256 of the canonical JSON of the sorted descriptors.
    """

    schema: Literal["tensor-set-identity.v1"]
    tensor_count: int
    total_byte_length: int
    tensor_set_digest: str


def tensor_set_identity(descriptors: Iterable[TensorDescriptor]) -> TensorSetIdentity:
    """Return the identity of a tensor set, whatever its shard order.

    Parameters
    ----------
    descriptors : Iterable[TensorDescriptor]
        Every tensor of the model, from one file or several shards.

    Returns
    -------
    TensorSetIdentity
        Count, total length and the canonical digest.

    Raises
    ------
    MalformedWeightsError
        If the set is empty or two tensors share a name.
    """
    ordered = sorted(descriptors, key=lambda item: item.name.encode("utf-8"))
    if not ordered:
        raise MalformedWeightsError("a tensor set needs at least one tensor")
    names = [item.name for item in ordered]
    if len(set(names)) != len(names):
        raise MalformedWeightsError("tensor names must be unique across shards")
    for item in ordered:
        require_sha256(item.data_sha256, f"tensor {item.name!r} data_sha256")
    document = {
        "schema": TENSOR_SET_IDENTITY_SCHEMA,
        "tensors": [
            {
                "data_sha256": item.data_sha256,
                "dtype": item.dtype,
                "name": item.name,
                "shape": list(item.shape),
            }
            for item in ordered
        ],
    }
    return TensorSetIdentity(
        schema=TENSOR_SET_IDENTITY_SCHEMA,
        tensor_count=len(ordered),
        total_byte_length=sum(item.byte_length for item in ordered),
        tensor_set_digest=record_digest(document),
    )
