# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ML bill-of-materials components

"""The components a machine-learning system is built from.

A supply-chain attack swaps a model, dataset, or dependency for a poisoned one.
The defence is provenance: record every component's SHA-256 digest at a known-good
point, then re-verify the deployed artefact against it. :class:`MLBOMComponent`
is one such record — a name, version, type, digest, and supplier — and
:meth:`MLBOMComponent.matches` re-derives the digest of the actual bytes to detect
substitution.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = ["ComponentType", "MLBOMComponent", "compute_sha256"]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ComponentType(StrEnum):
    """The kind of supply-chain component being tracked."""

    MODEL = "model"
    """A model weight artefact (checkpoint, ONNX, safetensors)."""

    DATASET = "dataset"
    """A training/evaluation dataset."""

    DEPENDENCY = "dependency"
    """A software package the system depends on."""

    CODE = "code"
    """A first-party source artefact or build output."""


def compute_sha256(data: bytes) -> str:
    """Return the lower-case hex SHA-256 of ``data``."""
    if not isinstance(data, bytes | bytearray):
        raise TypeError("data to hash must be bytes")
    return hashlib.sha256(bytes(data)).hexdigest()


@dataclass(frozen=True)
class MLBOMComponent:
    """One supply-chain component recorded with its known-good digest."""

    name: str
    version: str
    component_type: ComponentType
    sha256: str
    supplier: str = ""
    source: str = ""
    license: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("component name is required")
        if not self.version.strip():
            raise ValueError("component version is required")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lower-case hex characters")

    def matches(self, data: bytes) -> bool:
        """Whether ``data``'s SHA-256 equals this component's recorded digest.

        A mismatch means the deployed artefact differs from the one recorded at
        provenance time — i.e. substitution or poisoning.
        """
        return compute_sha256(data) == self.sha256

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict (digests + metadata, no payload)."""
        return {
            "name": self.name,
            "version": self.version,
            "component_type": str(self.component_type),
            "sha256": self.sha256,
            "supplier": self.supplier,
            "source": self.source,
            "license": self.license,
        }
