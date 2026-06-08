# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Machine-learning bill of materials

"""Assemble and verify a machine-learning bill of materials (ML-BOM).

A :class:`MachineLearningBOM` is the recorded supply chain of a deployed system —
its models, datasets, and dependencies, each pinned to a SHA-256 digest. The BOM
itself carries a digest over its components, so the inventory is tamper-evident;
:meth:`verify` re-derives the digest of each supplied artefact and reports which
components are intact, substituted (poisoned), or unverified.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .components import ComponentType, MLBOMComponent, compute_sha256

__all__ = ["MachineLearningBOM", "VerificationReport"]


@dataclass(frozen=True)
class VerificationReport:
    """The outcome of verifying deployed artefacts against the BOM."""

    intact: tuple[str, ...] = field(default_factory=tuple)
    tampered: tuple[str, ...] = field(default_factory=tuple)
    unverified: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """True when nothing was found tampered (unverified is not a failure)."""
        return not self.tampered

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a tenant-safe JSON dict (component names only)."""
        return {
            "ok": self.ok,
            "intact": list(self.intact),
            "tampered": list(self.tampered),
            "unverified": list(self.unverified),
        }


class MachineLearningBOM:
    """A tamper-evident inventory of a system's ML supply chain."""

    def __init__(self) -> None:
        self._components: dict[str, MLBOMComponent] = {}

    @property
    def components(self) -> tuple[MLBOMComponent, ...]:
        """The recorded components, ordered by name."""
        return tuple(self._components[name] for name in sorted(self._components))

    def add(self, component: MLBOMComponent) -> None:
        """Record one component; a duplicate name is rejected."""
        if component.name in self._components:
            raise ValueError(f"component already recorded: {component.name}")
        self._components[component.name] = component

    def add_artifact(
        self,
        name: str,
        version: str,
        component_type: ComponentType,
        data: bytes,
        **metadata: str,
    ) -> MLBOMComponent:
        """Record a component, digesting ``data`` to pin its SHA-256."""
        component = MLBOMComponent(
            name=name,
            version=version,
            component_type=component_type,
            sha256=compute_sha256(data),
            **metadata,
        )
        self.add(component)
        return component

    @property
    def bom_digest(self) -> str:
        """A SHA-256 over the canonical component list — the BOM's fingerprint.

        Any change to a recorded component (or the set of components) changes this
        digest, so a trusted copy of it makes the whole inventory tamper-evident.
        """
        canonical = json.dumps(
            [c.to_dict() for c in self.components],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify(self, actuals: Mapping[str, bytes]) -> VerificationReport:
        """Re-derive each supplied artefact's digest and classify it.

        ``actuals`` maps component name → the deployed bytes. A component whose
        bytes are supplied and match is *intact*; one whose bytes are supplied and
        differ is *tampered* (poisoned/substituted); a recorded component with no
        supplied bytes is *unverified*. An unknown name in ``actuals`` is reported
        as tampered — it is not part of the trusted inventory.
        """
        intact: list[str] = []
        tampered: list[str] = []
        for name, component in sorted(self._components.items()):
            if name not in actuals:
                continue
            if component.matches(actuals[name]):
                intact.append(name)
            else:
                tampered.append(name)
        unverified = sorted(set(self._components) - set(actuals))
        unknown = sorted(set(actuals) - set(self._components))
        tampered.extend(unknown)
        return VerificationReport(
            intact=tuple(intact),
            tampered=tuple(sorted(tampered)),
            unverified=tuple(unverified),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full BOM (components + its digest), tenant-safe."""
        return {
            "bom_digest": self.bom_digest,
            "components": [c.to_dict() for c in self.components],
        }
