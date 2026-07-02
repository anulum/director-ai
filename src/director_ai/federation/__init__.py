# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — studio federation surface (schema-A + architecture map)

"""Federation surface for the SCPN-STUDIO dedicated-portal contract.

Exposes the schema-A capability manifest and the additive architecture-map.v2
extension the Institute hub and Director-AI Tier-B portal consume to discover the
studio's verbs, evidence types, federated UI panel, runtime topology, and
cross-repo adapter boundaries.
"""

from .architecture_map import (
    ARCHITECTURE_MAP_VERSION,
    build_architecture_map_extension,
    build_federation_document,
)
from .manifest import (
    StudioManifest,
    UiModule,
    Verb,
    build_manifest,
)

__all__ = [
    "ARCHITECTURE_MAP_VERSION",
    "StudioManifest",
    "UiModule",
    "Verb",
    "build_architecture_map_extension",
    "build_federation_document",
    "build_manifest",
]
