# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — studio federation surface (schema-A capability manifest)

"""Federation surface for the SCPN-STUDIO dedicated-portal contract.

Exposes the schema-A capability manifest — the federation-gate artifact the
Institute hub and the Director-AI Tier-B portal consume to discover the studio's
verbs, evidence types, and federated UI panel. See
:mod:`director_ai.federation.manifest`.
"""

from .manifest import (
    StudioManifest,
    UiModule,
    Verb,
    build_manifest,
)

__all__ = [
    "StudioManifest",
    "UiModule",
    "Verb",
    "build_manifest",
]
