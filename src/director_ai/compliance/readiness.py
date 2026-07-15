# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SOC 2 / ISO 27001 / HIPAA readiness

"""Tenant-safe compliance readiness reporting.

The module builds operator-reviewable evidence packets for customer security
reviews. It deliberately records evidence references and deployment obligations
without serialising raw audit rows, customer text, PHI, secrets, or screenshots.
The reports are readiness aids only; they are not certifications, legal advice,
auditor opinions, or proof that a deployment satisfies a regulation.

The implementation is split by deliverable — the shared status vocabulary lives
in :mod:`._readiness_base`, the SOC 2 / ISO report in :mod:`._readiness_soc2_iso`,
and the HIPAA documentation packet in :mod:`._readiness_hipaa`. This module is the
stable public surface and re-exports each deliverable so existing imports of
``director_ai.compliance.readiness`` keep working unchanged.
"""

from __future__ import annotations

from ._readiness_base import ReadinessStatus as ReadinessStatus
from ._readiness_base import _risk_level as _risk_level
from ._readiness_hipaa import HipaaDeploymentObligation as HipaaDeploymentObligation
from ._readiness_hipaa import HipaaDocumentationPacket as HipaaDocumentationPacket
from ._readiness_hipaa import (
    build_hipaa_documentation_packet as build_hipaa_documentation_packet,
)
from ._readiness_hipaa import default_hipaa_obligations as default_hipaa_obligations
from ._readiness_soc2_iso import Soc2IsoControl as Soc2IsoControl
from ._readiness_soc2_iso import Soc2IsoReadinessReport as Soc2IsoReadinessReport
from ._readiness_soc2_iso import (
    build_soc2_iso_readiness_report as build_soc2_iso_readiness_report,
)
from ._readiness_soc2_iso import (
    default_readiness_controls as default_readiness_controls,
)

__all__ = [
    "HipaaDeploymentObligation",
    "HipaaDocumentationPacket",
    "ReadinessStatus",
    "Soc2IsoControl",
    "Soc2IsoReadinessReport",
    "build_hipaa_documentation_packet",
    "build_soc2_iso_readiness_report",
    "default_hipaa_obligations",
    "default_readiness_controls",
]
