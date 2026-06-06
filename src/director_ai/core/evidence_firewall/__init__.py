# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence firewall package

"""Pre-model evidence firewall.

Screens every retrieved chunk against eight admission checks — tenant
authorisation, provenance presence, signature verification, content-hash
integrity, expiry/age freshness, source owner, sensitivity label, allowed use
case, and indirect-prompt-injection poisoning — before the chunk reaches the
model. Failing chunks are quarantined with a stable, tenant-safe reason code
instead of silently shaping the answer.
"""

from __future__ import annotations

from .chunk import RetrievedChunk
from .factory import build_evidence_firewall, build_firewall_policy
from .firewall import ChunkVerdict, EvidenceFirewall, FirewallReport
from .poison import PoisonScanner, default_poison_scan
from .policy import CheckOutcome, FirewallContext, FirewallPolicy

__all__ = [
    "CheckOutcome",
    "ChunkVerdict",
    "EvidenceFirewall",
    "FirewallContext",
    "FirewallPolicy",
    "FirewallReport",
    "PoisonScanner",
    "RetrievedChunk",
    "build_evidence_firewall",
    "build_firewall_policy",
    "default_poison_scan",
]
