# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Evidence firewall factory

"""Build a configured :class:`EvidenceFirewall` from a settings object.

The builder reads ``evidence_firewall_*`` attributes off any settings object
(``DirectorConfig`` in production) so the package stays decoupled from the
config module and importable without it. It returns ``None`` when the firewall
is not enabled, which the retrieval store treats as "screen nothing".
"""

from __future__ import annotations

from typing import Protocol

from .firewall import EvidenceFirewall
from .policy import FirewallPolicy

__all__ = ["FirewallSettings", "build_evidence_firewall", "build_firewall_policy"]


class FirewallSettings(Protocol):
    """The ``evidence_firewall_*`` attribute surface the builder reads."""

    evidence_firewall_enabled: bool
    evidence_firewall_require_tenant_match: bool
    evidence_firewall_require_provenance: bool
    evidence_firewall_require_signature: bool
    evidence_firewall_verify_content_hash: bool
    evidence_firewall_enforce_expiry: bool
    evidence_firewall_max_age_seconds: float
    evidence_firewall_require_source_owner: bool
    evidence_firewall_enforce_sensitivity: bool
    evidence_firewall_allowed_sensitivity: tuple[str, ...]
    evidence_firewall_scan_poisoning: bool
    evidence_firewall_poison_threshold: float
    evidence_firewall_enforce_use_case: bool


def build_firewall_policy(settings: FirewallSettings) -> FirewallPolicy:
    """Map ``evidence_firewall_*`` settings onto a :class:`FirewallPolicy`."""
    return FirewallPolicy(
        require_tenant_match=settings.evidence_firewall_require_tenant_match,
        require_provenance=settings.evidence_firewall_require_provenance,
        require_signature=settings.evidence_firewall_require_signature,
        verify_content_hash=settings.evidence_firewall_verify_content_hash,
        enforce_expiry=settings.evidence_firewall_enforce_expiry,
        max_age_seconds=settings.evidence_firewall_max_age_seconds,
        require_source_owner=settings.evidence_firewall_require_source_owner,
        enforce_sensitivity=settings.evidence_firewall_enforce_sensitivity,
        allowed_sensitivity=frozenset(settings.evidence_firewall_allowed_sensitivity),
        scan_poisoning=settings.evidence_firewall_scan_poisoning,
        poison_threshold=settings.evidence_firewall_poison_threshold,
        enforce_use_case=settings.evidence_firewall_enforce_use_case,
    )


def build_evidence_firewall(
    settings: FirewallSettings,
) -> EvidenceFirewall | None:
    """Return a configured firewall, or ``None`` when it is disabled."""
    if not settings.evidence_firewall_enabled:
        return None
    return EvidenceFirewall(build_firewall_policy(settings))
