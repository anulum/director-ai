# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — threat-intelligence integration

"""Threat-intelligence matching with attribution (STIX-aligned IOCs).

Check prompts and responses against known indicators of compromise and report not
just a block but the attribution — "matches the APT29 phishing kit".
:class:`ThreatIndicator` is the STIX-aligned IOC, :func:`from_stix_bundle` imports
indicators from a STIX 2.1 feed (TAXII is the transport that delivers the bundle;
that network client is out of scope), and :class:`ThreatIntelligenceMatcher`
returns every indicator a text triggers, highest severity first.
"""

from .indicators import IndicatorType, Severity, ThreatIndicator
from .matcher import ThreatIntelligenceMatcher, ThreatMatch
from .stix import from_stix_bundle

__all__ = [
    "IndicatorType",
    "Severity",
    "ThreatIndicator",
    "ThreatIntelligenceMatcher",
    "ThreatMatch",
    "from_stix_bundle",
]
