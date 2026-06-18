# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — threat-intelligence matcher

"""Match prompts and responses against a set of threat indicators.

:class:`ThreatIntelligenceMatcher` holds a registry of
:class:`~director_ai.core.threat_intel.indicators.ThreatIndicator` and, for a
given text, returns every indicator it triggers as a :class:`ThreatMatch` — with
attribution and severity, so a host can block *and* report "matches the APT29
phishing kit". Indicators come from a STIX feed or are authored directly; this is
the detection half, independent of the transport that delivered them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .indicators import IndicatorType, Severity, ThreatIndicator, severity_rank

__all__ = ["ThreatIntelligenceMatcher", "ThreatMatch"]


@dataclass(frozen=True)
class ThreatMatch:
    """One indicator that fired against the inspected text."""

    indicator_id: str
    name: str
    indicator_type: IndicatorType
    attribution: str
    severity: Severity

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a tenant-safe JSON dict (no inspected text)."""
        return {
            "indicator_id": self.indicator_id,
            "name": self.name,
            "indicator_type": str(self.indicator_type),
            "attribution": self.attribution,
            "severity": str(self.severity),
        }


class ThreatIntelligenceMatcher:
    """A registry of threat indicators matched against inspected text."""

    def __init__(self, indicators: Iterable[ThreatIndicator] | None = None) -> None:
        self._indicators: dict[str, ThreatIndicator] = {}
        if indicators is not None:
            self.add_many(indicators)

    def add(self, indicator: ThreatIndicator) -> None:
        """Register one indicator; a duplicate id is rejected."""
        if indicator.id in self._indicators:
            raise ValueError(f"duplicate indicator id: {indicator.id}")
        self._indicators[indicator.id] = indicator

    def add_many(self, indicators: Iterable[ThreatIndicator]) -> None:
        """Register several indicators."""
        for indicator in indicators:
            self.add(indicator)

    @property
    def indicator_count(self) -> int:
        """Number of registered indicators."""
        return len(self._indicators)

    def match(self, text: str) -> list[ThreatMatch]:
        """Return every indicator that fires on ``text``, highest-severity first."""
        hits = [
            ThreatMatch(
                indicator_id=ind.id,
                name=ind.name,
                indicator_type=ind.indicator_type,
                attribution=ind.attribution,
                severity=ind.severity,
            )
            for ind in self._indicators.values()
            if ind.matches(text)
        ]
        hits.sort(key=lambda m: severity_rank(m.severity), reverse=True)
        return hits

    def is_threat(self, text: str) -> bool:
        """Whether any indicator fires on ``text``."""
        return any(ind.matches(text) for ind in self._indicators.values())

    def attributions(self, text: str) -> tuple[str, ...]:
        """Return distinct non-empty attributions of indicators firing on ``text``."""
        seen = {m.attribution for m in self.match(text) if m.attribution}
        return tuple(sorted(seen))
