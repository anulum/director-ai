# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — STIX 2.1 bundle ingestion

"""Import threat indicators from a STIX 2.1 bundle.

STIX is the standard format threat-intel feeds publish (TAXII is the transport
that delivers a bundle; supplying the bundle to this importer is the seam — the
network client is out of scope here). :func:`from_stix_bundle` reads the bundle's
``indicator`` objects, resolves attribution from ``intrusion-set`` / ``threat-actor``
objects via ``relationship`` objects, and maps each STIX pattern comparison to a
:class:`~director_ai.core.threat_intel.indicators.ThreatIndicator`.

**Supported pattern subset (documented, not full STIX patterning):** simple
comparison terms of the form ``[path OP 'value']``. ``=`` becomes a SUBSTRING (or
SHA256 when the path is a SHA-256 hash); ``MATCHES`` becomes a REGEX. Compound
patterns (multiple comparisons joined by AND/OR) are decomposed into one indicator
per comparison. Terms this subset cannot represent are skipped, never guessed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .indicators import IndicatorType, Severity, ThreatIndicator

__all__ = ["from_stix_bundle"]

# One comparison term inside a STIX pattern: [ path (= | MATCHES | LIKE) 'value' ].
_COMPARISON_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_:.'\-]+?)\s*"
    r"(?P<op>=|MATCHES|LIKE)\s*"
    r"'(?P<value>(?:[^'\\]|\\.)*)'"
)
_SHA256_PATH_RE = re.compile(r"sha-?256", re.IGNORECASE)


def _severity_of(obj: Mapping[str, Any]) -> Severity:
    raw = str(obj.get("x_severity", obj.get("severity", "medium"))).lower()
    try:
        return Severity(raw)
    except ValueError:
        return Severity.MEDIUM


def _indicator_type(path: str, op: str) -> IndicatorType:
    if op == "MATCHES":
        return IndicatorType.REGEX
    if _SHA256_PATH_RE.search(path):
        return IndicatorType.SHA256
    return IndicatorType.SUBSTRING


def _parse_pattern(pattern: str) -> list[tuple[IndicatorType, str]]:
    """Decompose a STIX pattern into (type, value) per supported comparison."""
    terms: list[tuple[IndicatorType, str]] = []
    for comparison in _COMPARISON_RE.finditer(pattern):
        value = comparison.group("value").replace("\\'", "'").replace("\\\\", "\\")
        if not value:
            continue
        terms.append(
            (_indicator_type(comparison.group("path"), comparison.group("op")), value)
        )
    return terms


def _attribution_map(objects: list[Mapping[str, Any]]) -> dict[str, str]:
    """Map indicator id → attributed actor/intrusion-set name via relationships."""
    names: dict[str, str] = {
        obj["id"]: str(obj.get("name", obj["id"]))
        for obj in objects
        if obj.get("type") in {"intrusion-set", "threat-actor", "malware", "campaign"}
        and "id" in obj
    }
    attribution: dict[str, str] = {}
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") not in {"indicates", "attributed-to"}:
            continue
        source = obj.get("source_ref", "")
        target = obj.get("target_ref", "")
        if source.startswith("indicator--") and target in names:
            attribution[source] = names[target]
    return attribution


def from_stix_bundle(bundle: Mapping[str, Any]) -> list[ThreatIndicator]:
    """Parse a STIX 2.1 bundle into threat indicators.

    Unsupported or unparsable indicator patterns are skipped, not approximated.
    """
    objects = [o for o in bundle.get("objects", []) if isinstance(o, Mapping)]
    attribution = _attribution_map(objects)

    indicators: list[ThreatIndicator] = []
    for obj in objects:
        if obj.get("type") != "indicator":
            continue
        stix_id = str(obj.get("id", "")).strip()
        if not stix_id:
            continue
        pattern = str(obj.get("pattern", ""))
        terms = _parse_pattern(pattern)
        if not terms:
            continue
        name = str(obj.get("name", stix_id))
        severity = _severity_of(obj)
        labels = tuple(str(label) for label in obj.get("labels", []))
        actor = attribution.get(stix_id, "")
        for index, (indicator_type, value) in enumerate(terms):
            suffix = "" if len(terms) == 1 else f"#{index}"
            indicators.append(
                ThreatIndicator(
                    id=f"{stix_id}{suffix}",
                    name=name,
                    indicator_type=indicator_type,
                    pattern=value,
                    attribution=actor,
                    severity=severity,
                    labels=labels,
                )
            )
    return indicators
