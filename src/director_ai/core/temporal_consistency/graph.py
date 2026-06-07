# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Cross-Document Temporal Consistency Graph
"""Track structured claims over time and flag temporal contradictions.

Where :class:`director_ai.core.memory.consistency.CrossDocumentConsistencyMemory`
compares whole documents by text similarity, this graph reasons over *structured*
claims on a timeline: an assertion ``(subject, predicate, value, polarity)`` made
at a point in time, in a session, from a document. It answers the
"diabetes on Monday, no diabetes on Tuesday" class of failure — a system that
flatly contradicts a claim it made earlier, across sessions and documents.

Two contradiction kinds are detected, both tenant-scoped:

* **polarity** — the same ``(subject, predicate, value)`` asserted and then
  negated (or vice versa);
* **functional_value** — a predicate declared single-valued (``functional``) that
  is asserted with two different values (e.g. one diagnosis per patient).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

POLARITY = "polarity"
FUNCTIONAL_VALUE = "functional_value"


def _validate_tenant_id(tenant_id: str) -> str:
    """Return a validated tenant id; empty means the default tenant."""
    if tenant_id and not _SAFE_TENANT_RE.fullmatch(tenant_id):
        raise ValueError(f"invalid tenant_id: {tenant_id!r}")
    return tenant_id


def _require_non_empty(name: str, value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


@dataclass(frozen=True)
class TemporalClaim:
    """A single structured claim asserted at a point in time.

    ``polarity`` is ``True`` for an assertion and ``False`` for a negation, so
    ``(patient, has_condition, diabetes, polarity=False)`` reads as "the patient
    does not have diabetes". ``source_text`` is the raw phrasing and is omitted
    from tenant-safe serialisation.
    """

    subject: str
    predicate: str
    timestamp: float
    value: str = ""
    polarity: bool = True
    tenant_id: str = ""
    session_id: str = ""
    document_id: str = ""
    source_text: str = ""
    claim_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject", _require_non_empty("subject", self.subject))
        object.__setattr__(
            self, "predicate", _require_non_empty("predicate", self.predicate)
        )
        _validate_tenant_id(self.tenant_id)
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

    @property
    def key(self) -> tuple[str, str, str]:
        """The ``(tenant_id, subject, predicate)`` grouping key."""
        return (self.tenant_id, self.subject, self.predicate)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Tenant-safe view; raw ``source_text`` only when ``include_text``."""
        payload: dict[str, Any] = {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "polarity": self.polarity,
            "timestamp": self.timestamp,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "document_id": self.document_id,
            "claim_id": self.claim_id,
        }
        if include_text:
            payload["source_text"] = self.source_text
        return payload


@dataclass(frozen=True)
class TemporalContradiction:
    """A contradiction between two claims, ordered ``earlier`` then ``later``."""

    subject: str
    predicate: str
    kind: str
    earlier: TemporalClaim
    later: TemporalClaim

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Tenant-safe contradiction record with both claim provenances."""
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "kind": self.kind,
            "earlier": self.earlier.to_dict(include_text=include_text),
            "later": self.later.to_dict(include_text=include_text),
        }


def _order(a: TemporalClaim, b: TemporalClaim) -> tuple[TemporalClaim, TemporalClaim]:
    """Return ``(earlier, later)`` by timestamp (stable on ties)."""
    return (a, b) if a.timestamp <= b.timestamp else (b, a)


class TemporalConsistencyGraph:
    """Record structured claims over time and detect temporal contradictions.

    Claims are grouped by ``(tenant_id, subject, predicate)``; every new claim is
    compared against the prior claims in its group, so the comparison cost per
    recorded claim is linear in that group's size.
    """

    def __init__(self, functional_predicates: Iterable[str] = ()) -> None:
        self._functional = {p for p in functional_predicates if p}
        self._claims: dict[tuple[str, str, str], list[TemporalClaim]] = {}
        self._contradictions: list[TemporalContradiction] = []

    @property
    def functional_predicates(self) -> frozenset[str]:
        """Predicates treated as single-valued (one value per subject)."""
        return frozenset(self._functional)

    def _conflict_kind(self, prior: TemporalClaim, claim: TemporalClaim) -> str | None:
        if prior.value == claim.value and prior.polarity != claim.polarity:
            return POLARITY
        if (
            claim.predicate in self._functional
            and prior.polarity
            and claim.polarity
            and prior.value != claim.value
            and prior.value
            and claim.value
        ):
            return FUNCTIONAL_VALUE
        return None

    def record(self, claim: TemporalClaim) -> tuple[TemporalContradiction, ...]:
        """Add ``claim`` and return every contradiction it introduces.

        The claim is compared against each prior claim in the same
        ``(tenant, subject, predicate)`` group; a contradiction is returned (and
        retained) for each conflicting prior claim.
        """
        group = self._claims.setdefault(claim.key, [])
        found: list[TemporalContradiction] = []
        for prior in group:
            kind = self._conflict_kind(prior, claim)
            if kind is None:
                continue
            earlier, later = _order(prior, claim)
            contradiction = TemporalContradiction(
                subject=claim.subject,
                predicate=claim.predicate,
                kind=kind,
                earlier=earlier,
                later=later,
            )
            found.append(contradiction)
        group.append(claim)
        self._contradictions.extend(found)
        return tuple(found)

    def history(
        self, subject: str, predicate: str, *, tenant_id: str = ""
    ) -> tuple[TemporalClaim, ...]:
        """Return the claims for one subject/predicate, oldest first."""
        key = (_validate_tenant_id(tenant_id), subject, predicate)
        return tuple(sorted(self._claims.get(key, ()), key=lambda c: c.timestamp))

    def contradictions(
        self, *, tenant_id: str | None = None
    ) -> tuple[TemporalContradiction, ...]:
        """Return recorded contradictions, optionally filtered by tenant."""
        if tenant_id is None:
            return tuple(self._contradictions)
        tid = _validate_tenant_id(tenant_id)
        return tuple(c for c in self._contradictions if c.later.tenant_id == tid)

    def subjects(self, *, tenant_id: str = "") -> tuple[str, ...]:
        """Return the distinct subjects recorded for a tenant."""
        tid = _validate_tenant_id(tenant_id)
        seen = {key[1] for key in self._claims if key[0] == tid}
        return tuple(sorted(seen))

    def claim_count(self, *, tenant_id: str | None = None) -> int:
        """Return the number of recorded claims, optionally per tenant."""
        if tenant_id is None:
            return sum(len(claims) for claims in self._claims.values())
        tid = _validate_tenant_id(tenant_id)
        return sum(len(claims) for key, claims in self._claims.items() if key[0] == tid)

    def delete_tenant(self, tenant_id: str) -> int:
        """Remove all claims and contradictions for a tenant; return claim count.

        Supports right-to-delete workflows.
        """
        tid = _validate_tenant_id(tenant_id)
        removed = 0
        for key in [k for k in self._claims if k[0] == tid]:
            removed += len(self._claims.pop(key))
        self._contradictions = [
            c for c in self._contradictions if c.later.tenant_id != tid
        ]
        return removed

    def report(self, *, tenant_id: str = "", include_text: bool = False) -> dict:
        """Return a tenant-safe consistency report for a tenant."""
        tid = _validate_tenant_id(tenant_id)
        conflicts = self.contradictions(tenant_id=tid)
        return {
            "tenant_id": tid,
            "subjects": list(self.subjects(tenant_id=tid)),
            "claim_count": self.claim_count(tenant_id=tid),
            "contradiction_count": len(conflicts),
            "contradictions": [c.to_dict(include_text=include_text) for c in conflicts],
            "consistent": not conflicts,
        }
