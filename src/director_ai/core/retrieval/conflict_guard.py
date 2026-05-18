# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — conflict-aware knowledge ingestion

"""Pre-ingestion conflict checks for knowledge-base writes."""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from director_ai.core.retrieval.knowledge import (
    GroundTruthStore,
    _require_non_empty_string,
)

__all__ = [
    "ConflictAwareKnowledgeGuard",
    "KnowledgeConflict",
    "KnowledgeConflictCheck",
    "KnowledgeFact",
]

_BLOCK_DECISIONS = frozenset({"block", "halt"})
_SAFE_ATTRIBUTE_BLOCKLIST = (
    "credential",
    "image",
    "password",
    "private-key",
    "prompt",
    "raw",
    "secret",
    "sensor",
    "token",
)


@dataclass(frozen=True)
class KnowledgeFact:
    """Fact proposed for insertion into a retrieval store."""

    key: str
    value: str
    tenant_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate fact fields and normalise tenant-safe metadata."""
        object.__setattr__(self, "key", _require_non_empty_string("key", self.key))
        object.__setattr__(
            self,
            "value",
            _require_non_empty_string("value", self.value),
        )
        object.__setattr__(self, "tenant_id", str(self.tenant_id).strip())
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class KnowledgeConflict:
    """Tenant-safe conflict report for an incoming KB fact."""

    conflict_type: str
    incoming_key: str
    existing_key: str
    incoming_hash: str
    existing_hash: str
    score: float
    evidence_refs: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        """Validate conflict type, score, and evidence references."""
        if not self.conflict_type.strip():
            raise ValueError("conflict_type is required")
        _validate_unit_interval("score", self.score)
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe report without raw fact text."""
        return {
            "conflict_type": self.conflict_type,
            "incoming_key": self.incoming_key,
            "existing_key": self.existing_key,
            "incoming_hash": self.incoming_hash,
            "existing_hash": self.existing_hash,
            "score": self.score,
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class KnowledgeConflictCheck:
    """Pre-ingestion decision for one proposed fact."""

    decision: str
    incoming_key: str
    tenant_id: str
    incoming_hash: str
    conflicts: tuple[KnowledgeConflict, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the decision and freeze conflict/reference tuples."""
        if self.decision not in {"allow", "warn", "block"}:
            raise ValueError(f"unsupported decision {self.decision!r}")
        object.__setattr__(self, "conflicts", tuple(self.conflicts))
        object.__setattr__(self, "evidence_refs", tuple(map(str, self.evidence_refs)))

    @property
    def blocked(self) -> bool:
        """Return True when the proposed fact must not enter retrieval."""
        return self.decision in _BLOCK_DECISIONS

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe decision payload without raw fact text."""
        return {
            "decision": self.decision,
            "incoming_key": self.incoming_key,
            "tenant_id": self.tenant_id,
            "incoming_hash": self.incoming_hash,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
            "evidence_refs": list(self.evidence_refs),
        }


class ConflictAwareKnowledgeGuard:
    """Validate incoming facts before they are admitted to retrieval."""

    def __init__(
        self,
        store: GroundTruthStore,
        *,
        score_fn: Callable[[str, str], float] | None = None,
        warn_threshold: float = 0.65,
        block_threshold: float = 0.85,
        block_on_same_key_mismatch: bool = True,
        block_on_explicit_contradiction: bool = True,
    ) -> None:
        """Initialise conflict thresholds and store integration policy."""
        self.store = store
        self.score_fn = score_fn
        self.warn_threshold = _validate_unit_interval("warn_threshold", warn_threshold)
        self.block_threshold = _validate_unit_interval(
            "block_threshold",
            block_threshold,
        )
        if self.warn_threshold > self.block_threshold:
            raise ValueError("warn_threshold must be <= block_threshold")
        self.block_on_same_key_mismatch = bool(block_on_same_key_mismatch)
        self.block_on_explicit_contradiction = bool(block_on_explicit_contradiction)

    def check_fact(self, fact: KnowledgeFact) -> KnowledgeConflictCheck:
        """Check whether *fact* can enter the configured store."""
        conflicts: list[KnowledgeConflict] = []
        conflicts.extend(self._same_key_conflicts(fact))
        conflicts.extend(self._explicit_conflicts(fact))
        conflicts.extend(self._semantic_conflicts(fact))
        conflicts = _dedupe_conflicts(conflicts)
        decision = self._decision(conflicts)
        return KnowledgeConflictCheck(
            decision=decision,
            incoming_key=fact.key,
            tenant_id=fact.tenant_id,
            incoming_hash=_content_hash(fact.value),
            conflicts=tuple(conflicts),
            evidence_refs=(f"kb://{fact.key}",),
        )

    def add_fact(self, fact: KnowledgeFact) -> KnowledgeConflictCheck:
        """Check and add *fact* only when the pre-ingestion decision permits it."""
        result = self.check_fact(fact)
        if result.blocked:
            return result
        metadata = _tenant_safe_metadata(fact.metadata)
        self._add_to_store(fact, metadata)
        return result

    def _add_to_store(self, fact: KnowledgeFact, metadata: dict[str, Any]) -> None:
        """Add a permitted fact while preserving store API compatibility."""
        add_fact: Any = self.store.add_fact
        add_fact_params = inspect.signature(add_fact).parameters
        if metadata and "metadata" in add_fact_params:
            add_fact(
                fact.key,
                fact.value,
                tenant_id=fact.tenant_id,
                metadata=metadata,
            )
            return
        if metadata:
            self.store.add(
                fact.key,
                fact.value,
                metadata=metadata,
                tenant_id=fact.tenant_id,
            )
            return
        add_fact(
            fact.key,
            fact.value,
            tenant_id=fact.tenant_id,
        )

    def _same_key_conflicts(self, fact: KnowledgeFact) -> list[KnowledgeConflict]:
        """Return conflicts caused by changing an existing fact key."""
        existing = self._existing_fact_value(fact)
        if existing is None or _normalise_fact(existing) == _normalise_fact(fact.value):
            return []
        return [
            KnowledgeConflict(
                conflict_type="same_key_value_mismatch",
                incoming_key=fact.key,
                existing_key=fact.key,
                incoming_hash=_content_hash(fact.value),
                existing_hash=_content_hash(existing),
                score=1.0,
                evidence_refs=(f"kb://{fact.key}",),
                reason="incoming fact changes an existing retrieval key",
            )
        ]

    def _explicit_conflicts(self, fact: KnowledgeFact) -> list[KnowledgeConflict]:
        """Return conflicts declared by incoming contradiction metadata."""
        targets = _metadata_refs(fact.metadata.get("contradicts", ()))
        if not targets:
            return []
        conflicts = []
        for existing_key, existing_value, refs in self._iter_existing_facts(
            fact.tenant_id
        ):
            if targets.isdisjoint(refs):
                continue
            conflicts.append(
                KnowledgeConflict(
                    conflict_type="explicit_contradiction",
                    incoming_key=fact.key,
                    existing_key=existing_key,
                    incoming_hash=_content_hash(fact.value),
                    existing_hash=_content_hash(existing_value),
                    score=1.0,
                    evidence_refs=(f"kb://{existing_key}",),
                    reason="incoming fact declares a contradiction target",
                )
            )
        return conflicts

    def _semantic_conflicts(self, fact: KnowledgeFact) -> list[KnowledgeConflict]:
        """Return score-function conflicts against existing tenant facts."""
        if self.score_fn is None:
            return []
        conflicts = []
        for existing_key, existing_value, _refs in self._iter_existing_facts(
            fact.tenant_id
        ):
            if existing_key == fact.key:
                continue
            score = _validate_unit_interval(
                "score",
                float(self.score_fn(existing_value, fact.value)),
            )
            if score < self.warn_threshold:
                continue
            conflicts.append(
                KnowledgeConflict(
                    conflict_type="semantic_contradiction",
                    incoming_key=fact.key,
                    existing_key=existing_key,
                    incoming_hash=_content_hash(fact.value),
                    existing_hash=_content_hash(existing_value),
                    score=score,
                    evidence_refs=(f"kb://{existing_key}",),
                    reason="incoming fact conflicts with existing retrieval state",
                )
            )
        return conflicts

    def _decision(self, conflicts: list[KnowledgeConflict]) -> str:
        """Return allow, warn, or block for collected conflicts."""
        if not conflicts:
            return "allow"
        for conflict in conflicts:
            if (
                conflict.conflict_type == "same_key_value_mismatch"
                and self.block_on_same_key_mismatch
            ):
                return "block"
            if (
                conflict.conflict_type == "explicit_contradiction"
                and self.block_on_explicit_contradiction
            ):
                return "block"
            if conflict.score >= self.block_threshold:
                return "block"
        return "warn"

    def _existing_fact_value(self, fact: KnowledgeFact) -> str | None:
        """Return an existing same-key fact value from simple stores."""
        key = f"{fact.tenant_id}:{fact.key}" if fact.tenant_id else fact.key
        value = getattr(self.store, "facts", {}).get(key)
        return str(value) if value is not None else None

    def _iter_existing_facts(self, tenant_id: str) -> list[tuple[str, str, set[str]]]:
        """Return tenant facts from either versioned or plain stores."""
        if hasattr(self.store, "version_manifest"):
            return self._iter_version_manifest(tenant_id)
        return self._iter_plain_facts(tenant_id)

    def _iter_plain_facts(self, tenant_id: str) -> list[tuple[str, str, set[str]]]:
        """Return tenant facts from an in-memory facts mapping."""
        rows = []
        prefix = f"{tenant_id}:" if tenant_id else ""
        for stored_key, value in getattr(self.store, "facts", {}).items():
            key = str(stored_key)
            if tenant_id:
                if not key.startswith(prefix):
                    continue
                key = key[len(prefix) :]
            rows.append((key, str(value), {key}))
        return rows

    def _iter_version_manifest(self, tenant_id: str) -> list[tuple[str, str, set[str]]]:
        """Return tenant facts from a version-manifest capable store."""
        manifest = self.store.version_manifest(tenant_id)  # type: ignore[attr-defined]
        rows = []
        for record in manifest.values():
            key = str(record.get("key", ""))
            value = self._value_for_record(key, tenant_id)
            rows.append((key, value, _record_refs(key, record)))
        return rows

    def _value_for_record(self, key: str, tenant_id: str) -> str:
        """Return fact text or content hash for a versioned record."""
        fact_key = f"{tenant_id}:{key}" if tenant_id else key
        value = getattr(self.store, "facts", {}).get(fact_key)
        if value is not None:
            return str(value)
        record = self.store.fact_version_record(key, tenant_id)  # type: ignore[attr-defined]
        if record is None:
            return ""
        return str(record.get("content_hash", ""))


def _content_hash(value: str) -> str:
    """Return a stable SHA-256 fingerprint for fact values."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_fact(value: str) -> str:
    """Return canonical fact text for same-key comparison."""
    return " ".join(value.casefold().split())


def _validate_unit_interval(name: str, value: float) -> float:
    """Return a finite unit-interval value."""
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _metadata_refs(value: Any) -> set[str]:
    """Return normalized metadata reference identifiers."""
    if value in (None, ""):
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    if isinstance(value, list | tuple | set):
        return {str(item).strip() for item in value if str(item).strip()}
    return {str(value).strip()}


def _record_refs(key: str, record: Mapping[str, Any]) -> set[str]:
    """Return identifiers that can be used as contradiction targets."""
    refs = {key}
    for ref_field in (
        "source_id",
        "external_id",
        "claim_id",
        "signed_fact_id",
        "passport_claim_id",
    ):
        value = str(record.get(ref_field, "")).strip()
        if value:
            refs.add(value)
    return refs - {""}


def _tenant_safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata with sensitive attribute names removed."""
    safe = {}
    for key, value in metadata.items():
        key_s = str(key)
        lowered = key_s.lower().replace("_", "-")
        if any(part in lowered for part in _SAFE_ATTRIBUTE_BLOCKLIST):
            continue
        safe[key_s] = value
    return safe


def _dedupe_conflicts(conflicts: list[KnowledgeConflict]) -> list[KnowledgeConflict]:
    """Return conflicts with duplicate evidence markers removed."""
    seen = set()
    deduped = []
    for conflict in conflicts:
        marker = (
            conflict.conflict_type,
            conflict.incoming_key,
            conflict.existing_key,
            conflict.incoming_hash,
            conflict.existing_hash,
        )
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(conflict)
    return deduped
