# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Version Ledger
"""Version-ledger surface of the vector ground-truth store.

:class:`VersionLedgerMixin` keeps the tenant-scoped semantic-version
records for every fact and derived chunk written through
:class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`:
content-hash based bump decisions, the retraction and replacement event
ledgers, and the citation-freshness signals derived from them. All state
is initialised by the composing store's ``__init__``.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from ..knowledge import _require_non_empty_string

__all__ = ["VersionLedgerMixin"]


def _require_string(field_name: str, value: str) -> str:
    """Validate that a field is a string without normalising it."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def _require_numeric_timestamp(field_name: str, value: str) -> float:
    """Validate and parse a timestamp-like metadata field."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a numeric timestamp; got {value!r}",
        ) from exc


class VersionLedgerMixin:
    """Semantic-version bookkeeping for facts and derived chunks.

    All state is initialised by the composing
    :class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`'s
    ``__init__``; tenant resolution, the write path, and the keyword fact
    map come from the composing store through the contracts declared below.
    """

    _version_records: dict[str, dict[str, str]]
    _retraction_records: list[dict[str, str]]
    _replacement_records: list[dict[str, str]]

    if TYPE_CHECKING:
        # Provided by the composing VectorGroundTruthStore / GroundTruthStore.
        facts: dict[str, str]

        def _resolved_tenant_id(self, tenant_id: str = "") -> str: ...

        def add(
            self,
            key: str,
            value: str,
            metadata: dict[str, Any] | None = None,
            tenant_id: str = "",
        ) -> None: ...

        def _bump_revision(self) -> None: ...

    def fact_version(self, key: str, tenant_id: str = "") -> str | None:
        """Return the semantic version currently recorded for *key*."""
        key = _require_non_empty_string("key", key)
        tenant_id = self._resolved_tenant_id(tenant_id)
        record = self._version_records.get(self._version_key(key, tenant_id))
        return record["version"] if record else None

    def fact_version_record(
        self,
        key: str,
        tenant_id: str = "",
    ) -> dict[str, str] | None:
        """Return version metadata for *key* without exposing mutable state."""
        key = _require_non_empty_string("key", key)
        tenant_id = self._resolved_tenant_id(tenant_id)
        record = self._version_records.get(self._version_key(key, tenant_id))
        return dict(record) if record else None

    def version_manifest(self, tenant_id: str = "") -> dict[str, dict[str, str]]:
        """Return version records visible to *tenant_id*."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        manifest: dict[str, dict[str, str]] = {}
        for key, record in self._version_records.items():
            if tenant_id and record.get("tenant_id", "") != tenant_id:
                continue
            manifest[key] = dict(record)
        return manifest

    def retraction_records(self, tenant_id: str = "") -> list[dict[str, str]]:
        """Return fact and chunk retraction events visible to *tenant_id*."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        return [
            dict(record)
            for record in self._retraction_records
            if not tenant_id or record.get("tenant_id", "") == tenant_id
        ]

    def replacement_records(self, tenant_id: str = "") -> list[dict[str, str]]:
        """Return fact replacement events visible to *tenant_id*."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        return [
            dict(record)
            for record in self._replacement_records
            if not tenant_id or record.get("tenant_id", "") == tenant_id
        ]

    def freshness_status_signals(
        self,
        tenant_id: str = "",
        key: str | None = None,
    ) -> list[dict[str, str | float]]:
        """Return citation status signals for temporal freshness scoring."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        if key is not None:
            key = _require_non_empty_string("key", key)
        signals: list[dict[str, str | float]] = []
        for record in self.version_manifest(tenant_id).values():
            if key is not None and record.get("key") != key:
                continue
            status = record.get("citation_status", "")
            published_at = record.get("source_timestamp", "")
            updated_at = record.get("updated_timestamp", "")
            if not status and not published_at and not updated_at:
                continue
            signal: dict[str, str | float] = {
                "source_id": record.get("external_id")
                or record.get("source_id")
                or record.get("key", ""),
                "status": status or "active",
                "status_source": record.get("status_source", ""),
            }
            if published_at:
                signal["published_at"] = _require_numeric_timestamp(
                    "source_timestamp",
                    published_at,
                )
            if updated_at:
                signal["updated_at"] = _require_numeric_timestamp(
                    "updated_timestamp",
                    updated_at,
                )
            observed_at = record.get("status_observed_at", "")
            if observed_at:
                signal["observed_at"] = _require_numeric_timestamp(
                    "status_observed_at",
                    observed_at,
                )
            signals.append(signal)
        return signals

    def retract_fact(
        self,
        key: str,
        *,
        tenant_id: str = "",
        reason: str = "",
    ) -> dict[str, str]:
        """Mark a fact or derived chunk source as unusable for retrieval."""
        key = _require_non_empty_string("key", key)
        reason = _require_string("reason", reason)
        tenant_id = self._resolved_tenant_id(tenant_id)
        version_key = self._version_key(key, tenant_id)
        record = self._version_records.get(version_key)
        if record is None:
            raise KeyError(f"cannot retract unknown fact {key!r}")

        event = {
            "key": key,
            "tenant_id": tenant_id,
            "version": record["version"],
            "content_hash": record["content_hash"],
            "reason": reason,
            "event": "retracted",
            "source_id": record.get("source_id", ""),
            "external_id": record.get("external_id", ""),
            "claim_id": record.get("claim_id", ""),
            "signed_fact_id": record.get("signed_fact_id", ""),
            "passport_claim_id": record.get("passport_claim_id", ""),
        }
        record["status"] = "retracted"
        record["retraction_reason"] = reason
        self._retraction_records.append(dict(event))
        fact_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts.pop(fact_key, None)
        self._bump_revision()
        return event

    def replace_fact(
        self,
        key: str,
        value: str,
        *,
        tenant_id: str = "",
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Add a replacement value and record the superseded hash."""
        key = _require_non_empty_string("key", key)
        reason = _require_string("reason", reason)
        tenant_id = self._resolved_tenant_id(tenant_id)
        version_key = self._version_key(key, tenant_id)
        previous = self._version_records.get(version_key)
        if previous is None:
            raise KeyError(f"cannot replace unknown fact {key!r}")

        self.add(key, value, metadata=metadata, tenant_id=tenant_id)
        current = self._version_records[version_key]
        event = {
            "key": key,
            "tenant_id": tenant_id,
            "from_version": previous["version"],
            "to_version": current["version"],
            "from_hash": previous["content_hash"],
            "to_hash": current["content_hash"],
            "reason": reason,
            "event": "replaced",
        }
        self._replacement_records.append(dict(event))
        current["replacement_reason"] = reason
        fact_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts[fact_key] = value
        return event

    def _build_version_metadata(
        self,
        *,
        key: str,
        value: str,
        tenant_id: str,
        record_kind: str,
        requested_bump: str,
        chunk_index: int,
    ) -> dict[str, str]:
        """Build semantic-version metadata for a fact or chunk write."""
        version_key = self._version_key(key, tenant_id)
        previous = self._version_records.get(version_key)
        content_hash = self._content_hash(value)
        if previous is None:
            version = "1.0.0"
            previous_hash = ""
        elif previous["content_hash"] == content_hash:
            version = previous["version"]
            previous_hash = previous.get("previous_hash", "")
        else:
            version = self._next_semver(previous["version"], requested_bump)
            previous_hash = previous["content_hash"]

        return {
            "kb_version": version,
            "kb_chunk_version": version,
            "kb_content_hash": content_hash,
            "kb_previous_hash": previous_hash,
            "kb_record_kind": record_kind,
            "kb_source_key": key,
            "kb_chunk_index": str(chunk_index),
        }

    def _commit_version_metadata(
        self,
        key: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> None:
        """Persist immutable-facing version metadata for a stored record."""
        version_key = self._version_key(key, tenant_id)
        self._version_records[version_key] = {
            "key": key,
            "tenant_id": tenant_id,
            "version": str(metadata["kb_version"]),
            "chunk_version": str(metadata["kb_chunk_version"]),
            "content_hash": str(metadata["kb_content_hash"]),
            "previous_hash": str(metadata["kb_previous_hash"]),
            "record_kind": str(metadata["kb_record_kind"]),
            "chunk_index": str(metadata["kb_chunk_index"]),
            "source_id": str(metadata.get("source_id", "")),
            "external_id": str(metadata.get("external_id", "")),
            "source_timestamp": str(
                metadata.get(
                    "kb_source_timestamp", metadata.get("source_timestamp", "")
                )
            ),
            "updated_timestamp": str(
                metadata.get(
                    "kb_updated_timestamp",
                    metadata.get("updated_timestamp", ""),
                )
            ),
            "citation_status": str(
                metadata.get("kb_citation_status", metadata.get("citation_status", ""))
            ),
            "status_source": str(
                metadata.get("kb_status_source", metadata.get("status_source", ""))
            ),
            "status_observed_at": str(
                metadata.get(
                    "kb_status_observed_at",
                    metadata.get("status_observed_at", ""),
                )
            ),
            "claim_id": str(metadata.get("kb_claim_id", metadata.get("claim_id", ""))),
            "claim_source": str(
                metadata.get("kb_claim_source", metadata.get("claim_source", ""))
            ),
            "signed_fact_id": str(
                metadata.get("kb_signed_fact_id", metadata.get("signed_fact_id", ""))
            ),
            "passport_claim_id": str(
                metadata.get(
                    "kb_passport_claim_id",
                    metadata.get("passport_claim_id", ""),
                )
            ),
        }

    @staticmethod
    def _normalised_claim_metadata(metadata: dict[str, Any]) -> dict[str, str]:
        """Return stable claim reference fields from mixed metadata keys."""
        return {
            "claim_id": str(metadata.get("kb_claim_id", metadata.get("claim_id", ""))),
            "claim_source": str(
                metadata.get("kb_claim_source", metadata.get("claim_source", ""))
            ),
            "signed_fact_id": str(
                metadata.get("kb_signed_fact_id", metadata.get("signed_fact_id", ""))
            ),
            "passport_claim_id": str(
                metadata.get(
                    "kb_passport_claim_id",
                    metadata.get("passport_claim_id", ""),
                )
            ),
        }

    @staticmethod
    def _version_key(key: str, tenant_id: str) -> str:
        """Return the tenant-qualified key used for version bookkeeping."""
        return f"{tenant_id}::{key}" if tenant_id else key

    @staticmethod
    def _content_hash(value: str) -> str:
        """Return a stable SHA-256 hash for fact content."""
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _next_semver(current: str, requested_bump: str) -> str:
        """Return the next semantic version for a requested bump type."""
        parts = current.split(".")
        if len(parts) != 3:
            raise ValueError(f"invalid semantic version {current!r}")
        major, minor, patch = (int(part) for part in parts)
        match requested_bump:
            case "major":
                return f"{major + 1}.0.0"
            case "minor":
                return f"{major}.{minor + 1}.0"
            case "patch":
                return f"{major}.{minor}.{patch + 1}"
            case _:
                raise ValueError("kb_version_bump must be major, minor, or patch")
