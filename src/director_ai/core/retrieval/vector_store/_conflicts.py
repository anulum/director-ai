# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Conflict Ledger
"""Conflict-ledger surface of the vector ground-truth store.

:class:`ConflictLedgerMixin` detects and records knowledge-base
conflicts for every fact written through
:class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`:
overlaps with retracted ledger entries, divergence from signed or
passport-backed claim state, and explicitly declared contradiction
targets, deduplicated into tenant-safe reports without raw fact
payloads. All state is initialised by the composing store's ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..knowledge import _require_non_empty_string

__all__ = ["ConflictLedgerMixin"]


class ConflictLedgerMixin:
    """Tenant-safe conflict detection and reporting for fact writes.

    All state is initialised by the composing
    :class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`'s
    ``__init__``; tenant resolution and the version/retraction ledgers come
    from the composing store through the contracts declared below.
    """

    _conflict_records: list[dict[str, str]]

    if TYPE_CHECKING:
        # Provided by the composing VectorGroundTruthStore / VersionLedgerMixin.
        def _resolved_tenant_id(self, tenant_id: str = "") -> str: ...

        def retraction_records(self, tenant_id: str = "") -> list[dict[str, str]]: ...

        def version_manifest(
            self, tenant_id: str = ""
        ) -> dict[str, dict[str, str]]: ...

    def conflict_reports(
        self,
        tenant_id: str = "",
        key: str | None = None,
    ) -> list[dict[str, str]]:
        """Return KB conflict reports visible to *tenant_id*."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        if key is not None:
            key = _require_non_empty_string("key", key)
        reports = []
        for record in self._conflict_records:
            if tenant_id and record.get("tenant_id", "") != tenant_id:
                continue
            if key is not None and record.get("key") != key:
                continue
            reports.append(dict(record))
        return reports

    def _build_conflict_reports(
        self,
        key: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> list[dict[str, str]]:
        """Build tenant-safe conflict reports for a pending fact write."""
        reports: list[dict[str, str]] = []
        reports.extend(self._retraction_conflicts(key, metadata, tenant_id))
        reports.extend(self._protected_claim_conflicts(key, metadata, tenant_id))
        reports.extend(self._explicit_conflicts(key, metadata, tenant_id))
        return self._dedupe_conflicts(reports)

    def _retraction_conflicts(
        self,
        key: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> list[dict[str, str]]:
        """Return conflicts between an incoming fact and retraction records."""
        incoming_refs = self._record_refs(key, metadata)
        reports = []
        for event in self.retraction_records(tenant_id):
            event_refs = self._record_refs(event.get("key", ""), event)
            if incoming_refs.isdisjoint(event_refs):
                continue
            reports.append(
                self._conflict_record(
                    key=key,
                    tenant_id=tenant_id,
                    conflict_type="retraction_record",
                    existing=event,
                    incoming=metadata,
                    reason="new fact overlaps a retracted ledger entry",
                )
            )
        return reports

    def _protected_claim_conflicts(
        self,
        key: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> list[dict[str, str]]:
        """Return conflicts against signed or passport-backed claim state."""
        incoming_hash = str(metadata["kb_content_hash"])
        incoming_refs = self._record_refs(key, metadata)
        reports = []
        for record in self.version_manifest(tenant_id).values():
            if record.get("content_hash") == incoming_hash:
                continue
            existing_refs = self._record_refs(record.get("key", ""), record)
            shared_refs = incoming_refs & existing_refs
            if not shared_refs:
                continue
            conflict_type = self._protected_conflict_type(record, metadata)
            if conflict_type == "":
                continue
            reports.append(
                self._conflict_record(
                    key=key,
                    tenant_id=tenant_id,
                    conflict_type=conflict_type,
                    existing=record,
                    incoming=metadata,
                    reason="new fact differs from protected claim state",
                )
            )
        return reports

    def _explicit_conflicts(
        self,
        key: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> list[dict[str, str]]:
        """Return conflicts declared through explicit metadata references."""
        targets = self._metadata_list(metadata.get("contradicts", ()))
        if not targets:
            return []
        reports = []
        for record in self.version_manifest(tenant_id).values():
            record_refs = self._record_refs(record.get("key", ""), record)
            if record_refs.isdisjoint(targets):
                continue
            reports.append(
                self._conflict_record(
                    key=key,
                    tenant_id=tenant_id,
                    conflict_type=self._protected_conflict_type(record, metadata)
                    or "explicit_relation",
                    existing=record,
                    incoming=metadata,
                    reason="new fact declares a contradiction target",
                )
            )
        return reports

    @staticmethod
    def _protected_conflict_type(
        existing: dict[str, Any],
        incoming: dict[str, Any],
    ) -> str:
        """Classify protected-claim conflict type for two records."""
        source = str(
            existing.get("claim_source", "") or incoming.get("claim_source", "")
        )
        if existing.get("signed_fact_id") or incoming.get("signed_fact_id"):
            return "signed_fact"
        if existing.get("passport_claim_id") or incoming.get("passport_claim_id"):
            return "passport_claim"
        if source == "signed_fact":
            return "signed_fact"
        if source == "passport_claim":
            return "passport_claim"
        return ""

    @classmethod
    def _conflict_record(
        cls,
        *,
        key: str,
        tenant_id: str,
        conflict_type: str,
        existing: dict[str, Any],
        incoming: dict[str, Any],
        reason: str,
    ) -> dict[str, str]:
        """Build a tenant-safe conflict record without raw fact payloads."""
        return {
            "event": "kb_conflict",
            "key": key,
            "tenant_id": tenant_id,
            "conflict_type": conflict_type,
            "existing_key": str(existing.get("key", "")),
            "existing_version": str(existing.get("version", "")),
            "existing_hash": str(existing.get("content_hash", "")),
            "incoming_hash": str(incoming.get("kb_content_hash", "")),
            "claim_id": str(incoming.get("claim_id") or existing.get("claim_id", "")),
            "signed_fact_id": str(
                incoming.get("signed_fact_id") or existing.get("signed_fact_id", "")
            ),
            "passport_claim_id": str(
                incoming.get("passport_claim_id")
                or existing.get("passport_claim_id", "")
            ),
            "reference": cls._first_ref(existing),
            "reason": reason,
        }

    @classmethod
    def _record_refs(cls, key: str, record: dict[str, Any]) -> set[str]:
        """Return all stable reference identifiers for a record."""
        refs = {key}
        for field in (
            "source_id",
            "external_id",
            "claim_id",
            "signed_fact_id",
            "passport_claim_id",
        ):
            value = str(record.get(field, "")).strip()
            if value:
                refs.add(value)
        return refs - {""}

    @classmethod
    def _first_ref(cls, record: dict[str, Any]) -> str:
        """Return the lexicographically first stable reference for a record."""
        refs = sorted(cls._record_refs(str(record.get("key", "")), record))
        return refs[0] if refs else ""

    @staticmethod
    def _metadata_list(value: Any) -> set[str]:
        """Normalise comma-separated or iterable metadata references."""
        if value in (None, ""):
            return set()
        if isinstance(value, str):
            return {item.strip() for item in value.split(",") if item.strip()}
        if isinstance(value, list | tuple | set):
            return {str(item).strip() for item in value if str(item).strip()}
        return {str(value).strip()}

    @staticmethod
    def _dedupe_conflicts(records: list[dict[str, str]]) -> list[dict[str, str]]:
        """Remove duplicate conflict reports while preserving order."""
        seen: set[tuple[str, str, str, str]] = set()
        out: list[dict[str, str]] = []
        for record in records:
            marker = (
                record["conflict_type"],
                record["key"],
                record["existing_key"],
                record["reference"],
            )
            if marker in seen:
                continue
            seen.add(marker)
            out.append(record)
        return out
