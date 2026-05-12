# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — VectorGroundTruthStore

"""``VectorGroundTruthStore`` — semantic RAG store on top of any
``VectorBackend``.

Extends the keyword-based :class:`GroundTruthStore` with
embedding-based similarity search and exposes a ``grounded()``
factory that wires the recommended hybrid (BM25 + dense) recipe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ...metrics import metrics
from ...otel import trace_vector_add, trace_vector_query
from ...types import EvidenceChunk
from ..knowledge import GroundTruthStore, _require_non_empty_string
from .base import (
    RECOMMENDED_EMBEDDING_MODEL,
    InMemoryBackend,
    VectorBackend,
    logger,
)
from .composite import HybridBackend
from .embedding import SentenceTransformerBackend

__all__ = ["VectorGroundTruthStore"]


def _is_vector_backend_like(backend: object) -> bool:
    return all(
        callable(getattr(backend, method, None)) for method in ("add", "query", "count")
    )


def _require_non_negative_top_k(top_k: int) -> int:
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0; got {top_k!r}")
    return top_k


class VectorGroundTruthStore(GroundTruthStore):
    """Ground truth store with vector-based semantic retrieval.

    Extends the keyword-based ``GroundTruthStore`` with embedding-based
    similarity search. Falls back to keyword matching when the vector
    backend returns no results.

    Parameters
    ----------
    backend : VectorBackend — vector DB backend (default: InMemoryBackend).

    """

    def __init__(
        self,
        backend: VectorBackend | None = None,
        tenant_id: str = "",
    ) -> None:
        super().__init__()
        if backend is not None and not _is_vector_backend_like(backend):
            raise ValueError("backend must provide add, query, and count methods")
        if not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a string")
        self.backend = backend if backend is not None else InMemoryBackend()
        self.tenant_id = tenant_id.strip()
        self._version_records: dict[str, dict[str, str]] = {}
        self._retraction_records: list[dict[str, str]] = []
        self._replacement_records: list[dict[str, str]] = []
        self._conflict_records: list[dict[str, str]] = []

    def _resolved_tenant_id(self, tenant_id: str = "") -> str:
        if not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a string")
        return (tenant_id or self.tenant_id).strip()

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

    def conflict_reports(
        self,
        tenant_id: str = "",
        key: str | None = None,
    ) -> list[dict[str, str]]:
        """Return KB conflict reports visible to *tenant_id*."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        reports = []
        for record in self._conflict_records:
            if tenant_id and record.get("tenant_id", "") != tenant_id:
                continue
            if key is not None and record.get("key") != key:
                continue
            reports.append(dict(record))
        return reports

    def kb_snapshot_records(self, tenant_id: str = "") -> list[dict[str, str]]:
        """Return canonical KB snapshot records visible to *tenant_id*."""
        records = []
        for record in self.version_manifest(tenant_id).values():
            snapshot_record = {
                "key": record.get("key", ""),
                "tenant_id": record.get("tenant_id", ""),
                "version": record.get("version", ""),
                "chunk_version": record.get("chunk_version", ""),
                "content_hash": record.get("content_hash", ""),
                "previous_hash": record.get("previous_hash", ""),
                "record_kind": record.get("record_kind", ""),
                "chunk_index": record.get("chunk_index", ""),
                "status": record.get("status", "active"),
                "retraction_reason": record.get("retraction_reason", ""),
                "replacement_reason": record.get("replacement_reason", ""),
                "source_id": record.get("source_id", ""),
                "external_id": record.get("external_id", ""),
                "source_timestamp": record.get("source_timestamp", ""),
                "updated_timestamp": record.get("updated_timestamp", ""),
                "citation_status": record.get("citation_status", ""),
                "status_source": record.get("status_source", ""),
                "status_observed_at": record.get("status_observed_at", ""),
                "claim_id": record.get("claim_id", ""),
                "claim_source": record.get("claim_source", ""),
                "signed_fact_id": record.get("signed_fact_id", ""),
                "passport_claim_id": record.get("passport_claim_id", ""),
            }
            records.append(snapshot_record)
        return sorted(
            records,
            key=lambda item: (
                item["tenant_id"],
                item["key"],
                item["record_kind"],
                item["chunk_index"],
            ),
        )

    def kb_snapshot_root(self, tenant_id: str = "") -> str:
        """Return a deterministic Merkle root for the current KB snapshot."""
        leaves = [
            self._snapshot_leaf(record)
            for record in self.kb_snapshot_records(tenant_id)
        ]
        return self._merkle_root_hex(leaves)

    def kb_snapshot_audit_record(self, tenant_id: str = "") -> dict[str, str | int]:
        """Return a compact audit payload for the current KB snapshot."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        records = self.kb_snapshot_records(tenant_id)
        return {
            "event": "kb_snapshot",
            "tenant_id": tenant_id,
            "revision": self.revision,
            "record_count": len(records),
            "retraction_count": len(self.retraction_records(tenant_id)),
            "replacement_count": len(self.replacement_records(tenant_id)),
            "conflict_count": len(self.conflict_reports(tenant_id)),
            "merkle_root": self._merkle_root_hex(
                [self._snapshot_leaf(record) for record in records]
            ),
        }

    def freshness_status_signals(
        self,
        tenant_id: str = "",
        key: str | None = None,
    ) -> list[dict[str, str | float]]:
        """Return citation status signals for temporal freshness scoring."""
        tenant_id = self._resolved_tenant_id(tenant_id)
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
                signal["published_at"] = float(published_at)
            if updated_at:
                signal["updated_at"] = float(updated_at)
            observed_at = record.get("status_observed_at", "")
            if observed_at:
                signal["observed_at"] = float(observed_at)
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
        return event

    def add_fact(
        self,
        key: str,
        value: str,
        tenant_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Alias for add() — also populates parent keyword store."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        key = _require_non_empty_string("key", key)
        value = _require_non_empty_string("value", value)
        self.add(key, value, metadata=metadata, tenant_id=tenant_id)
        fact_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts[fact_key] = value

    def ingest(self, texts: list[str], tenant_id: str = "") -> int:
        """Bulk-add plain text documents into the vector backend."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of non-empty strings")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings")
        for i, text in enumerate(texts):
            doc_id = f"ingest_{i}_{tenant_id}"
            metadata = {
                "source": "ingest",
                **self._build_version_metadata(
                    key=doc_id,
                    value=text,
                    tenant_id=tenant_id,
                    record_kind="derived_chunk",
                    requested_bump="patch",
                    chunk_index=i,
                ),
            }
            if tenant_id:
                metadata["tenant_id"] = tenant_id
            self.backend.add(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
            )
            self._commit_version_metadata(doc_id, metadata, tenant_id)
        if texts:
            self._bump_revision()
        logger.info("Ingested %d documents into vector backend.", len(texts))
        return len(texts)

    def add(
        self,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "",
    ) -> None:
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        key = _require_non_empty_string("key", key)
        value = _require_non_empty_string("value", value)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        doc_id = f"{tenant_id}::{key}" if tenant_id else key
        combined_text = f"{key}: {value}"
        incoming = dict(metadata or {})
        requested_bump = str(incoming.pop("kb_version_bump", "patch"))
        meta = {
            **incoming,
            "key": key,
            "value": value,
            **self._build_version_metadata(
                key=key,
                value=value,
                tenant_id=tenant_id,
                record_kind="fact",
                requested_bump=requested_bump,
                chunk_index=0,
            ),
        }
        meta.update(self._normalised_claim_metadata(meta))
        if tenant_id:
            meta["tenant_id"] = tenant_id

        with trace_vector_add() as span:
            start_time = time.monotonic()
            metrics.inc("knowledge_adds_total")
            try:
                conflicts = self._build_conflict_reports(key, meta, tenant_id)
                self.backend.add(doc_id=doc_id, text=combined_text, metadata=meta)
                self._commit_version_metadata(key, meta, tenant_id)
                self._conflict_records.extend(conflicts)
                self._bump_revision()
                duration = time.monotonic() - start_time
                metrics.observe("knowledge_add_duration_seconds", duration)
                span.set_attribute("vector.doc_id", doc_id)
                span.set_attribute("vector.tenant_id", tenant_id)
            except Exception as e:
                metrics.inc("knowledge_add_errors")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                raise ValueError(f"Failed to add to vector store: {e}") from e

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

    def _build_conflict_reports(
        self,
        key: str,
        metadata: dict[str, Any],
        tenant_id: str,
    ) -> list[dict[str, str]]:
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
        refs = sorted(cls._record_refs(str(record.get("key", "")), record))
        return refs[0] if refs else ""

    @staticmethod
    def _metadata_list(value: Any) -> set[str]:
        if value in (None, ""):
            return set()
        if isinstance(value, str):
            return {item.strip() for item in value.split(",") if item.strip()}
        if isinstance(value, list | tuple | set):
            return {str(item).strip() for item in value if str(item).strip()}
        return {str(value).strip()}

    @staticmethod
    def _dedupe_conflicts(records: list[dict[str, str]]) -> list[dict[str, str]]:
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

    @staticmethod
    def _version_key(key: str, tenant_id: str) -> str:
        return f"{tenant_id}::{key}" if tenant_id else key

    @staticmethod
    def _content_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _snapshot_leaf(record: dict[str, str]) -> bytes:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(b"director-ai/kb-snapshot/v1/leaf\x00" + payload).digest()

    @staticmethod
    def _merkle_root_hex(leaves: list[bytes]) -> str:
        if not leaves:
            return hashlib.sha256(b"director-ai/kb-snapshot/v1/empty").hexdigest()
        level = list(leaves)
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha256(
                    b"director-ai/kb-snapshot/v1/node\x00" + level[i] + level[i + 1]
                ).digest()
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    @staticmethod
    def _next_semver(current: str, requested_bump: str) -> str:
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

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> str | None:
        """Retrieve context as a string (matching parent interface).

        Falls back to keyword-based parent if vector search returns nothing.
        """
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        query = _require_non_empty_string("query", query)
        top_k = _require_non_negative_top_k(top_k)
        with trace_vector_query() as span:
            start_time = time.monotonic()
            metrics.inc("knowledge_queries_total")
            try:
                try:
                    results = self.backend.query(
                        query,
                        n_results=top_k,
                        tenant_id=tenant_id,
                    )
                except TypeError:
                    # Backend doesn't accept tenant_id
                    results = self.backend.query(query, n_results=top_k)
                span.set_attribute("vector.query.k", top_k)
                span.set_attribute("vector.tenant_id", tenant_id)

                if results:
                    active_results = self._active_results(results, tenant_id)
                    texts = [r["text"] for r in active_results]
                    if not texts:
                        return super().retrieve_context(
                            query,
                            tenant_id=tenant_id,
                            top_k=top_k,
                        )
                    duration = time.monotonic() - start_time
                    metrics.observe("knowledge_query_duration_seconds", duration)
                    return "; ".join(texts)

                duration = time.monotonic() - start_time
                metrics.observe("knowledge_query_duration_seconds", duration)
                # Fall back to keyword-based parent
                return super().retrieve_context(
                    query,
                    tenant_id=tenant_id,
                    top_k=top_k,
                )
            except Exception as e:
                metrics.inc("knowledge_query_errors")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                raise ValueError(f"Failed to query vector store: {e}") from e

    @classmethod
    def grounded(
        cls,
        embedding_model: str = RECOMMENDED_EMBEDDING_MODEL,
        use_hybrid: bool = True,
        rrf_k: int = 60,
        tenant_id: str = "",
    ) -> VectorGroundTruthStore:
        """Factory for the recommended grounded retrieval recipe.

        Sets up hybrid retrieval (BM25 + dense) with a sentence-transformer
        embedding model. This is the intended production path for domain
        profiles (medical, finance, legal) where NLI-only scoring has
        100% FPR without KB grounding.

        Usage::

            store = VectorGroundTruthStore.grounded()
            store.ingest(["Your product documentation...", ...])
            scorer = CoherenceScorer(ground_truth_store=store, use_nli=True)

        Parameters
        ----------
        embedding_model : str
            HuggingFace model ID for dense embeddings.
            Default: ``BAAI/bge-large-en-v1.5``.
        use_hybrid : bool
            Wrap dense backend with BM25 + RRF fusion (default True).
        rrf_k : int
            Reciprocal Rank Fusion parameter (default 60).
        tenant_id : str
            Default tenant scope for multi-tenant deployments.
        """
        dense: VectorBackend
        try:
            dense = SentenceTransformerBackend(model_name=embedding_model)
        except Exception:
            logger.warning(
                "sentence-transformers not available, falling back to InMemoryBackend. "
                "Install with: pip install director-ai[vector]"
            )
            dense = InMemoryBackend()

        backend = HybridBackend(base=dense, rrf_k=rrf_k) if use_hybrid else dense

        return cls(backend=backend, tenant_id=tenant_id)

    def retrieve_context_with_chunks(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> list[EvidenceChunk]:
        """Retrieve context as EvidenceChunk objects."""
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        query = _require_non_empty_string("query", query)
        top_k = _require_non_negative_top_k(top_k)
        with trace_vector_query() as span:
            start_time = time.monotonic()
            try:
                try:
                    results = self.backend.query(
                        query,
                        n_results=top_k,
                        tenant_id=tenant_id,
                    )
                except TypeError:
                    # Backend doesn't accept tenant_id
                    results = self.backend.query(query, n_results=top_k)
                chunks = []
                for r in self._active_results(results, tenant_id):
                    chunks.append(
                        EvidenceChunk(
                            text=r["text"],
                            distance=r.get("distance", 0.0),
                            source=f"vector:{r['id']}",
                        ),
                    )
                duration = time.monotonic() - start_time
                metrics.observe("knowledge_query_duration_seconds", duration)
                if not chunks:
                    return super().retrieve_context_with_chunks(
                        query,
                        top_k=top_k,
                        tenant_id=tenant_id,
                    )
                return chunks
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise ValueError(f"Failed to query vector store: {e}") from e

    def _active_results(
        self,
        results: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                active.append(result)
                continue
            source_key = str(metadata.get("kb_source_key", result.get("id", "")))
            result_tenant = str(metadata.get("tenant_id", tenant_id))
            version_key = self._version_key(source_key, result_tenant)
            record = self._version_records.get(version_key)
            if record is not None and record.get("status") == "retracted":
                continue
            result_hash = str(metadata.get("kb_content_hash", ""))
            if (
                record is not None
                and result_hash
                and result_hash != record["content_hash"]
            ):
                continue
            active.append(result)
        return active
