# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Vector Store Snapshot Audit
"""Merkle snapshot-audit surface of the vector ground-truth store.

:class:`SnapshotAuditMixin` renders the canonical, deterministic view of
the knowledge base held by
:class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`:
sorted tenant-scoped snapshot records, a domain-separated Merkle root
over them, and the compact audit payload combining the root with ledger
counts. It holds no state of its own — every read comes from the
composing store through the contracts declared below.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

__all__ = ["SnapshotAuditMixin"]


class SnapshotAuditMixin:
    """Deterministic KB snapshot records, Merkle root, and audit payload.

    Tenant resolution, the revision counter, and the version/retraction/
    replacement/conflict ledgers come from the composing
    :class:`~director_ai.core.retrieval.vector_store.store.VectorGroundTruthStore`
    through the contracts declared below.
    """

    if TYPE_CHECKING:
        # Provided by the composing VectorGroundTruthStore / its mixins.
        @property
        def revision(self) -> int: ...

        def _resolved_tenant_id(self, tenant_id: str = "") -> str: ...

        def version_manifest(
            self, tenant_id: str = ""
        ) -> dict[str, dict[str, str]]: ...

        def retraction_records(self, tenant_id: str = "") -> list[dict[str, str]]: ...

        def replacement_records(self, tenant_id: str = "") -> list[dict[str, str]]: ...

        def conflict_reports(
            self,
            tenant_id: str = "",
            key: str | None = None,
        ) -> list[dict[str, str]]: ...

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

    @staticmethod
    def _snapshot_leaf(record: dict[str, str]) -> bytes:
        """Return a domain-separated Merkle leaf digest for a record."""
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(b"director-ai/kb-snapshot/v1/leaf\x00" + payload).digest()

    @staticmethod
    def _merkle_root_hex(leaves: list[bytes]) -> str:
        """Return the Merkle root for snapshot leaves as hex."""
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
