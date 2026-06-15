# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Cross-Document Consistency Memory

"""Durable tenant-scoped consistency memory for generated outputs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any

from ..text_overlap import word_overlap

__all__ = [
    "CrossDocumentConflict",
    "CrossDocumentConsistencyMemory",
    "CrossDocumentConsistencyReport",
    "StoredDocument",
]

ConsistencyScoreFn = Callable[[str, str], float]
_TENANT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_tenant_id(tenant_id: str) -> str:
    """Return a validated tenant identifier for storage queries."""
    value = str(tenant_id).strip()
    if not _TENANT_ID_RE.fullmatch(value):
        raise ValueError("tenant_id must match ^[A-Za-z0-9_-]{1,64}$")
    return value


def _validate_non_empty(name: str, value: str) -> str:
    """Return stripped text after enforcing a non-empty value."""
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _validate_score(score: float) -> float:
    """Return a finite unit-interval consistency score."""
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("consistency score must be finite and in [0, 1]")
    return float(score)


def _content_hash(text: str) -> str:
    """Return a stable tenant-safe content fingerprint."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(frozen=True)
class StoredDocument:
    """One tenant-scoped document retained for consistency checks."""

    tenant_id: str
    document_id: str
    text: str
    content_hash: str
    created_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialise the stored document with optional raw text."""
        return {
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "text": self.text if include_text else None,
        }


@dataclass(frozen=True)
class CrossDocumentConflict:
    """Tenant-safe report for one cross-document contradiction."""

    tenant_id: str
    incoming_document_id: str
    existing_document_id: str
    incoming_hash: str
    existing_hash: str
    score: float
    existing_text: str = ""
    incoming_text: str = ""

    def __post_init__(self) -> None:
        """Validate conflict score after dataclass construction."""
        _validate_score(self.score)

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialise the conflict with optional raw document text."""
        return {
            "tenant_id": self.tenant_id,
            "incoming_document_id": self.incoming_document_id,
            "existing_document_id": self.existing_document_id,
            "incoming_hash": self.incoming_hash,
            "existing_hash": self.existing_hash,
            "score": self.score,
            "existing_text": self.existing_text if include_text else None,
            "incoming_text": self.incoming_text if include_text else None,
        }


@dataclass(frozen=True)
class CrossDocumentConsistencyReport:
    """Decision for an incoming document against tenant memory."""

    decision: str
    tenant_id: str
    document_id: str
    incoming_hash: str
    checked_documents: int
    conflicts: tuple[CrossDocumentConflict, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate decision state and freeze conflict ordering."""
        if self.decision not in {"allow", "warn", "block"}:
            raise ValueError(f"unsupported decision {self.decision!r}")
        object.__setattr__(self, "conflicts", tuple(self.conflicts))

    @property
    def blocked(self) -> bool:
        """Return whether the incoming document must not be recorded."""
        return self.decision == "block"

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialise the consistency decision for API or audit output."""
        return {
            "decision": self.decision,
            "tenant_id": self.tenant_id,
            "document_id": self.document_id,
            "incoming_hash": self.incoming_hash,
            "checked_documents": self.checked_documents,
            "conflicts": [
                conflict.to_dict(include_text=include_text)
                for conflict in self.conflicts
            ],
        }


class CrossDocumentConsistencyMemory:
    """SQLite-backed tenant memory for cross-document consistency checks."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        *,
        score_fn: ConsistencyScoreFn | None = None,
        use_builtin_similarity: bool = False,
        warn_threshold: float = 0.65,
        contradiction_threshold: float = 0.85,
        max_documents_per_tenant: int = 1_000,
    ) -> None:
        """Initialise SQLite storage and retention policy settings."""
        self.db_path = str(db_path)
        self.score_fn = score_fn
        self._use_builtin_similarity = bool(use_builtin_similarity)
        self.warn_threshold = _validate_score(warn_threshold)
        self.contradiction_threshold = _validate_score(contradiction_threshold)
        if self.warn_threshold > self.contradiction_threshold:
            raise ValueError("warn_threshold must be <= contradiction_threshold")
        if max_documents_per_tenant < 1:
            raise ValueError("max_documents_per_tenant must be positive")
        self.max_documents_per_tenant = max_documents_per_tenant
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _builtin_similarity(self, text_a: str, text_b: str) -> float:
        # Delegates to the shared measured-fast-path word-overlap helper (pure
        # Python below a large-input threshold, Rust above it).
        return _validate_score(word_overlap(text_a, text_b, logger_name=__name__))

    def _init_schema(self) -> None:
        """Create the durable cross-document memory schema."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cross_document_memory (
                tenant_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at REAL NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (tenant_id, document_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_cross_document_memory_tenant_created
            ON cross_document_memory (tenant_id, created_at)
            """
        )
        self._conn.commit()

    def check_document(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
    ) -> CrossDocumentConsistencyReport:
        """Evaluate an incoming document against tenant memory."""
        tenant = _validate_tenant_id(tenant_id)
        doc_id = _validate_non_empty("document_id", document_id)
        incoming_text = _validate_non_empty("text", text)
        incoming_hash = _content_hash(incoming_text)
        existing = self.list_documents(tenant)
        conflicts: list[CrossDocumentConflict] = []
        if self.score_fn is not None or self._use_builtin_similarity:
            score_fn = self.score_fn or self._builtin_similarity
            for document in existing:
                if document.document_id == doc_id:
                    continue
                score = _validate_score(score_fn(document.text, incoming_text))
                if score >= self.warn_threshold:
                    conflicts.append(
                        CrossDocumentConflict(
                            tenant_id=tenant,
                            incoming_document_id=doc_id,
                            existing_document_id=document.document_id,
                            incoming_hash=incoming_hash,
                            existing_hash=document.content_hash,
                            score=score,
                            existing_text=document.text,
                            incoming_text=incoming_text,
                        )
                    )
        decision = "allow"
        if any(c.score >= self.contradiction_threshold for c in conflicts):
            decision = "block"
        elif conflicts:
            decision = "warn"
        return CrossDocumentConsistencyReport(
            decision=decision,
            tenant_id=tenant,
            document_id=doc_id,
            incoming_hash=incoming_hash,
            checked_documents=len(existing),
            conflicts=tuple(sorted(conflicts, key=lambda c: c.score, reverse=True)),
        )

    def record_document(
        self,
        tenant_id: str,
        document_id: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> CrossDocumentConsistencyReport:
        """Record a document unless consistency checks block it."""
        report = self.check_document(tenant_id, document_id, text)
        if report.blocked:
            return report
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cross_document_memory
            (tenant_id, document_id, text, content_hash, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                report.tenant_id,
                report.document_id,
                _validate_non_empty("text", text),
                report.incoming_hash,
                time(),
                json.dumps(dict(metadata or {}), sort_keys=True),
            ),
        )
        self._conn.commit()
        self._enforce_retention(report.tenant_id)
        return report

    def get_document(self, tenant_id: str, document_id: str) -> StoredDocument | None:
        """Return one stored tenant document by identifier."""
        tenant = _validate_tenant_id(tenant_id)
        doc_id = _validate_non_empty("document_id", document_id)
        row = self._conn.execute(
            """
            SELECT tenant_id, document_id, text, content_hash, created_at, metadata_json
            FROM cross_document_memory
            WHERE tenant_id = ? AND document_id = ?
            """,
            (tenant, doc_id),
        ).fetchone()
        return _row_to_document(row) if row is not None else None

    def list_documents(self, tenant_id: str) -> tuple[StoredDocument, ...]:
        """Return stored documents for a tenant in retention order."""
        tenant = _validate_tenant_id(tenant_id)
        rows = self._conn.execute(
            """
            SELECT tenant_id, document_id, text, content_hash, created_at, metadata_json
            FROM cross_document_memory
            WHERE tenant_id = ?
            ORDER BY created_at ASC, document_id ASC
            """,
            (tenant,),
        ).fetchall()
        return tuple(_row_to_document(row) for row in rows)

    def count(self, *, tenant_id: str | None = None) -> int:
        """Return total document count globally or for one tenant."""
        if tenant_id is None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM cross_document_memory"
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM cross_document_memory WHERE tenant_id = ?",
                (_validate_tenant_id(tenant_id),),
            ).fetchone()
        return int(row["n"])

    def delete_tenant(self, tenant_id: str) -> int:
        """Delete all documents for a tenant and return the removed count."""
        tenant = _validate_tenant_id(tenant_id)
        before = self.count(tenant_id=tenant)
        self._conn.execute(
            "DELETE FROM cross_document_memory WHERE tenant_id = ?",
            (tenant,),
        )
        self._conn.commit()
        return before

    def close(self) -> None:
        """Close the backing SQLite connection."""
        self._conn.close()

    def _enforce_retention(self, tenant_id: str) -> None:
        """Delete oldest tenant documents above the retention limit."""
        rows = self._conn.execute(
            """
            SELECT document_id
            FROM cross_document_memory
            WHERE tenant_id = ?
            ORDER BY created_at ASC, document_id ASC
            """,
            (tenant_id,),
        ).fetchall()
        overflow = len(rows) - self.max_documents_per_tenant
        if overflow <= 0:
            return
        to_delete = [row["document_id"] for row in rows[:overflow]]
        self._conn.executemany(
            """
            DELETE FROM cross_document_memory
            WHERE tenant_id = ? AND document_id = ?
            """,
            [(tenant_id, doc_id) for doc_id in to_delete],
        )
        self._conn.commit()


def _row_to_document(row: sqlite3.Row) -> StoredDocument:
    """Convert a SQLite row into a StoredDocument."""
    return StoredDocument(
        tenant_id=row["tenant_id"],
        document_id=row["document_id"],
        text=row["text"],
        content_hash=row["content_hash"],
        created_at=float(row["created_at"]),
        metadata=json.loads(row["metadata_json"] or "{}"),
    )
