# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Compliance audit log — every LLM interaction scored and recorded.

Extends the calibration FeedbackStore with model, provider, and latency
metadata needed for EU AI Act Article 15 documentation.

Every call through the gateway gets logged here. The log is the compliance trail.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

__all__ = ["AuditEntry", "AuditLog"]


@dataclass
class AuditEntry:
    """A single scored LLM interaction."""

    prompt: str
    response: str
    model: str
    provider: str
    score: float
    approved: bool
    verdict_confidence: float
    task_type: str
    domain: str
    latency_ms: float
    timestamp: float
    tenant_id: str = ""
    human_override: bool | None = None


class AuditLog:
    """Thread-safe SQLite audit log for compliance reporting.

    Every LLM call scored by the gateway gets an entry. The log supports
    time-range queries, model/domain filtering, and export for Article 15
    documentation.
    """

    def __init__(self, db_path: str | Path = "director_audit.db"):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self._db_path, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                score REAL NOT NULL,
                approved INTEGER NOT NULL,
                verdict_confidence REAL NOT NULL DEFAULT 0.0,
                task_type TEXT NOT NULL DEFAULT '',
                domain TEXT NOT NULL DEFAULT '',
                latency_ms REAL NOT NULL DEFAULT 0.0,
                tenant_id TEXT NOT NULL DEFAULT '',
                human_override INTEGER,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_log(model)
        """)
        self._conn.commit()

    def log(self, entry: AuditEntry) -> None:
        """Record a scored LLM interaction."""
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                """INSERT INTO audit_log
                   (prompt, response, model, provider, score, approved,
                    verdict_confidence, task_type, domain, latency_ms,
                    tenant_id, human_override, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.prompt,
                    entry.response,
                    entry.model,
                    entry.provider,
                    entry.score,
                    int(entry.approved),
                    entry.verdict_confidence,
                    entry.task_type,
                    entry.domain,
                    entry.latency_ms,
                    entry.tenant_id,
                    int(entry.human_override)
                    if entry.human_override is not None
                    else None,
                    entry.timestamp,
                ),
            )
            self._conn.commit()

    def query(
        self,
        since: float | None = None,
        until: float | None = None,
        model: str | None = None,
        domain: str | None = None,
        tenant_id: str | None = None,
        limit: int = 0,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters."""
        with self._lock:
            if self._conn is None:
                return []
            clauses: list[str] = []
            params: list = []
            if since is not None:
                clauses.append("timestamp >= ?")
                params.append(since)
            if until is not None:
                clauses.append("timestamp <= ?")
                params.append(until)
            if model is not None:
                clauses.append("model = ?")
                params.append(model)
            if domain is not None:
                clauses.append("domain = ?")
                params.append(domain)
            if tenant_id is not None:
                clauses.append("tenant_id = ?")
                params.append(tenant_id)

            query = "SELECT prompt, response, model, provider, score, approved, verdict_confidence, task_type, domain, latency_ms, timestamp, tenant_id, human_override FROM audit_log"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY timestamp DESC"
            if limit > 0:
                query += " LIMIT ?"
                params.append(limit)

            rows = self._conn.execute(query, params).fetchall()

        return [
            AuditEntry(
                prompt=r[0],
                response=r[1],
                model=r[2],
                provider=r[3],
                score=r[4],
                approved=bool(r[5]),
                verdict_confidence=r[6],
                task_type=r[7],
                domain=r[8],
                latency_ms=r[9],
                timestamp=r[10],
                tenant_id=r[11],
                human_override=bool(r[12]) if r[12] is not None else None,
            )
            for r in rows
        ]

    def count(
        self,
        since: float | None = None,
        model: str | None = None,
    ) -> int:
        """Count entries with optional filters."""
        with self._lock:
            if self._conn is None:
                return 0
            clauses: list[str] = []
            params: list = []
            if since is not None:
                clauses.append("timestamp >= ?")
                params.append(since)
            if model is not None:
                clauses.append("model = ?")
                params.append(model)
            query = "SELECT COUNT(*) FROM audit_log"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            row = self._conn.execute(query, params).fetchone()
            return row[0] if row else 0

    def close(self) -> None:
        """Close the database. Safe to call multiple times."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
