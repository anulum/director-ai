# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Enterprise Postgres Audit Sink
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Immutable Audit Logging to PostgreSQL.

Provides compliance-grade persistence for AI decisions, halt events, and policy violations.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any

from ..core.audit import AuditEntry

logger = logging.getLogger("DirectorAI.Audit.PG")


class PostgresAuditSink:
    """Synchronous Postgres sink for immutable audit trails.

    Uses standard psycopg2 (or fallback sqlite for testing/local edge).
    """

    def __init__(self, db_url: str, table_name: str = "director_audit_logs"):
        self.db_url = db_url
        self.table_name = table_name
        self._lock = threading.Lock()
        self._conn: Any = None
        self._is_sqlite = db_url.startswith("sqlite")
        self._connect()

    def _connect(self) -> None:
        """Establish connection and verify schema."""
        try:
            if self._is_sqlite:
                import sqlite3

                # e.g sqlite:///audit.db -> audit.db
                path = self.db_url.replace("sqlite:///", "").replace("sqlite://", "")
                if not path:
                    path = ":memory:"
                self._conn = sqlite3.connect(path, check_same_thread=False)
            else:
                import psycopg2

                self._conn = psycopg2.connect(self.db_url)

            self._ensure_schema()
            logger.info(
                "Connected to Audit Sink: %s (table: %s)",
                "SQLite" if self._is_sqlite else "Postgres",
                self.table_name,
            )
        except Exception as e:
            logger.error("Failed to connect to PostgresAuditSink: %s", e)
            self._conn = None

    def _ensure_schema(self) -> None:
        if not self._conn:
            return

        with self._lock:
            cur = self._conn.cursor()
            query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(64) NOT NULL,
                query_hash VARCHAR(64) NOT NULL,
                response_length INT NOT NULL,
                approved BOOLEAN NOT NULL,
                score FLOAT NOT NULL,
                h_logical FLOAT NOT NULL,
                h_factual FLOAT NOT NULL,
                policy_violations TEXT NOT NULL,
                tenant_id VARCHAR(128) NOT NULL,
                halt_reason TEXT NOT NULL,
                latency_ms FLOAT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """

            if self._is_sqlite:
                query = query.replace(
                    "SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT"
                )
                query = query.replace("BOOLEAN", "INTEGER")

            try:
                cur.execute(query)
                self._conn.commit()
            except Exception as e:
                self._conn.rollback()
                logger.error("Failed to initialize audit schema: %s", e)
            finally:
                cur.close()

    def write(self, entry: AuditEntry) -> None:
        """Write a single AuditEntry immutably to the database."""
        if not self._conn:
            return

        with self._lock:
            try:
                cur = self._conn.cursor()

                # In SQLite we might need placeholders like ? instead of %s
                placeholder = "?" if self._is_sqlite else "%s"

                query = f"""
                INSERT INTO {self.table_name} (
                    timestamp, query_hash, response_length, approved,
                    score, h_logical, h_factual, policy_violations,
                    tenant_id, halt_reason, latency_ms
                ) VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                """

                values = (
                    entry.timestamp,
                    entry.query_hash,
                    entry.response_length,
                    int(entry.approved) if self._is_sqlite else entry.approved,
                    entry.score,
                    entry.h_logical,
                    entry.h_factual,
                    json.dumps(entry.policy_violations),
                    entry.tenant_id,
                    entry.halt_reason,
                    entry.latency_ms,
                )

                cur.execute(query, values)
                self._conn.commit()
            except Exception as e:
                self._conn.rollback()
                logger.error("Failed to persist audit log to DB: %s", e)
            finally:
                cur.close()
