# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Enterprise Postgres Audit Sink

"""Immutable audit logging to PostgreSQL (or SQLite for testing)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from typing import Any

from ..core.safety.audit import AuditEntry

logger = logging.getLogger("DirectorAI.Audit.PG")

SCHEMA_VERSION = 2

_COLUMNS = (
    "timestamp",
    "query_hash",
    "response_length",
    "approved",
    "score",
    "h_logical",
    "h_factual",
    "policy_violations",
    "tenant_id",
    "halt_reason",
    "latency_ms",
)


class PostgresAuditSink:
    """Synchronous Postgres/SQLite sink for immutable audit trails.

    Includes forward-only schema migrations tracked in a ``_schema_version``
    table.  Use ``query()`` to retrieve audit records for dashboarding.
    """

    def __init__(
        self,
        db_url: str,
        table_name: str = "director_audit_logs",
        pool_min: int = 1,
        pool_max: int = 5,
    ):
        self.db_url = db_url
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")
        self.table_name = table_name
        self._lock = threading.Lock()
        self._conn: Any = None
        self._pool: Any = None
        self._is_sqlite = db_url.startswith("sqlite")
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._connect()

    def _connect(self) -> None:
        try:
            if self._is_sqlite:
                import sqlite3

                path = self.db_url.replace("sqlite:///", "").replace("sqlite://", "")
                self._conn = sqlite3.connect(
                    path or ":memory:",
                    check_same_thread=False,
                )
            else:
                import psycopg2.pool

                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    self._pool_min,
                    self._pool_max,
                    self.db_url,
                )
                self._conn = self._pool.getconn()
            self._migrate()
            if self._pool:
                self._pool.putconn(self._conn)
                self._conn = None
            logger.info(
                "Audit sink ready: %s (table: %s, schema v%d)",
                "SQLite" if self._is_sqlite else "Postgres",
                self.table_name,
                SCHEMA_VERSION,
            )
        except Exception as e:
            logger.error("Failed to connect to PostgresAuditSink: %s", e)
            self._conn = None

    def _get_conn(self) -> Any:
        """Borrow a connection from pool (Postgres) or return the single conn (SQLite)."""
        if self._pool:
            return self._pool.getconn()
        return self._conn

    def _put_conn(self, conn: Any) -> None:
        """Return a connection to the pool (no-op for SQLite)."""
        if self._pool and conn is not None:
            self._pool.putconn(conn)

    # â”€â”€ Schema migration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _migrate(self) -> None:
        if not self._conn:
            return
        with self._lock:
            cur = self._conn.cursor()
            try:
                self._ensure_version_table(cur)
                current = self._current_version(cur)
                if current < 1:
                    self._apply_v1(cur)
                if current < 2:
                    self._apply_v2(cur)
                self._conn.commit()
            except Exception as e:
                self._conn.rollback()
                logger.error("Schema migration failed: %s", e)
            finally:
                cur.close()

    def _ensure_version_table(self, cur: Any) -> None:
        pk = (
            "INTEGER PRIMARY KEY AUTOINCREMENT"
            if self._is_sqlite
            else "SERIAL PRIMARY KEY"
        )
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS _schema_version ("
            f"id {pk}, version INT NOT NULL, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        )

    def _current_version(self, cur: Any) -> int:
        cur.execute("SELECT COALESCE(MAX(version), 0) FROM _schema_version")
        return int(cur.fetchone()[0])

    def _set_version(self, cur: Any, version: int) -> None:
        ph = "?" if self._is_sqlite else "%s"
        cur.execute(f"INSERT INTO _schema_version (version) VALUES ({ph})", (version,))  # nosec B608 — ph is literal "?" or "%s" dialect switch

    def _apply_v1(self, cur: Any) -> None:
        bool_type = "INTEGER" if self._is_sqlite else "BOOLEAN"
        pk = (
            "INTEGER PRIMARY KEY AUTOINCREMENT"
            if self._is_sqlite
            else "SERIAL PRIMARY KEY"
        )
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            f"id {pk},"
            f"timestamp VARCHAR(64) NOT NULL,"
            f"query_hash VARCHAR(64) NOT NULL,"
            f"response_length INT NOT NULL,"
            f"approved {bool_type} NOT NULL,"
            f"score FLOAT NOT NULL,"
            f"h_logical FLOAT NOT NULL,"
            f"h_factual FLOAT NOT NULL,"
            f"policy_violations TEXT NOT NULL,"
            f"tenant_id VARCHAR(128) NOT NULL,"
            f"halt_reason TEXT NOT NULL,"
            f"latency_ms FLOAT NOT NULL,"
            f"created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)",
        )
        self._set_version(cur, 1)

    def _apply_v2(self, cur: Any) -> None:
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_tenant "
            f"ON {self.table_name} (tenant_id)",
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_ts "
            f"ON {self.table_name} (created_at)",
        )
        self._set_version(cur, 2)

    # â”€â”€ Write â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def write(self, entry: AuditEntry) -> None:
        """Write a single AuditEntry immutably."""
        conn = self._get_conn()
        if conn is None:
            return
        ph = "?" if self._is_sqlite else "%s"
        placeholders = ", ".join([ph] * len(_COLUMNS))
        sql = f"INSERT INTO {self.table_name} ({', '.join(_COLUMNS)}) VALUES ({placeholders})"  # nosec B608 — table_name code-controlled; values via placeholders
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
        try:
            cur = conn.cursor()
            cur.execute(sql, values)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error("Failed to persist audit log: %s", e)
        finally:
            cur.close()
            self._put_conn(conn)

    async def async_write(self, entry: AuditEntry) -> None:
        """Non-blocking write for async callers (FastAPI handlers)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.write, entry)

    async def async_write_batch(self, entries: list[AuditEntry]) -> int:
        """Non-blocking batch write for async callers."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.write_batch, entries)

    def write_batch(self, entries: list[AuditEntry]) -> int:
        """Write multiple entries in a single transaction. Returns count written."""
        if not entries:
            return 0
        conn = self._get_conn()
        if conn is None:
            return 0
        ph = "?" if self._is_sqlite else "%s"
        placeholders = ", ".join([ph] * len(_COLUMNS))
        sql = f"INSERT INTO {self.table_name} ({', '.join(_COLUMNS)}) VALUES ({placeholders})"  # nosec B608 — table_name code-controlled; values via placeholders
        rows = [
            (
                e.timestamp,
                e.query_hash,
                e.response_length,
                int(e.approved) if self._is_sqlite else e.approved,
                e.score,
                e.h_logical,
                e.h_factual,
                json.dumps(e.policy_violations),
                e.tenant_id,
                e.halt_reason,
                e.latency_ms,
            )
            for e in entries
        ]
        try:
            cur = conn.cursor()
            cur.executemany(sql, rows)
            conn.commit()
            return len(rows)
        except Exception as e:
            conn.rollback()
            logger.error("Batch write failed: %s", e)
            return 0
        finally:
            cur.close()
            self._put_conn(conn)

    # â”€â”€ Query â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def query(
        self,
        tenant_id: str | None = None,
        approved: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve audit records with optional filters."""
        conn = self._get_conn()
        if conn is None:
            return []
        ph = "?" if self._is_sqlite else "%s"
        clauses: list[str] = []
        params: list[Any] = []
        if tenant_id is not None:
            clauses.append(f"tenant_id = {ph}")
            params.append(tenant_id)
        if approved is not None:
            val = int(approved) if self._is_sqlite else approved
            clauses.append(f"approved = {ph}")
            params.append(val)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT {', '.join(_COLUMNS)}, created_at FROM {self.table_name}"  # nosec B608 — table_name code-controlled
            f"{where} ORDER BY created_at DESC LIMIT {ph}"
        )
        params.append(limit)
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            cols = list(_COLUMNS) + ["created_at"]
            return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
        except Exception as e:
            logger.error("Audit query failed: %s", e)
            return []
        finally:
            cur.close()
            self._put_conn(conn)

    def count(self, tenant_id: str | None = None) -> int:
        """Count audit records, optionally filtered by tenant."""
        conn = self._get_conn()
        if conn is None:
            return 0
        ph = "?" if self._is_sqlite else "%s"
        if tenant_id is not None:
            sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE tenant_id = {ph}"  # nosec B608
            params: tuple = (tenant_id,)
        else:
            sql = f"SELECT COUNT(*) FROM {self.table_name}"  # nosec B608
            params = ()
        try:
            cur = conn.cursor()
            cur.execute(sql, params)
            return int(cur.fetchone()[0])
        except Exception as e:
            logger.error("Audit count failed: %s", e)
            return 0
        finally:
            cur.close()
            self._put_conn(conn)
