# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
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

import hashlib
import hmac as _hmac
import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

__all__ = ["AuditEntry", "AuditLog"]

_ZERO_HASH = "0" * 64
# Entry columns that carry the tamper-evident chain itself, excluded from the
# content hash so it covers only the interaction payload. Mirrors the sealing
# scheme proven in ``core/safety/audit.py`` (SHA-256 content hash + prev_hash
# linkage + HMAC chain tag) applied to this SQLite compliance trail.
_CHAIN_COLUMNS = ("prev_hash", "entry_hash", "chain_tag")

# DDL below is composed by f-string; every identifier must satisfy this
# pattern so a future caller cannot smuggle SQL through a column name.
_SQL_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

_logger = logging.getLogger("DirectorAI.ComplianceAudit")


class _Redactor(Protocol):
    """Structural type for a PII redactor (satisfied by ``core.redactor``).

    Only ``redact`` is required so the audit log stays dependency-light and
    never imports the redaction pipeline directly.
    """

    def redact(self, text: str) -> str:
        """Return ``text`` with any detected PII spans masked."""
        ...


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

    When a ``redactor`` is supplied (opt-in; enabled by ``redact_pii`` at the
    wiring layer) the prompt and response are masked **before** they are sealed
    into the tamper-evident chain, so raw PII never touches disk and the seal
    covers exactly the redacted content that is stored. The default (no
    redactor) preserves the raw compliance trail unchanged.

    ``strict_mode`` (KIMI2-C) makes the confidentiality posture fail-closed
    for regulated deployments: construction raises unless a ``redactor`` is
    supplied (or raw storage is explicitly acknowledged with
    ``allow_raw=True``), and raises when no durable HMAC secret is available
    (a per-process random key would make the seal unverifiable across
    restarts). The non-strict default keeps today's behaviour.
    """

    def __init__(
        self,
        db_path: str | Path = "director_audit.db",
        hmac_secret: str | None = None,
        redactor: _Redactor | None = None,
        strict_mode: bool = False,
        allow_raw: bool = False,
    ):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._redactor = redactor
        if strict_mode and redactor is None and not allow_raw:
            raise ValueError(
                "strict_mode requires a redactor so raw prompts/responses "
                "never reach the durable audit trail; pass allow_raw=True "
                "to explicitly store raw content anyway",
            )
        explicit = hmac_secret or os.environ.get("DIRECTOR_AUDIT_HMAC_SECRET") or ""
        if explicit:
            self._hmac_key = explicit.encode("utf-8")
        elif strict_mode:
            raise ValueError(
                "strict_mode requires a durable HMAC secret "
                "(hmac_secret= or DIRECTOR_AUDIT_HMAC_SECRET) — a per-process "
                "random key would make the audit seal unverifiable across "
                "restarts",
            )
        else:
            self._hmac_key = os.urandom(32)
            _logger.warning(
                "DIRECTOR_AUDIT_HMAC_SECRET not set — audit chain tags will not "
                "be reproducible across restarts; set it to make the seal durable"
            )
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
                timestamp REAL NOT NULL,
                prev_hash TEXT,
                entry_hash TEXT,
                chain_tag TEXT
            )
        """)
        # Older databases predate the tamper-evident chain columns; add them so
        # the seal can extend an existing compliance trail (legacy rows stay
        # unsealed and are skipped by verify_chain).
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(audit_log)")
        }
        for column in _CHAIN_COLUMNS:
            if not _SQL_IDENTIFIER_RE.fullmatch(column):
                raise ValueError(f"unsafe SQL column identifier: {column!r}")
            if column not in existing:
                self._conn.execute(f"ALTER TABLE audit_log ADD COLUMN {column} TEXT")
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_model ON audit_log(model)
        """)
        self._conn.commit()
        self._prev_hash = self._load_chain_head()

    def log(self, entry: AuditEntry) -> None:
        """Record a scored LLM interaction, sealed into the tamper-evident chain.

        The parent-hash read, chain update and insert run under the lock so a
        concurrent ``log`` cannot fork the on-disk chain. When a redactor is
        configured, the prompt and response are masked first — the caller's
        ``entry`` is left unmodified and the redacted copy is what gets hashed,
        stored, and sealed.
        """
        with self._lock:
            if self._conn is None:
                return
            if self._redactor is not None:
                entry = replace(
                    entry,
                    prompt=self._redactor.redact(entry.prompt),
                    response=self._redactor.redact(entry.response),
                )
            prev_hash = self._prev_hash
            entry_hash = self._content_hash(entry, prev_hash)
            chain_tag = self._chain_tag(prev_hash, entry_hash)
            self._conn.execute(
                """INSERT INTO audit_log
                   (prompt, response, model, provider, score, approved,
                    verdict_confidence, task_type, domain, latency_ms,
                    tenant_id, human_override, timestamp,
                    prev_hash, entry_hash, chain_tag)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    prev_hash,
                    entry_hash,
                    chain_tag,
                ),
            )
            self._conn.commit()
            self._prev_hash = entry_hash

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
            params: list[Any] = []
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
            params: list[Any] = []
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

    @staticmethod
    def _canonical_payload(
        *,
        prompt: str,
        response: str,
        model: str,
        provider: str,
        score: float,
        approved: int,
        verdict_confidence: float,
        task_type: str,
        domain: str,
        latency_ms: float,
        tenant_id: str,
        human_override: int | None,
        timestamp: float,
        prev_hash: str,
    ) -> str:
        """SHA-256 over the canonical stored interaction payload plus the parent hash.

        Built from the persisted primitive forms (``approved``/``human_override``
        as integers) so the log-time and verify-time digests are byte-identical.
        """
        content = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "provider": provider,
            "score": score,
            "approved": approved,
            "verdict_confidence": verdict_confidence,
            "task_type": task_type,
            "domain": domain,
            "latency_ms": latency_ms,
            "tenant_id": tenant_id,
            "human_override": human_override,
            "timestamp": timestamp,
            "prev_hash": prev_hash,
        }
        payload = json.dumps(content, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _content_hash(self, entry: AuditEntry, prev_hash: str) -> str:
        """Content hash for an in-memory entry about to be sealed."""
        return self._canonical_payload(
            prompt=entry.prompt,
            response=entry.response,
            model=entry.model,
            provider=entry.provider,
            score=entry.score,
            approved=int(entry.approved),
            verdict_confidence=entry.verdict_confidence,
            task_type=entry.task_type,
            domain=entry.domain,
            latency_ms=entry.latency_ms,
            tenant_id=entry.tenant_id,
            human_override=(
                int(entry.human_override) if entry.human_override is not None else None
            ),
            timestamp=entry.timestamp,
            prev_hash=prev_hash,
        )

    def _chain_tag(self, prev_hash: str, entry_hash: str) -> str:
        """HMAC-SHA-256 over ``prev_hash + entry_hash`` keyed on the audit secret."""
        message = (prev_hash + entry_hash).encode("utf-8")
        return _hmac.new(self._hmac_key, message, hashlib.sha256).hexdigest()

    def current_head(self) -> str:
        """Return the current chain head (latest sealed ``entry_hash``).

        The genesis zero hash when the chain is empty. This is the digest an
        external timestamp anchor commits to (see
        :mod:`director_ai.compliance.timestamp_anchor`); through the ``prev_hash``
        linkage it commits to the whole sealed history up to this point.
        """
        with self._lock:
            return self._prev_hash

    def _load_chain_head(self) -> str:
        """Return the most recent sealed ``entry_hash``, or the genesis zero hash."""
        if self._conn is None:
            return _ZERO_HASH
        row = self._conn.execute(
            "SELECT entry_hash FROM audit_log WHERE entry_hash IS NOT NULL "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row and row[0] else _ZERO_HASH

    def verify_chain(self) -> tuple[bool, int | None]:
        """Re-derive the sealed hash chain and report the first tampered row.

        Returns ``(ok, first_bad_id)``. ``ok`` is ``False`` when any row's
        payload was altered (content-hash mismatch), the linkage was broken (a
        row deleted, reordered, or its ``prev_hash`` edited), or a tag fails
        (HMAC mismatch — needs the same secret that wrote the log). Rows written
        before sealing was enabled (NULL ``entry_hash``) are skipped.
        """
        with self._lock:
            if self._conn is None:
                return True, None
            rows = self._conn.execute(
                "SELECT id, prompt, response, model, provider, score, approved, "
                "verdict_confidence, task_type, domain, latency_ms, tenant_id, "
                "human_override, timestamp, prev_hash, entry_hash, chain_tag "
                "FROM audit_log WHERE entry_hash IS NOT NULL ORDER BY id ASC"
            ).fetchall()
        previous_hash = _ZERO_HASH
        for r in rows:
            row_id = r[0]
            prev_hash = r[14]
            entry_hash = r[15]
            chain_tag = r[16]
            recomputed = self._canonical_payload(
                prompt=r[1],
                response=r[2],
                model=r[3],
                provider=r[4],
                score=r[5],
                approved=r[6],
                verdict_confidence=r[7],
                task_type=r[8],
                domain=r[9],
                latency_ms=r[10],
                tenant_id=r[11],
                human_override=r[12],
                timestamp=r[13],
                prev_hash=prev_hash,
            )
            if recomputed != entry_hash:
                return False, row_id
            if prev_hash != previous_hash:
                return False, row_id
            expected_tag = self._chain_tag(prev_hash, entry_hash)
            if not _hmac.compare_digest(expected_tag, str(chain_tag or "")):
                return False, row_id
            previous_hash = entry_hash
        return True, None

    def close(self) -> None:
        """Close the database. Safe to call multiple times."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
