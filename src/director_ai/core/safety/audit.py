# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Structured Audit Logger

"""Structured JSON audit trail for every review decision.

Every call to ``log_review()`` produces a JSON object with timestamp,
decision, scores, policy violations, and tenant context.

Usage::

    audit = AuditLogger()              # stdout only
    audit = AuditLogger("audit.jsonl") # file sink
    audit.log_review(
        query="What is 2+2?",
        response="4",
        approved=True,
        score=0.95,
    )
"""

from __future__ import annotations

import datetime
import hashlib
import hmac as _hmac
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AuditEntry:
    """Single audit record."""

    timestamp: str
    query_hash: str
    response_length: int
    approved: bool
    score: float
    h_logical: float = 0.0
    h_factual: float = 0.0
    policy_violations: list[str] = field(default_factory=list)
    tenant_id: str = ""
    halt_reason: str = ""
    latency_ms: float = 0.0
    kb_snapshot_root: str = ""
    kb_snapshot_revision: int = 0
    kb_snapshot_record_count: int = 0
    kb_snapshot_retraction_count: int = 0
    kb_snapshot_replacement_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class AuditLogger:
    """Structured audit logger with file and logging sinks.

    Parameters
    ----------
    path : str | Path | None — JSONL file path. None = logging-only.
    logger_name : str — Python logger name for audit events.

    """

    def __init__(
        self,
        path: str | Path | None = None,
        logger_name: str = "DirectorAI.Audit",
        hmac_secret: str | None = None,
    ) -> None:
        self._path = Path(path) if path else None
        self._logger = logging.getLogger(logger_name)
        explicit = hmac_secret or os.environ.get("DIRECTOR_AUDIT_HMAC_SECRET") or ""
        if explicit:
            self._hmac_key = explicit.encode("utf-8")
        else:
            self._hmac_key = os.urandom(32)
            self._logger.warning(
                "DIRECTOR_AUDIT_HMAC_SECRET not set — query hashes "
                "will not be stable across restarts"
            )
        self._sinks: list[Any] = []
        self._file_lock = threading.Lock()
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def add_sink(self, sink: Any) -> None:
        """Add an external consumer for audit records (e.g. PostgresAuditSink)."""
        self._sinks.append(sink)

    def log_review(
        self,
        query: str,
        response: str,
        approved: bool,
        score: float,
        h_logical: float = 0.0,
        h_factual: float = 0.0,
        policy_violations: list[str] | None = None,
        tenant_id: str = "",
        halt_reason: str = "",
        latency_ms: float = 0.0,
        kb_snapshot: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record a review decision."""
        snapshot = kb_snapshot or {}
        entry = AuditEntry(
            timestamp=datetime.datetime.now(datetime.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            query_hash=_hmac.new(
                self._hmac_key,
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:16],
            response_length=len(response),
            approved=approved,
            score=round(score, 4),
            h_logical=round(h_logical, 4),
            h_factual=round(h_factual, 4),
            policy_violations=policy_violations or [],
            tenant_id=tenant_id,
            halt_reason=halt_reason,
            latency_ms=round(latency_ms, 2),
            kb_snapshot_root=str(snapshot.get("merkle_root", "")),
            kb_snapshot_revision=self._snapshot_int(snapshot, "revision"),
            kb_snapshot_record_count=self._snapshot_int(snapshot, "record_count"),
            kb_snapshot_retraction_count=self._snapshot_int(
                snapshot,
                "retraction_count",
            ),
            kb_snapshot_replacement_count=self._snapshot_int(
                snapshot,
                "replacement_count",
            ),
        )
        line = entry.to_json()
        self._logger.info(line)
        if self._path:
            with self._file_lock, open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        for sink in self._sinks:
            try:
                sink.write(entry)
            except Exception:
                self._logger.exception(
                    "Audit sink %s.write() failed", type(sink).__name__
                )

        return entry

    @staticmethod
    def _snapshot_int(snapshot: dict[str, Any], key: str) -> int:
        value = snapshot.get(key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
