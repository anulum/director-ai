# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Structured Audit Logger
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Structured JSON audit trail for every review decision.

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

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


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
    ) -> None:
        self._path = Path(path) if path else None
        self._logger = logging.getLogger(logger_name)
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

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
    ) -> AuditEntry:
        """Record a review decision."""
        entry = AuditEntry(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
            response_length=len(response),
            approved=approved,
            score=round(score, 4),
            h_logical=round(h_logical, 4),
            h_factual=round(h_factual, 4),
            policy_violations=policy_violations or [],
            tenant_id=tenant_id,
            halt_reason=halt_reason,
            latency_ms=round(latency_ms, 2),
        )
        line = entry.to_json()
        self._logger.info(line)
        if self._path:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return entry
