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

_ZERO_HASH = "0" * 64
# Fields that carry the tamper-evident chain itself, excluded from the content
# hash so the hash covers only the decision payload.
_CHAIN_FIELDS = ("prev_hash", "entry_hash", "chain_tag")


@dataclass
class AuditEntry:
    """Single audit record.

    The trailing ``prev_hash`` / ``entry_hash`` / ``chain_tag`` fields form a
    tamper-evident hash chain: ``entry_hash`` is SHA-256 over the decision
    payload, ``prev_hash`` links to the previous entry's ``entry_hash``, and
    ``chain_tag`` is an HMAC over ``prev_hash + entry_hash`` keyed on the
    logger's secret. Editing, deleting, or reordering any record breaks the
    chain — see :meth:`AuditLogger.verify_chain`.
    """

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
    prev_hash: str = ""
    entry_hash: str = ""
    chain_tag: str = ""

    def to_json(self) -> str:
        """Serialise the audit entry as compact JSON for JSONL sinks."""
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
        # Continue the hash chain from an existing file so restarts do not
        # break verification (stable tags also need DIRECTOR_AUDIT_HMAC_SECRET).
        self._prev_hash = self._load_last_hash()

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
        # Seal the entry into the hash chain and persist it atomically: the
        # parent-hash read, the chain update, and the append must not interleave
        # with a concurrent log_review, or the on-disk chain would fork.
        with self._file_lock:
            entry.prev_hash = self._prev_hash
            entry.entry_hash = self._content_hash(entry)
            entry.chain_tag = self._chain_tag(entry.prev_hash, entry.entry_hash)
            self._prev_hash = entry.entry_hash
            line = entry.to_json()
            if self._path:
                with open(self._path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

        self._logger.info(line)

        for sink in self._sinks:
            try:
                sink.write(entry)
            except Exception:
                self._logger.exception(
                    "Audit sink %s.write() failed", type(sink).__name__
                )

        return entry

    @staticmethod
    def _content_hash(entry: AuditEntry) -> str:
        """SHA-256 over the decision payload (chain fields excluded)."""
        content = {k: v for k, v in asdict(entry).items() if k not in _CHAIN_FIELDS}
        payload = json.dumps(content, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    def _chain_tag(self, prev_hash: str, entry_hash: str) -> str:
        """HMAC-SHA-256 over ``prev_hash + entry_hash`` keyed on the secret."""
        message = (prev_hash + entry_hash).encode("utf-8")
        return _hmac.new(self._hmac_key, message, hashlib.sha256).hexdigest()

    def _load_last_hash(self) -> str:
        """Return the last persisted ``entry_hash`` (chain head), or the zero hash."""
        if not self._path or not self._path.exists():
            return _ZERO_HASH
        last = ""
        with open(self._path, encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    last = raw
        if not last:
            return _ZERO_HASH
        try:
            value = json.loads(last).get("entry_hash", "")
        except json.JSONDecodeError:
            return _ZERO_HASH
        return value or _ZERO_HASH

    def verify_chain(self, path: str | Path | None = None) -> tuple[bool, int | None]:
        """Re-derive the on-disk hash chain and report the first tampered entry.

        Returns ``(ok, first_bad_index)``. ``ok`` is ``False`` when any record's
        payload was altered (content hash mismatch), the chain linkage was broken
        (a record deleted, reordered, or its ``prev_hash`` edited), or a tag fails
        (HMAC mismatch — needs the same secret that wrote the log).
        ``first_bad_index`` is the 0-based line index, ``None`` on success.
        """
        target = Path(path) if path else self._path
        if target is None:
            raise ValueError("verify_chain needs a path (logger has no file sink)")
        previous_hash = _ZERO_HASH
        index = 0
        with open(target, encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                record = json.loads(raw)
                content = {k: v for k, v in record.items() if k not in _CHAIN_FIELDS}
                payload = json.dumps(
                    content, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                entry_hash = hashlib.sha256(payload).hexdigest()
                if entry_hash != record.get("entry_hash"):
                    return False, index
                if record.get("prev_hash") != previous_hash:
                    return False, index
                expected_tag = self._chain_tag(previous_hash, entry_hash)
                if not _hmac.compare_digest(
                    expected_tag, str(record.get("chain_tag", ""))
                ):
                    return False, index
                previous_hash = entry_hash
                index += 1
        return True, None

    @staticmethod
    def _snapshot_int(snapshot: dict[str, Any], key: str) -> int:
        value = snapshot.get(key, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
