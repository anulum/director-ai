# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Human Review Queue

"""Durable human-in-the-loop review gates for halted or corrected outputs."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from director_ai.core.safety_event import SafetyEvent

from .correction import CorrectionProposal

ReviewStatus = Literal["pending", "approved", "rejected", "retry_requested", "released"]
ReviewAction = Literal["approve", "reject", "request_retry", "release"]

_VALID_STATUSES = frozenset(
    {"pending", "approved", "rejected", "retry_requested", "released"}
)
_VALID_ACTIONS = frozenset({"approve", "reject", "request_retry", "release"})

__all__ = [
    "HumanReviewCase",
    "HumanReviewDecision",
    "HumanReviewQueue",
    "ReviewAction",
    "ReviewStatus",
]


@dataclass(frozen=True)
class HumanReviewDecision:
    """Append-only reviewer decision for one queued case."""

    decision_id: str
    case_id: str
    reviewer_id: str
    action: ReviewAction
    reason: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.decision_id.strip():
            raise ValueError("decision_id is required")
        if not self.case_id.strip():
            raise ValueError("case_id is required")
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        if self.action not in _VALID_ACTIONS:
            raise ValueError(f"unsupported review action {self.action!r}")
        object.__setattr__(
            self,
            "metadata",
            {str(key): str(value) for key, value in self.metadata.items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe audit record."""
        return {
            "decision_id": self.decision_id,
            "case_id": self.case_id,
            "reviewer_id": self.reviewer_id,
            "action": self.action,
            "reason": self.reason,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class HumanReviewCase:
    """Durable case awaiting explicit reviewer decision."""

    case_id: str
    status: ReviewStatus
    source_kind: str
    candidate_text: str
    evidence_refs: Sequence[str]
    tenant_id: str = ""
    request_id: str = ""
    reason: str = ""
    safety_event: Mapping[str, Any] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    release_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id is required")
        if self.status not in _VALID_STATUSES:
            raise ValueError(f"unsupported review status {self.status!r}")
        if not self.source_kind.strip():
            raise ValueError("source_kind is required")
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(ref for ref in map(str, self.evidence_refs) if ref.strip()),
        )
        object.__setattr__(
            self,
            "metadata",
            {str(key): str(value) for key, value in self.metadata.items()},
        )

    def to_dict(self, *, include_candidate: bool = False) -> dict[str, Any]:
        """Serialise a tenant-safe case payload.

        Candidate text is excluded by default because review queues commonly
        feed dashboards, alerts, and audit exports outside the tenant boundary.
        """
        payload: dict[str, Any] = {
            "case_id": self.case_id,
            "status": self.status,
            "source_kind": self.source_kind,
            "tenant_id": self.tenant_id,
            "request_id": self.request_id,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "safety_event": dict(self.safety_event) if self.safety_event else None,
            "metadata": dict(self.metadata),
            "release_id": self.release_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_candidate:
            payload["candidate_text"] = self.candidate_text
        return payload


class HumanReviewQueue:
    """SQLite-backed human review queue with gated release and retry paths."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema(self._conn)

    def enqueue_case(
        self,
        *,
        candidate_text: str,
        evidence_refs: Sequence[str],
        tenant_id: str = "",
        request_id: str = "",
        source_kind: str = "halt",
        reason: str = "",
        safety_event: SafetyEvent | Mapping[str, Any] | None = None,
        metadata: Mapping[str, str] | None = None,
        case_id: str | None = None,
    ) -> HumanReviewCase:
        """Add a pending review case."""
        if not candidate_text.strip():
            raise ValueError("candidate_text is required")
        evidence = tuple(ref for ref in map(str, evidence_refs) if ref.strip())
        if not evidence:
            raise ValueError("evidence_refs are required")
        now = time.time()
        case = HumanReviewCase(
            case_id=case_id or f"hrev_{uuid4().hex}",
            status="pending",
            source_kind=source_kind,
            candidate_text=candidate_text,
            evidence_refs=evidence,
            tenant_id=tenant_id,
            request_id=request_id,
            reason=reason,
            safety_event=_safety_event_payload(safety_event),
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            conn = self._require_conn()
            conn.execute(
                """INSERT INTO human_review_cases
                   (case_id, status, source_kind, candidate_text, evidence_refs,
                    tenant_id, request_id, reason, safety_event, metadata,
                    release_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                _case_row(case),
            )
            conn.commit()
        return case

    def enqueue_correction_proposal(
        self,
        proposal: CorrectionProposal,
        *,
        tenant_id: str = "",
        request_id: str = "",
        reason: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> HumanReviewCase:
        """Queue a correction proposal for human release approval."""
        proposal_payload = proposal.to_dict(include_candidate=True)
        merged_metadata = {
            "proposal_id": proposal.proposal_id,
            "guard_decision": proposal.guard_decision.decision,
            **{str(k): str(v) for k, v in (metadata or {}).items()},
        }
        return self.enqueue_case(
            candidate_text=str(proposal_payload["candidate_text"]),
            evidence_refs=proposal.evidence_refs,
            tenant_id=tenant_id,
            request_id=request_id,
            source_kind="correction",
            reason=reason or proposal.guard_decision.reason,
            metadata=merged_metadata,
        )

    def get_case(self, case_id: str) -> HumanReviewCase:
        """Return one review case by id."""
        with self._lock:
            row = (
                self._require_conn()
                .execute(
                    "SELECT * FROM human_review_cases WHERE case_id = ?",
                    (case_id,),
                )
                .fetchone()
            )
        if row is None:
            raise KeyError(case_id)
        return _case_from_row(row)

    def list_cases(
        self,
        *,
        status: ReviewStatus | None = None,
        tenant_id: str | None = None,
        limit: int = 0,
    ) -> list[HumanReviewCase]:
        """List cases, newest first, optionally filtered by status or tenant."""
        clauses: list[str] = []
        params: list[Any] = []
        if status is not None:
            if status not in _VALID_STATUSES:
                raise ValueError(f"unsupported review status {status!r}")
            clauses.append("status = ?")
            params.append(status)
        if tenant_id is not None:
            clauses.append("tenant_id = ?")
            params.append(tenant_id)
        query = "SELECT * FROM human_review_cases"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC"
        if limit > 0:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._require_conn().execute(query, params).fetchall()
        return [_case_from_row(row) for row in rows]

    def decide(
        self,
        case_id: str,
        *,
        reviewer_id: str,
        action: str,
        reason: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> HumanReviewCase:
        """Apply a reviewer decision and append it to the audit trail."""
        if action not in {"approve", "reject", "request_retry"}:
            raise ValueError("action must be approve, reject, or request_retry")
        if not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        with self._lock:
            conn = self._require_conn()
            case = self._get_case_locked(case_id)
            if case.status == "released":
                raise PermissionError("review case is already released")
            if case.status in {"approved", "rejected", "retry_requested"}:
                raise PermissionError(f"review case is already {case.status}")
            status = _status_for_action(action)
            now = time.time()
            decision = HumanReviewDecision(
                decision_id=f"hdec_{uuid4().hex}",
                case_id=case_id,
                reviewer_id=reviewer_id,
                action=cast(ReviewAction, action),
                reason=reason,
                metadata=metadata or {},
                timestamp=now,
            )
            self._insert_decision_locked(conn, decision)
            conn.execute(
                """UPDATE human_review_cases
                   SET status = ?, updated_at = ?
                   WHERE case_id = ?""",
                (status, now, case_id),
            )
            conn.commit()
            return self._get_case_locked(case_id)

    def release(self, case_id: str, *, reviewer_id: str, release_id: str) -> str:
        """Return candidate text only after a case has been approved."""
        if not reviewer_id.strip():
            raise ValueError("reviewer_id is required")
        if not release_id.strip():
            raise ValueError("release_id is required")
        with self._lock:
            conn = self._require_conn()
            case = self._get_case_locked(case_id)
            if case.status == "released":
                raise PermissionError("review case is already released")
            if case.status != "approved":
                raise PermissionError("review case is not approved")
            now = time.time()
            decision = HumanReviewDecision(
                decision_id=f"hdec_{uuid4().hex}",
                case_id=case_id,
                reviewer_id=reviewer_id,
                action="release",
                reason="released approved candidate",
                metadata={"release_id": release_id},
                timestamp=now,
            )
            self._insert_decision_locked(conn, decision)
            conn.execute(
                """UPDATE human_review_cases
                   SET status = 'released', release_id = ?, updated_at = ?
                   WHERE case_id = ?""",
                (release_id, now, case_id),
            )
            conn.commit()
            return case.candidate_text

    def retry_payload(self, case_id: str) -> dict[str, Any]:
        """Return tenant-safe retry instructions only after retry approval."""
        case = self.get_case(case_id)
        if case.status != "retry_requested":
            raise PermissionError("retry was not requested for this review case")
        retry_decisions = [
            decision
            for decision in self.decisions(case_id)
            if decision.action == "request_retry"
        ]
        decision = retry_decisions[-1]
        return {
            "case_id": case.case_id,
            "tenant_id": case.tenant_id,
            "request_id": case.request_id,
            "evidence_refs": list(case.evidence_refs),
            "reason": decision.reason,
            **decision.metadata,
        }

    def decisions(self, case_id: str) -> list[HumanReviewDecision]:
        """Return append-only decision history for one case."""
        with self._lock:
            rows = (
                self._require_conn()
                .execute(
                    """SELECT decision_id, case_id, reviewer_id, action, reason,
                          metadata, timestamp
                   FROM human_review_decisions
                   WHERE case_id = ?
                   ORDER BY timestamp ASC""",
                    (case_id,),
                )
                .fetchall()
            )
        return [_decision_from_row(row) for row in rows]

    def close(self) -> None:
        """Close the backing database. Safe to call multiple times."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("human review queue is closed")
        return self._conn

    def _get_case_locked(self, case_id: str) -> HumanReviewCase:
        row = (
            self._require_conn()
            .execute(
                "SELECT * FROM human_review_cases WHERE case_id = ?",
                (case_id,),
            )
            .fetchone()
        )
        if row is None:
            raise KeyError(case_id)
        return _case_from_row(row)

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_review_cases (
                case_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                candidate_text TEXT NOT NULL,
                evidence_refs TEXT NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT '',
                request_id TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                safety_event TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                release_id TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS human_review_decisions (
                decision_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                timestamp REAL NOT NULL,
                FOREIGN KEY(case_id) REFERENCES human_review_cases(case_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_human_review_cases_status
            ON human_review_cases(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_human_review_cases_tenant
            ON human_review_cases(tenant_id)
        """)
        conn.commit()

    @staticmethod
    def _insert_decision_locked(
        conn: sqlite3.Connection,
        decision: HumanReviewDecision,
    ) -> None:
        conn.execute(
            """INSERT INTO human_review_decisions
               (decision_id, case_id, reviewer_id, action, reason, metadata, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.decision_id,
                decision.case_id,
                decision.reviewer_id,
                decision.action,
                decision.reason,
                _json_dumps(decision.metadata),
                decision.timestamp,
            ),
        )


def _case_row(case: HumanReviewCase) -> tuple[Any, ...]:
    return (
        case.case_id,
        case.status,
        case.source_kind,
        case.candidate_text,
        _json_dumps(list(case.evidence_refs)),
        case.tenant_id,
        case.request_id,
        case.reason,
        _json_dumps(case.safety_event) if case.safety_event else "",
        _json_dumps(case.metadata),
        case.release_id,
        case.created_at,
        case.updated_at,
    )


def _case_from_row(row: sqlite3.Row | tuple[Any, ...]) -> HumanReviewCase:
    return HumanReviewCase(
        case_id=row[0],
        status=row[1],
        source_kind=row[2],
        candidate_text=row[3],
        evidence_refs=tuple(json.loads(row[4] or "[]")),
        tenant_id=row[5],
        request_id=row[6],
        reason=row[7],
        safety_event=json.loads(row[8]) if row[8] else None,
        metadata=json.loads(row[9] or "{}"),
        release_id=row[10],
        created_at=float(row[11]),
        updated_at=float(row[12]),
    )


def _decision_from_row(row: sqlite3.Row | tuple[Any, ...]) -> HumanReviewDecision:
    return HumanReviewDecision(
        decision_id=row[0],
        case_id=row[1],
        reviewer_id=row[2],
        action=row[3],
        reason=row[4],
        metadata=json.loads(row[5] or "{}"),
        timestamp=float(row[6]),
    )


def _safety_event_payload(
    event: SafetyEvent | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if event is None:
        return None
    if isinstance(event, SafetyEvent):
        return event.to_dict()
    return dict(event)


def _status_for_action(action: str) -> ReviewStatus:
    if action == "approve":
        return "approved"
    if action == "reject":
        return "rejected"
    if action == "request_retry":
        return "retry_requested"
    raise ValueError(f"unsupported review action {action!r}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
