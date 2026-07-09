# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Fine-tuning job records and persistent stores

"""Job records and persistent stores for the fine-tuning API.

``_JobStore`` (local fine-tuning jobs) and ``_ManagedJobStore`` (managed
training submissions) keep live records in memory and, when constructed with
a database path, write every mutation through to SQLite (WAL journal), so job
state — including ``activated`` designations that protect models from
deletion — survives server restarts and is visible across worker processes.

Constructed without a path the stores are ephemeral, which is the degraded
fallback for read-only filesystems and the mode unit tests use. On opening a
database, local jobs left in a non-terminal state by a previous process are
marked ``failed`` (``interrupted by restart``): their worker threads are gone,
so resurrecting them as live would wedge the concurrency cap.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("DirectorAI.FinetuneJobs")

_MAX_CONCURRENT_JOBS = 4

#: Local-job states that count against the concurrency cap.
_ACTIVE_STATES = ("validating", "training", "benchmarking")
#: Local-job states that no longer change.
_TERMINAL_STATES = ("completed", "failed")

_INTERRUPTED_ERROR = "interrupted by restart"


@dataclass
class FinetuneJob:
    """In-process state for one local fine-tuning job."""

    job_id: str
    state: str = (
        "pending"  # pending, validating, training, benchmarking, completed, failed
    )
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    validation_report: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    regression_report: dict[str, Any] = field(default_factory=dict)
    model_path: str = ""
    error: str = ""
    created_at: float = 0.0
    completed_at: float = 0.0
    activated: bool = False


@dataclass
class ManagedTrainingRecord:
    """Ledger entry for one managed training submission."""

    job_id: str
    backend: str
    state: str
    tenant_id: str
    dry_run: bool
    submitted_at: float
    display_name: str
    output_uri: str
    console_uri: str = ""
    error: str = ""


def _open_job_db(db_path: str) -> None:
    """Create the job tables when absent (idempotent)."""
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        with conn:  # one transaction; ``with`` commits but does not close
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS finetune_jobs ("
                "job_id TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS managed_jobs ("
                "job_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, "
                "payload TEXT NOT NULL)"
            )
    finally:
        conn.close()


class _SqliteBackedStore:
    """Shared SQLite plumbing for the two job stores.

    Subclasses hold live records in memory (the write owners within one
    process) and call :meth:`_db_write` after every mutation, so the database
    stays the durable, cross-process source of truth. A ``None`` database
    path disables persistence.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._lock = threading.Lock()
        self._db_path: str | None = None
        if db_path is not None:
            try:
                _open_job_db(str(db_path))
                self._db_path = str(db_path)
            except sqlite3.Error as exc:
                logger.warning(
                    "Job persistence unavailable at %s (%s); "
                    "falling back to an in-memory store",
                    db_path,
                    exc,
                )

    @property
    def persistent(self) -> bool:
        """Whether mutations are written through to SQLite."""
        return self._db_path is not None

    @staticmethod
    def _connect(db_path: str) -> sqlite3.Connection:
        """Open a short-lived connection (WAL, 5 s busy timeout)."""
        conn = sqlite3.connect(db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _db_write(self, sql: str, params: tuple[Any, ...]) -> None:
        """Execute one write statement when persistence is enabled."""
        if self._db_path is None:
            return
        try:
            conn = self._connect(self._db_path)
            try:
                with conn:  # commit-on-success
                    conn.execute(sql, params)
            finally:
                conn.close()
        except sqlite3.Error as exc:  # pragma: no cover - disk-level failure
            logger.error("Job store write failed: %s", exc)

    def _db_rows(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        """Return rows for a query, or an empty list without persistence."""
        if self._db_path is None:
            return []
        try:
            conn = self._connect(self._db_path)
            try:
                return list(conn.execute(sql, params))
            finally:
                conn.close()
        except sqlite3.Error as exc:  # pragma: no cover - disk-level failure
            logger.error("Job store read failed: %s", exc)
            return []


class _JobStore(_SqliteBackedStore):
    """Thread-safe local fine-tuning job store with optional persistence.

    Parameters
    ----------
    db_path : str | Path | None
        SQLite database file. ``None`` keeps the store in memory only.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        super().__init__(db_path)
        self._jobs: dict[str, FinetuneJob] = {}
        if self.persistent:
            self._recover_interrupted()

    def _recover_interrupted(self) -> None:
        """Fail-close local jobs a previous process left mid-flight."""
        rows = self._db_rows(
            "SELECT job_id, payload FROM finetune_jobs WHERE state NOT IN (?, ?)",
            _TERMINAL_STATES,
        )
        for job_id, payload in rows:
            data = json.loads(payload)
            data["state"] = "failed"
            data["error"] = _INTERRUPTED_ERROR
            self._db_write(
                "UPDATE finetune_jobs SET state = ?, payload = ? WHERE job_id = ?",
                ("failed", json.dumps(data), job_id),
            )
            logger.warning("Job %s from a previous run marked failed", job_id)

    def save(self, job: FinetuneJob) -> None:
        """Write a job's current state through to the database."""
        self._db_write(
            "INSERT INTO finetune_jobs (job_id, state, payload) VALUES (?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET state = excluded.state, "
            "payload = excluded.payload",
            (job.job_id, job.state, json.dumps(asdict(job))),
        )

    def _active_count(self) -> int:
        """Count in-flight jobs (cross-process when persistent)."""
        if self.persistent:
            rows = self._db_rows(
                "SELECT COUNT(*) FROM finetune_jobs WHERE state IN (?, ?, ?)",
                _ACTIVE_STATES,
            )
            return int(rows[0][0]) if rows else 0
        return sum(1 for j in self._jobs.values() if j.state in _ACTIVE_STATES)

    def create(self, config: dict[str, Any]) -> FinetuneJob:
        """Create a job unless the concurrency cap is reached."""
        with self._lock:
            active = self._active_count()
            if active >= _MAX_CONCURRENT_JOBS:
                raise ValueError(
                    f"Too many concurrent jobs ({active}/{_MAX_CONCURRENT_JOBS})",
                )
            job = FinetuneJob(
                job_id=uuid.uuid4().hex,
                config=config,
                created_at=time.time(),
            )
            self._jobs[job.job_id] = job
            self.save(job)
        return job

    def get(self, job_id: str) -> FinetuneJob | None:
        """Return a job by id, or ``None`` when it is unknown.

        Live in-process records win; otherwise the record is loaded from the
        database (jobs from a previous run or another worker) and cached so
        subsequent mutations act on one object.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return job
            rows = self._db_rows(
                "SELECT payload FROM finetune_jobs WHERE job_id = ?", (job_id,)
            )
            if not rows:
                return None
            loaded = FinetuneJob(**json.loads(rows[0][0]))
            self._jobs[job_id] = loaded
            return loaded

    def list_all(self) -> list[FinetuneJob]:
        """Return a snapshot of all known jobs (persisted ones included)."""
        with self._lock:
            merged: dict[str, FinetuneJob] = {}
            for job_id, payload in self._db_rows(
                "SELECT job_id, payload FROM finetune_jobs"
            ):
                merged[job_id] = FinetuneJob(**json.loads(payload))
            merged.update(self._jobs)
            return list(merged.values())

    def delete(self, job_id: str) -> bool:
        """Delete a job record and report whether it existed."""
        with self._lock:
            existed_in_db = bool(
                self._db_rows("SELECT 1 FROM finetune_jobs WHERE job_id = ?", (job_id,))
            )
            self._db_write("DELETE FROM finetune_jobs WHERE job_id = ?", (job_id,))
            return self._jobs.pop(job_id, None) is not None or existed_in_db


class _ManagedJobStore(_SqliteBackedStore):
    """Thread-safe ledger for managed training submissions.

    Parameters
    ----------
    db_path : str | Path | None
        SQLite database file. ``None`` keeps the ledger in memory only.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        super().__init__(db_path)
        self._jobs: dict[str, ManagedTrainingRecord] = {}

    def _save(self, record: ManagedTrainingRecord) -> None:
        self._db_write(
            "INSERT INTO managed_jobs (job_id, tenant_id, payload) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(job_id) DO UPDATE SET tenant_id = excluded.tenant_id, "
            "payload = excluded.payload",
            (record.job_id, record.tenant_id, json.dumps(asdict(record))),
        )

    def add(self, record: ManagedTrainingRecord) -> None:
        """Store or replace a managed-training record by job id."""
        with self._lock:
            self._jobs[record.job_id] = record
            self._save(record)

    def _load(self, job_id: str) -> ManagedTrainingRecord | None:
        """Return the live record, falling back to the database."""
        record = self._jobs.get(job_id)
        if record is not None:
            return record
        rows = self._db_rows(
            "SELECT payload FROM managed_jobs WHERE job_id = ?", (job_id,)
        )
        if not rows:
            return None
        loaded = ManagedTrainingRecord(**json.loads(rows[0][0]))
        self._jobs[job_id] = loaded
        return loaded

    def get(self, tenant_id: str, job_id: str) -> ManagedTrainingRecord | None:
        """Return a tenant-owned managed-training record."""
        with self._lock:
            record = self._load(job_id)
        if record is None or record.tenant_id != tenant_id:
            return None
        return record

    def list_for_tenant(self, tenant_id: str) -> list[ManagedTrainingRecord]:
        """Return all managed-training records visible to one tenant."""
        with self._lock:
            merged: dict[str, ManagedTrainingRecord] = {}
            for (payload,) in self._db_rows(
                "SELECT payload FROM managed_jobs WHERE tenant_id = ?",
                (tenant_id,),
            ):
                record = ManagedTrainingRecord(**json.loads(payload))
                merged[record.job_id] = record
            for job_id, record in self._jobs.items():
                if record.tenant_id == tenant_id:
                    merged[job_id] = record
            return list(merged.values())

    def update_state(
        self,
        tenant_id: str,
        job_id: str,
        state: str,
        *,
        error: str = "",
    ) -> ManagedTrainingRecord | None:
        """Update a tenant-owned record state and optional error."""
        with self._lock:
            record = self._load(job_id)
            if record is None or record.tenant_id != tenant_id:
                return None
            record.state = state
            record.error = error
            self._save(record)
            return record
