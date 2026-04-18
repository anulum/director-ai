# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Feedback store — collect human corrections for online calibration.

Stores binary corrections: "the guardrail said X, the human says Y."
Each correction is a labeled training example that accumulates into a
deployment-specific calibration dataset.

Usage::

    store = FeedbackStore("feedback.db")
    store.report("What is X?", "X is Y.", guardrail_approved=True, human_approved=False)
    corrections = store.get_corrections(limit=100)
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Correction", "FeedbackStore"]


@dataclass
class Correction:
    """A single human correction to a guardrail verdict."""

    prompt: str
    response: str
    guardrail_score: float
    guardrail_approved: bool
    human_approved: bool
    timestamp: float
    domain: str = ""


class FeedbackStore:
    """Thread-safe SQLite store for human feedback on guardrail verdicts.

    Parameters
    ----------
    db_path : str | Path
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: str | Path = "feedback.db"):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn: sqlite3.Connection | None = conn
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                guardrail_score REAL NOT NULL DEFAULT 0.0,
                guardrail_approved INTEGER NOT NULL,
                human_approved INTEGER NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL
            )
        """)
        conn.commit()

    def _require_conn(self) -> sqlite3.Connection:
        """Return the live connection. Raises :class:`RuntimeError`
        when the store has already been closed."""
        if self._conn is None:
            raise RuntimeError("feedback store is closed")
        return self._conn

    def report(
        self,
        prompt: str,
        response: str,
        guardrail_approved: bool,
        human_approved: bool,
        guardrail_score: float = 0.0,
        domain: str = "",
    ) -> None:
        """Record a human correction."""
        with self._lock:
            self._require_conn().execute(
                """INSERT INTO corrections
                   (prompt, response, guardrail_score, guardrail_approved,
                    human_approved, domain, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    prompt,
                    response,
                    guardrail_score,
                    int(guardrail_approved),
                    int(human_approved),
                    domain,
                    time.time(),
                ),
            )
            self._require_conn().commit()

    def get_corrections(
        self,
        limit: int = 0,
        domain: str | None = None,
    ) -> list[Correction]:
        """Retrieve corrections, optionally filtered by domain."""
        with self._lock:
            query = "SELECT prompt, response, guardrail_score, guardrail_approved, human_approved, timestamp, domain FROM corrections"
            params: list = []
            if domain is not None:
                query += " WHERE domain = ?"
                params.append(domain)
            query += " ORDER BY timestamp DESC"
            if limit > 0:
                query += " LIMIT ?"
                params.append(limit)
            rows = self._require_conn().execute(query, params).fetchall()

        return [
            Correction(
                prompt=r[0],
                response=r[1],
                guardrail_score=r[2],
                guardrail_approved=bool(r[3]),
                human_approved=bool(r[4]),
                timestamp=r[5],
                domain=r[6],
            )
            for r in rows
        ]

    def count(self, domain: str | None = None) -> int:
        """Count total corrections."""
        with self._lock:
            if domain is not None:
                row = (
                    self._require_conn()
                    .execute(
                        "SELECT COUNT(*) FROM corrections WHERE domain = ?",
                        (domain,),
                    )
                    .fetchone()
                )
            else:
                row = (
                    self._require_conn()
                    .execute("SELECT COUNT(*) FROM corrections")
                    .fetchone()
                )
            return row[0] if row else 0

    def get_disagreements(self, limit: int = 0) -> list[Correction]:
        """Get only corrections where guardrail and human disagree."""
        with self._lock:
            query = """SELECT prompt, response, guardrail_score, guardrail_approved,
                              human_approved, timestamp, domain
                       FROM corrections
                       WHERE guardrail_approved != human_approved
                       ORDER BY timestamp DESC"""
            params: list = []
            if limit > 0:
                query += " LIMIT ?"
                params.append(limit)
            rows = self._require_conn().execute(query, params).fetchall()

        return [
            Correction(
                prompt=r[0],
                response=r[1],
                guardrail_score=r[2],
                guardrail_approved=bool(r[3]),
                human_approved=bool(r[4]),
                timestamp=r[5],
                domain=r[6],
            )
            for r in rows
        ]

    def export_training_data(self) -> list[dict]:
        """Export corrections as training data dicts for fine-tuning."""
        corrections = self.get_corrections()
        return [
            {
                "prompt": c.prompt,
                "response": c.response,
                "label": 1 if c.human_approved else 0,
                "domain": c.domain,
            }
            for c in corrections
        ]

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
