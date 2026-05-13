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
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

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
    review_id: str = ""
    tenant_id: str = ""


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
                review_id TEXT NOT NULL DEFAULT '',
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                guardrail_score REAL NOT NULL DEFAULT 0.0,
                guardrail_approved INTEGER NOT NULL,
                human_approved INTEGER NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                tenant_id TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL
            )
        """)
        _ensure_column(conn, "review_id", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "tenant_id", "TEXT NOT NULL DEFAULT ''")
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
        review_id: str = "",
        tenant_id: str = "",
    ) -> None:
        """Record a human correction."""
        with self._lock:
            self._require_conn().execute(
                """INSERT INTO corrections
                   (review_id, prompt, response, guardrail_score, guardrail_approved,
                    human_approved, domain, tenant_id, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id,
                    prompt,
                    response,
                    guardrail_score,
                    int(guardrail_approved),
                    int(human_approved),
                    domain,
                    tenant_id,
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
            query = (
                "SELECT prompt, response, guardrail_score, guardrail_approved, "
                "human_approved, timestamp, domain, review_id, tenant_id "
                "FROM corrections"
            )
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
                review_id=r[7],
                tenant_id=r[8],
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
                              human_approved, timestamp, domain, review_id, tenant_id
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
                review_id=r[7],
                tenant_id=r[8],
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
                "review_id": c.review_id,
                "tenant_id": c.tenant_id,
            }
            for c in corrections
        ]

    def export_calibration_rows(
        self,
        limit: int = 0,
        domain: str | None = None,
        *,
        include_text: bool = True,
    ) -> list[dict]:
        """Export canonical calibration rows for analytics and MLOps.

        The schema is intentionally explicit and versioned so downstream
        systems can consume the dataset without inferring boolean semantics
        from historical training-export fields.
        """
        rows = []
        for correction in self.get_corrections(limit=limit, domain=domain):
            row = {
                "schema_version": "director-ai.calibration-feedback.v1",
                "prompt": correction.prompt if include_text else "",
                "response": correction.response if include_text else "",
                "guardrail_score": float(correction.guardrail_score),
                "guardrail_approved": bool(correction.guardrail_approved),
                "human_approved": bool(correction.human_approved),
                "label": 1 if correction.human_approved else 0,
                "disagreement": correction.guardrail_approved
                != correction.human_approved,
                "domain": correction.domain,
                "review_id": correction.review_id,
                "tenant_id": correction.tenant_id,
                "timestamp": float(correction.timestamp),
            }
            rows.append(row)
        return rows

    def export_parquet(
        self,
        path: str | Path,
        limit: int = 0,
        domain: str | None = None,
        *,
        include_text: bool = True,
    ) -> Path:
        """Export calibration feedback to Parquet with an optional dependency.

        Requires ``pyarrow`` only at call time. Writes to a same-directory
        temporary file first and atomically replaces the final path.
        """
        try:
            pa = import_module("pyarrow")
            pq = import_module("pyarrow.parquet")
            table_cls = pa.Table
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "FeedbackStore.export_parquet() requires the optional pyarrow "
                "package. Install a deployment extra that provides pyarrow.",
            ) from exc
        if pa is None or pq is None:  # supports tests that mask optional modules
            raise ImportError(
                "FeedbackStore.export_parquet() requires the optional pyarrow "
                "package. Install a deployment extra that provides pyarrow.",
            )

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output.with_name(f"{output.name}.tmp")
        rows = self.export_calibration_rows(
            limit=limit,
            domain=domain,
            include_text=include_text,
        )
        table = table_cls.from_pylist(rows)
        pq.write_table(table, tmp_path)
        tmp_path.replace(output)
        return output

    def log_export_artifact(
        self,
        path: str | Path,
        *,
        backend: str,
        artifact_name: str = "director-ai-calibration-feedback",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, str]:
        """Log an exported calibration artefact to an optional MLOps backend.

        Supported backends are ``"mlflow"`` and ``"wandb"``. Both are imported
        only when requested; no core runtime dependency is introduced.
        """
        artifact_path = Path(path)
        if not artifact_path.is_file():
            raise FileNotFoundError(str(artifact_path))
        backend_name = backend.strip().lower()
        if backend_name == "mlflow":
            self._log_mlflow_artifact(artifact_path, artifact_name, metadata or {})
        elif backend_name == "wandb":
            self._log_wandb_artifact(artifact_path, artifact_name, metadata or {})
        else:
            raise ValueError("backend must be 'mlflow' or 'wandb'")
        return {
            "backend": backend_name,
            "artifact_name": artifact_name,
            "path": str(artifact_path),
        }

    @staticmethod
    def _log_mlflow_artifact(
        path: Path,
        artifact_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        try:
            mlflow = import_module("mlflow")
        except ImportError as exc:
            raise ImportError(
                "MLflow artefact logging requires the optional mlflow package.",
            ) from exc
        active_run = getattr(mlflow, "active_run", lambda: None)()
        if active_run is None:
            raise RuntimeError("MLflow artefact logging requires an active MLflow run")
        mlflow.log_artifact(str(path), artifact_path=artifact_name)
        if metadata:
            mlflow.log_params(
                {f"calibration_{key}": value for key, value in metadata.items()},
            )

    @staticmethod
    def _log_wandb_artifact(
        path: Path,
        artifact_name: str,
        metadata: Mapping[str, Any],
    ) -> None:
        try:
            wandb = import_module("wandb")
        except ImportError as exc:
            raise ImportError(
                "W&B artefact logging requires the optional wandb package.",
            ) from exc
        run = getattr(wandb, "run", None)
        if run is None:
            raise RuntimeError("W&B artefact logging requires an active wandb run")
        artifact = wandb.Artifact(
            artifact_name,
            type="calibration-feedback",
            metadata=dict(metadata),
        )
        artifact.add_file(str(path))
        run.log_artifact(artifact)

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _ensure_column(conn: sqlite3.Connection, name: str, ddl: str) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(corrections)")}
    if name not in columns:
        conn.execute(f"ALTER TABLE corrections ADD COLUMN {name} {ddl}")
