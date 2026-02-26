from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_DEFAULT_DB = Path.home() / ".director-ai" / "stats.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  REAL    NOT NULL,
    approved   INTEGER NOT NULL,
    score      REAL,
    h_logical  REAL,
    h_factual  REAL,
    latency_ms REAL,
    halted     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reviews_ts ON reviews(timestamp);
"""


class StatsStore:
    """SQLite-backed usage statistics for Director-AI reviews."""

    def __init__(self, db_path: str | Path | None = None):
        self._path = Path(db_path) if db_path else _DEFAULT_DB
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def record_review(
        self,
        approved: bool,
        score: float | None = None,
        h_logical: float | None = None,
        h_factual: float | None = None,
        latency_ms: float | None = None,
        halted: bool = False,
    ) -> None:
        self._conn.execute(
            "INSERT INTO reviews "
            "(timestamp, approved, score, h_logical, h_factual, latency_ms, halted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                int(approved),
                score,
                h_logical,
                h_factual,
                latency_ms,
                int(halted),
            ),
        )
        self._conn.commit()

    def summary(self, since: float | None = None) -> dict:
        clause = "WHERE timestamp >= ?" if since else ""
        params = (since,) if since else ()
        row = self._conn.execute(
            f"SELECT COUNT(*) as total, "
            f"SUM(approved) as approved, "
            f"SUM(halted) as halted, "
            f"AVG(score) as avg_score, "
            f"AVG(latency_ms) as avg_latency_ms "
            f"FROM reviews {clause}",
            params,
        ).fetchone()
        return {
            "total": row["total"] or 0,
            "approved": row["approved"] or 0,
            "rejected": (row["total"] or 0) - (row["approved"] or 0),
            "halted": row["halted"] or 0,
            "avg_score": round(row["avg_score"], 4) if row["avg_score"] else None,
            "avg_latency_ms": round(row["avg_latency_ms"], 1)
            if row["avg_latency_ms"]
            else None,
        }

    def hourly_breakdown(self, days: int = 7) -> list[dict]:
        since = time.time() - days * 86400
        rows = self._conn.execute(
            "SELECT CAST((timestamp - ?) / 3600 AS INTEGER) as hour_bucket, "
            "COUNT(*) as total, SUM(approved) as approved, SUM(halted) as halted, "
            "AVG(score) as avg_score "
            "FROM reviews WHERE timestamp >= ? "
            "GROUP BY hour_bucket ORDER BY hour_bucket",
            (since, since),
        ).fetchall()
        return [
            {
                "hours_ago": int(r["hour_bucket"]),
                "total": r["total"],
                "approved": r["approved"] or 0,
                "halted": r["halted"] or 0,
                "avg_score": round(r["avg_score"], 4) if r["avg_score"] else None,
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
