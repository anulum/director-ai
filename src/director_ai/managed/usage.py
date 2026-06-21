# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Per-account usage metering for the managed Director-AI service.

Every billable call is recorded as one row so two questions can be answered
cheaply: "how many requests has this account made since the billing window
opened?" (the quota gate) and "what did this account use last month?" (the
invoice). The meter is window-agnostic — callers pass an ISO timestamp boundary,
because ISO-8601 strings sort the same way the instants do — so the same store
serves both the quota check and the billing export without a second schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._db import connect, new_id, utc_now_iso


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One recorded billable call against an account."""

    event_id: str
    account_id: str
    key_id: str | None
    endpoint: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    decision: str
    ts: str


@dataclass(frozen=True, slots=True)
class UsageSummary:
    """Aggregated usage for an account over a time window."""

    account_id: str
    request_count: int
    tokens_in: int
    tokens_out: int


class UsageMeter:
    """SQLite-backed append-only log of billable calls, aggregated on read."""

    def __init__(self, db_path: str | Path = "director_managed.db") -> None:
        self._db_path = str(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with connect(self._db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    event_id   TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    key_id     TEXT,
                    endpoint   TEXT NOT NULL,
                    tokens_in  INTEGER NOT NULL,
                    tokens_out INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    decision   TEXT NOT NULL,
                    ts         TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usage_account_ts
                    ON usage_events(account_id, ts);
                """
            )

    def record(
        self,
        account_id: str,
        endpoint: str,
        *,
        key_id: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        decision: str = "",
    ) -> UsageEvent:
        """Append one usage event and return the stored record."""
        event = UsageEvent(
            event_id=new_id("use"),
            account_id=account_id,
            key_id=key_id,
            endpoint=endpoint,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            decision=decision,
            ts=utc_now_iso(),
        )
        with connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO usage_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id,
                    event.account_id,
                    event.key_id,
                    event.endpoint,
                    event.tokens_in,
                    event.tokens_out,
                    event.latency_ms,
                    event.decision,
                    event.ts,
                ),
            )
        return event

    def request_count(self, account_id: str, *, since: str | None = None) -> int:
        """Count requests for an account, optionally only at/after ``since``."""
        sql = "SELECT COUNT(*) AS n FROM usage_events WHERE account_id = ?"
        params: list[str] = [account_id]
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since)
        with connect(self._db_path) as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["n"])

    def summary(
        self,
        account_id: str,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> UsageSummary:
        """Aggregate request count and token totals over an optional window."""
        sql = (
            "SELECT COUNT(*) AS n, "
            "COALESCE(SUM(tokens_in), 0) AS ti, "
            "COALESCE(SUM(tokens_out), 0) AS too "
            "FROM usage_events WHERE account_id = ?"
        )
        params: list[str] = [account_id]
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since)
        if until is not None:
            sql += " AND ts < ?"
            params.append(until)
        with connect(self._db_path) as conn:
            row = conn.execute(sql, params).fetchone()
        return UsageSummary(
            account_id=account_id,
            request_count=int(row["n"]),
            tokens_in=int(row["ti"]),
            tokens_out=int(row["too"]),
        )

    def events(
        self,
        account_id: str,
        *,
        since: str | None = None,
        until: str | None = None,
    ) -> list[UsageEvent]:
        """Return the raw events for an account over a window, oldest first."""
        sql = "SELECT * FROM usage_events WHERE account_id = ?"
        params: list[str] = [account_id]
        if since is not None:
            sql += " AND ts >= ?"
            params.append(since)
        if until is not None:
            sql += " AND ts < ?"
            params.append(until)
        sql += " ORDER BY ts ASC"
        with connect(self._db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            UsageEvent(
                event_id=row["event_id"],
                account_id=row["account_id"],
                key_id=row["key_id"],
                endpoint=row["endpoint"],
                tokens_in=row["tokens_in"],
                tokens_out=row["tokens_out"],
                latency_ms=row["latency_ms"],
                decision=row["decision"],
                ts=row["ts"],
            )
            for row in rows
        ]
