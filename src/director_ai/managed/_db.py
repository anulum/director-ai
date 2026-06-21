# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Shared SQLite access for the managed control plane.

Accounts, usage, and plan state live in one SQLite file. A single short-lived
connection per call (WAL, foreign keys on) keeps every store correct under the
server's worker threads without a shared cursor — `connect` is that one place so
the stores do not each re-derive the pragmas.
"""

from __future__ import annotations

import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    """Yield a configured connection that commits on success and always closes."""
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def utc_now_iso() -> str:
    """Return the current UTC instant as a sortable ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def new_id(kind: str) -> str:
    """Return a short opaque identifier such as ``acct_…`` / ``key_…``."""
    return f"{kind}_{secrets.token_hex(12)}"
