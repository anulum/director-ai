# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Persistent accounts and issued API keys for the managed Director-AI service.

The hosted beta needs what the static ``DIRECTOR_API_KEYS`` set cannot give: a key
that belongs to an account, can be shown once then only verified by hash, and can
be rotated or revoked without redeploying. ``AccountStore`` is that record — a
small SQLite-backed store (WAL, one short-lived connection per call so it is safe
under the threaded server) that keeps only the SHA-256 of each key and maps a
presented key back to its owning account.

Postgres can replace SQLite later without touching callers: the store's public
methods are the contract, and ``enterprise/audit_pg.py`` already carries the
pooled-Postgres pattern to mirror.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from ._db import connect, new_id, utc_now_iso

_KEY_PREFIX = "dai_"
_KEY_ENTROPY_BYTES = 32
_DISPLAY_PREFIX_LEN = len(_KEY_PREFIX) + 8


class AccountError(Exception):
    """Base class for account-store errors."""


class UnknownAccountError(AccountError):
    """Raised when an operation targets an account id that does not exist."""


class UnknownAPIKeyError(AccountError):
    """Raised when an operation targets a key id that does not exist."""


@dataclass(frozen=True, slots=True)
class Account:
    """A managed-service tenant: the unit that owns keys, usage, and a plan."""

    account_id: str
    email: str
    plan: str
    status: str
    created_at: str

    @property
    def is_active(self) -> bool:
        """Whether the account may authenticate (not suspended or closed)."""
        return self.status == "active"


@dataclass(frozen=True, slots=True)
class APIKey:
    """An issued key record. The raw secret is shown once and never stored."""

    key_id: str
    account_id: str
    prefix: str
    created_at: str
    last_used_at: str | None
    revoked_at: str | None

    @property
    def is_active(self) -> bool:
        """Whether the key is still usable (issued and not revoked)."""
        return self.revoked_at is None


def generate_api_key() -> str:
    """Return a fresh high-entropy key string with the ``dai_`` prefix."""
    return _KEY_PREFIX + secrets.token_urlsafe(_KEY_ENTROPY_BYTES)


def hash_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest used to store and look up a key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class AccountStore:
    """SQLite-backed store of accounts and the keys issued against them.

    One connection is opened per operation (WAL mode), which keeps the store
    correct under the server's worker threads without a shared cursor. Keys are
    persisted only as their SHA-256 digest; ``issue_key`` returns the raw secret
    exactly once for the caller to hand to the account holder.
    """

    def __init__(self, db_path: str | Path = "director_managed.db") -> None:
        self._db_path = str(db_path)
        self._init_schema()

    def _connect(self) -> AbstractContextManager[sqlite3.Connection]:
        return connect(self._db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    account_id TEXT PRIMARY KEY,
                    email      TEXT NOT NULL,
                    plan       TEXT NOT NULL,
                    status     TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id       TEXT PRIMARY KEY,
                    account_id   TEXT NOT NULL REFERENCES accounts(account_id),
                    key_hash     TEXT NOT NULL UNIQUE,
                    prefix       TEXT NOT NULL,
                    created_at   TEXT NOT NULL,
                    last_used_at TEXT,
                    revoked_at   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
                CREATE INDEX IF NOT EXISTS idx_api_keys_account
                    ON api_keys(account_id);
                """
            )

    # ── accounts ────────────────────────────────────────────────────────────

    def create_account(self, email: str, plan: str = "free") -> Account:
        """Create and persist a new active account on the given plan."""
        account = Account(
            account_id=new_id("acct"),
            email=email,
            plan=plan,
            status="active",
            created_at=utc_now_iso(),
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO accounts VALUES (?, ?, ?, ?, ?)",
                (
                    account.account_id,
                    account.email,
                    account.plan,
                    account.status,
                    account.created_at,
                ),
            )
        return account

    def get_account(self, account_id: str) -> Account | None:
        """Return the account, or ``None`` when no such id exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM accounts WHERE account_id = ?", (account_id,)
            ).fetchone()
        return _row_to_account(row) if row is not None else None

    def set_plan(self, account_id: str, plan: str) -> Account:
        """Move an account to a different plan and return the updated record."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE accounts SET plan = ? WHERE account_id = ?",
                (plan, account_id),
            )
            if cur.rowcount == 0:
                raise UnknownAccountError(account_id)
        updated = self.get_account(account_id)
        assert updated is not None  # noqa: S101 — just updated, row exists
        return updated

    def set_status(self, account_id: str, status: str) -> Account:
        """Set the account status (e.g. ``active`` / ``suspended``)."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE accounts SET status = ? WHERE account_id = ?",
                (status, account_id),
            )
            if cur.rowcount == 0:
                raise UnknownAccountError(account_id)
        updated = self.get_account(account_id)
        assert updated is not None  # noqa: S101 — just updated, row exists
        return updated

    # ── keys ────────────────────────────────────────────────────────────────

    def issue_key(self, account_id: str) -> tuple[APIKey, str]:
        """Issue a key for an account; return the record and the raw secret.

        The raw secret is the only time the full key is available — the store
        keeps just its hash. Raises :class:`UnknownAccountError` for a missing id.
        """
        if self.get_account(account_id) is None:
            raise UnknownAccountError(account_id)
        raw_key = generate_api_key()
        record = APIKey(
            key_id=new_id("key"),
            account_id=account_id,
            prefix=raw_key[:_DISPLAY_PREFIX_LEN],
            created_at=utc_now_iso(),
            last_used_at=None,
            revoked_at=None,
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.key_id,
                    record.account_id,
                    hash_key(raw_key),
                    record.prefix,
                    record.created_at,
                    record.last_used_at,
                    record.revoked_at,
                ),
            )
        return record, raw_key

    def authenticate(self, raw_key: str) -> Account | None:
        """Resolve a presented key to its active account, or ``None``.

        Returns ``None`` when the key is unknown, revoked, or owned by a
        non-active account. On success the key's ``last_used_at`` is stamped.
        """
        digest = hash_key(raw_key)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT account_id, revoked_at FROM api_keys WHERE key_hash = ?",
                (digest,),
            ).fetchone()
            if row is None or row["revoked_at"] is not None:
                return None
            account_row = conn.execute(
                "SELECT * FROM accounts WHERE account_id = ?",
                (row["account_id"],),
            ).fetchone()
            if account_row is None:
                return None
            account = _row_to_account(account_row)
            if not account.is_active:
                return None
            conn.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_hash = ?",
                (utc_now_iso(), digest),
            )
        return account

    def revoke_key(self, key_id: str) -> None:
        """Revoke a key by id; revoking an already-revoked key is a no-op."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET revoked_at = ? "
                "WHERE key_id = ? AND revoked_at IS NULL",
                (utc_now_iso(), key_id),
            )
            if cur.rowcount == 0:
                exists = conn.execute(
                    "SELECT 1 FROM api_keys WHERE key_id = ?", (key_id,)
                ).fetchone()
                if exists is None:
                    raise UnknownAPIKeyError(key_id)

    def rotate_key(self, key_id: str) -> tuple[APIKey, str]:
        """Revoke a key and issue a fresh one for the same account."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT account_id FROM api_keys WHERE key_id = ?", (key_id,)
            ).fetchone()
            if row is None:
                raise UnknownAPIKeyError(key_id)
            account_id = row["account_id"]
        self.revoke_key(key_id)
        return self.issue_key(account_id)

    def list_keys(self, account_id: str) -> list[APIKey]:
        """Return all key records for an account, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE account_id = ? ORDER BY created_at DESC",
                (account_id,),
            ).fetchall()
        return [_row_to_key(row) for row in rows]


def _row_to_account(row: sqlite3.Row) -> Account:
    return Account(
        account_id=row["account_id"],
        email=row["email"],
        plan=row["plan"],
        status=row["status"],
        created_at=row["created_at"],
    )


def _row_to_key(row: sqlite3.Row) -> APIKey:
    return APIKey(
        key_id=row["key_id"],
        account_id=row["account_id"],
        prefix=row["prefix"],
        created_at=row["created_at"],
        last_used_at=row["last_used_at"],
        revoked_at=row["revoked_at"],
    )
