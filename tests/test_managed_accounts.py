# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the managed-service account + API-key store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from director_ai.managed import Account, AccountStore, APIKey
from director_ai.managed.accounts import (
    UnknownAccountError,
    UnknownAPIKeyError,
    generate_api_key,
    hash_key,
)


@pytest.fixture
def store(tmp_path: Path) -> AccountStore:
    return AccountStore(tmp_path / "managed.db")


# ── key primitives ──────────────────────────────────────────────────────────


def test_generated_keys_are_prefixed_and_unique() -> None:
    keys = {generate_api_key() for _ in range(200)}
    assert len(keys) == 200  # high entropy → no collisions
    assert all(k.startswith("dai_") for k in keys)


def test_hash_is_deterministic_and_hides_the_key() -> None:
    raw = generate_api_key()
    assert hash_key(raw) == hash_key(raw)
    assert raw not in hash_key(raw)
    assert len(hash_key(raw)) == 64  # sha-256 hex


# ── accounts ─────────────────────────────────────────────────────────────────


def test_create_account_defaults_to_active_free(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    assert isinstance(acct, Account)
    assert acct.plan == "free"
    assert acct.is_active
    assert acct.account_id.startswith("acct_")
    assert store.get_account(acct.account_id) == acct


def test_get_unknown_account_returns_none(store: AccountStore) -> None:
    assert store.get_account("acct_missing") is None


def test_set_plan_and_status_update_in_place(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    assert store.set_plan(acct.account_id, "pro").plan == "pro"
    suspended = store.set_status(acct.account_id, "suspended")
    assert suspended.status == "suspended"
    assert not suspended.is_active


@pytest.mark.parametrize("op", ["set_plan", "set_status"])
def test_mutating_unknown_account_raises(store: AccountStore, op: str) -> None:
    with pytest.raises(UnknownAccountError):
        getattr(store, op)("acct_missing", "x")


# ── key issuance + authentication ───────────────────────────────────────────


def test_issue_key_returns_raw_once_and_stores_only_hash(
    store: AccountStore, tmp_path: Path
) -> None:
    acct = store.create_account("a@b.io")
    record, raw = store.issue_key(acct.account_id)
    assert isinstance(record, APIKey)
    assert raw.startswith("dai_")
    assert record.prefix == raw[: len(record.prefix)]
    assert record.is_active
    # the raw secret must never be persisted — only its hash
    db_text = (tmp_path / "managed.db").read_bytes()
    assert raw.encode() not in db_text
    assert hash_key(raw).encode() in db_text


def test_authenticate_resolves_active_key_to_account(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    _, raw = store.issue_key(acct.account_id)
    resolved = store.authenticate(raw)
    assert resolved is not None
    assert resolved.account_id == acct.account_id


def test_authenticate_stamps_last_used(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    record, raw = store.issue_key(acct.account_id)
    assert record.last_used_at is None
    store.authenticate(raw)
    assert store.list_keys(acct.account_id)[0].last_used_at is not None


def test_issue_key_for_unknown_account_raises(store: AccountStore) -> None:
    with pytest.raises(UnknownAccountError):
        store.issue_key("acct_missing")


@pytest.mark.parametrize("bad", ["", "dai_not_a_real_key", "garbage"])
def test_authenticate_rejects_unknown_keys(store: AccountStore, bad: str) -> None:
    assert store.authenticate(bad) is None


def test_authenticate_rejects_revoked_key(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    record, raw = store.issue_key(acct.account_id)
    store.revoke_key(record.key_id)
    assert store.authenticate(raw) is None


def test_authenticate_rejects_suspended_account(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    _, raw = store.issue_key(acct.account_id)
    store.set_status(acct.account_id, "suspended")
    assert store.authenticate(raw) is None


# ── revoke / rotate / list ──────────────────────────────────────────────────


def test_revoke_is_idempotent_and_marks_inactive(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    record, _ = store.issue_key(acct.account_id)
    store.revoke_key(record.key_id)
    store.revoke_key(record.key_id)  # second revoke is a no-op, not an error
    assert not store.list_keys(acct.account_id)[0].is_active


def test_revoke_unknown_key_raises(store: AccountStore) -> None:
    with pytest.raises(UnknownAPIKeyError):
        store.revoke_key("key_missing")


def test_rotate_revokes_old_and_issues_new_for_same_account(
    store: AccountStore,
) -> None:
    acct = store.create_account("a@b.io")
    old_record, old_raw = store.issue_key(acct.account_id)
    new_record, new_raw = store.rotate_key(old_record.key_id)
    assert new_record.account_id == acct.account_id
    assert new_record.key_id != old_record.key_id
    assert store.authenticate(old_raw) is None  # old no longer works
    resolved = store.authenticate(new_raw)
    assert resolved is not None and resolved.account_id == acct.account_id


def test_rotate_unknown_key_raises(store: AccountStore) -> None:
    with pytest.raises(UnknownAPIKeyError):
        store.rotate_key("key_missing")


def test_list_keys_returns_newest_first(store: AccountStore) -> None:
    acct = store.create_account("a@b.io")
    first, _ = store.issue_key(acct.account_id)
    second, _ = store.issue_key(acct.account_id)
    ids = [k.key_id for k in store.list_keys(acct.account_id)]
    assert ids == [second.key_id, first.key_id]


def test_keys_are_isolated_per_account(store: AccountStore) -> None:
    a = store.create_account("a@b.io")
    b = store.create_account("b@b.io")
    store.issue_key(a.account_id)
    assert store.list_keys(b.account_id) == []


# ── persistence ─────────────────────────────────────────────────────────────


def test_store_reopens_existing_database(tmp_path: Path) -> None:
    path = tmp_path / "managed.db"
    first = AccountStore(path)
    acct = first.create_account("a@b.io")
    _, raw = first.issue_key(acct.account_id)
    # a fresh store over the same file sees the persisted account + key
    reopened = AccountStore(path)
    resolved = reopened.authenticate(raw)
    assert resolved is not None and resolved.account_id == acct.account_id


def test_foreign_key_constraint_is_enforced(store: AccountStore) -> None:
    # api_keys.account_id references accounts; a dangling insert is rejected
    with pytest.raises(sqlite3.IntegrityError), store._connect() as conn:
        conn.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("key_x", "acct_missing", "h", "dai_x", "t", None, None),
        )


def test_authenticate_returns_none_for_orphaned_key(store: AccountStore) -> None:
    """A key whose account vanished authenticates to nothing.

    The foreign key normally prevents an orphan; this forces the state with FK
    enforcement off to prove ``authenticate`` still refuses rather than
    dereferencing a missing account.
    """
    acct = store.create_account("a@b.io")
    _, raw = store.issue_key(acct.account_id)
    with store._connect() as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("DELETE FROM accounts WHERE account_id = ?", (acct.account_id,))
    assert store.authenticate(raw) is None
