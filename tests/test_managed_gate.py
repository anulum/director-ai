# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the managed request gate (auth + metering + quota)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from director_ai.managed import Plan, PlanRegistry
from director_ai.managed.gate import (
    ManagedGate,
    _extract_key,
    fastapi_dependency,
    utc_month_start,
)


@pytest.fixture
def gate(tmp_path: Path) -> ManagedGate:
    # a 3-request cap makes the quota boundary cheap to exercise
    plans = PlanRegistry(
        {"free": Plan("free", monthly_request_cap=3, requests_per_minute=20)}
    )
    return ManagedGate(tmp_path / "managed.db", plans=plans)


class _FakeURL:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeState:
    pass


class _FakeRequest:
    def __init__(self, headers: dict[str, str], path: str = "/v1/review") -> None:
        self.headers = headers
        self.url = _FakeURL(path)
        self.state = _FakeState()


# ── month boundary ──────────────────────────────────────────────────────────


def test_utc_month_start_is_first_of_month_midnight() -> None:
    start = utc_month_start()
    assert start.endswith("+00:00")
    assert "T00:00:00" in start
    assert start[8:10] == "01"  # day-of-month


# ── check(): auth ────────────────────────────────────────────────────────────


def test_check_missing_key_is_401(gate: ManagedGate) -> None:
    outcome = gate.check(None)
    assert outcome.status == 401
    assert not outcome.allowed
    assert outcome.account is None


def test_check_invalid_key_is_401(gate: ManagedGate) -> None:
    outcome = gate.check("dai_not_real")
    assert outcome.status == 401
    assert "invalid" in outcome.reason


def test_check_revoked_key_is_401(gate: ManagedGate) -> None:
    acct = gate.accounts.create_account("a@b.io")
    record, raw = gate.accounts.issue_key(acct.account_id)
    gate.accounts.revoke_key(record.key_id)
    assert gate.check(raw).status == 401


# ── check(): quota ───────────────────────────────────────────────────────────


def test_check_allows_and_meters_within_cap(gate: ManagedGate) -> None:
    acct = gate.accounts.create_account("a@b.io")
    _, raw = gate.accounts.issue_key(acct.account_id)
    outcome = gate.check(raw, endpoint="/v1/review")
    assert outcome.allowed
    assert outcome.status == 200
    assert outcome.account is not None
    assert outcome.headers["X-RateLimit-Limit"] == "3"
    assert outcome.headers["X-RateLimit-Remaining"] == "3"  # before this call
    # the allowed call was metered
    assert gate.usage.request_count(acct.account_id) == 1


def test_check_denies_over_cap_without_metering(gate: ManagedGate) -> None:
    acct = gate.accounts.create_account("a@b.io")
    _, raw = gate.accounts.issue_key(acct.account_id)
    for _ in range(3):
        assert gate.check(raw).allowed  # fill the 3-request cap
    blocked = gate.check(raw)
    assert blocked.status == 429
    assert not blocked.allowed
    assert blocked.headers["X-RateLimit-Remaining"] == "0"
    # the rejected call did NOT consume quota
    assert gate.usage.request_count(acct.account_id) == 3


def test_check_unlimited_plan_never_caps(tmp_path: Path) -> None:
    g = ManagedGate(tmp_path / "managed.db")  # default plans: scale = unlimited
    acct = g.accounts.create_account("a@b.io", plan="scale")
    _, raw = g.accounts.issue_key(acct.account_id)
    outcome = g.check(raw)
    assert outcome.allowed
    assert outcome.headers["X-RateLimit-Limit"] == "unlimited"
    assert outcome.headers["X-RateLimit-Remaining"] == "unlimited"


# ── key extraction ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Authorization": "Bearer dai_abc"}, "dai_abc"),
        ({"X-API-Key": "dai_xyz"}, "dai_xyz"),
        ({"Authorization": "Basic nope"}, None),
        ({}, None),
    ],
)
def test_extract_key(headers: dict[str, str], expected: str | None) -> None:
    assert _extract_key(_FakeRequest(headers)) == expected


# ── FastAPI dependency adapter ──────────────────────────────────────────────


def test_dependency_returns_account_and_stamps_state(gate: ManagedGate) -> None:
    acct = gate.accounts.create_account("a@b.io")
    _, raw = gate.accounts.issue_key(acct.account_id)
    dep = fastapi_dependency(gate)
    req = _FakeRequest({"Authorization": f"Bearer {raw}"})
    resolved = asyncio.run(dep(req))
    assert resolved.account_id == acct.account_id
    assert req.state.managed_account.account_id == acct.account_id
    assert req.state.managed_quota_headers["X-RateLimit-Limit"] == "3"


def test_dependency_raises_401_for_bad_key(gate: ManagedGate) -> None:
    from fastapi import HTTPException

    dep = fastapi_dependency(gate)
    req = _FakeRequest({"X-API-Key": "dai_bad"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(req))
    assert exc.value.status_code == 401


def test_dependency_raises_429_over_cap(gate: ManagedGate) -> None:
    from fastapi import HTTPException

    acct = gate.accounts.create_account("a@b.io")
    _, raw = gate.accounts.issue_key(acct.account_id)
    for _ in range(3):
        gate.check(raw)
    dep = fastapi_dependency(gate)
    req = _FakeRequest({"Authorization": f"Bearer {raw}"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(dep(req))
    assert exc.value.status_code == 429
    assert exc.value.headers is not None
