# SPDX-License-Identifier: BUSL-1.1
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the managed-service usage meter."""

from __future__ import annotations

from pathlib import Path

import pytest

from director_ai.managed import UsageEvent, UsageMeter, UsageSummary


@pytest.fixture
def meter(tmp_path: Path) -> UsageMeter:
    return UsageMeter(tmp_path / "managed.db")


def test_record_returns_a_stamped_event(meter: UsageMeter) -> None:
    event = meter.record(
        "acct_1",
        "/v1/review",
        key_id="key_1",
        tokens_in=120,
        tokens_out=40,
        latency_ms=12.5,
        decision="approved",
    )
    assert isinstance(event, UsageEvent)
    assert event.account_id == "acct_1"
    assert event.endpoint == "/v1/review"
    assert event.tokens_in == 120
    assert event.decision == "approved"
    assert event.event_id.startswith("use_")
    assert event.ts  # ISO timestamp recorded


def test_record_defaults_are_zero_and_key_optional(meter: UsageMeter) -> None:
    event = meter.record("acct_1", "/v1/process")
    assert event.key_id is None
    assert event.tokens_in == 0
    assert event.tokens_out == 0
    assert event.latency_ms == 0.0
    assert event.decision == ""


def test_request_count_total_and_since(meter: UsageMeter) -> None:
    meter.record("acct_1", "/v1/review")
    boundary = meter.record("acct_1", "/v1/review").ts
    meter.record("acct_1", "/v1/review")
    assert meter.request_count("acct_1") == 3
    # at/after the boundary event's timestamp → the boundary one + the later one
    assert meter.request_count("acct_1", since=boundary) >= 2


def test_request_count_isolated_per_account(meter: UsageMeter) -> None:
    meter.record("acct_1", "/v1/review")
    assert meter.request_count("acct_2") == 0


def test_summary_aggregates_tokens(meter: UsageMeter) -> None:
    meter.record("acct_1", "/v1/review", tokens_in=100, tokens_out=10)
    meter.record("acct_1", "/v1/review", tokens_in=50, tokens_out=5)
    summary = meter.summary("acct_1")
    assert isinstance(summary, UsageSummary)
    assert summary.request_count == 2
    assert summary.tokens_in == 150
    assert summary.tokens_out == 15


def test_summary_of_empty_account_is_zeroed(meter: UsageMeter) -> None:
    summary = meter.summary("acct_unknown")
    assert summary.request_count == 0
    assert summary.tokens_in == 0
    assert summary.tokens_out == 0


def test_summary_window_excludes_out_of_range(meter: UsageMeter) -> None:
    first = meter.record("acct_1", "/v1/review", tokens_in=10)
    mid = meter.record("acct_1", "/v1/review", tokens_in=20).ts
    meter.record("acct_1", "/v1/review", tokens_in=30)
    # [mid, end) drops the first event
    windowed = meter.summary("acct_1", since=mid)
    assert windowed.request_count == 2
    assert windowed.tokens_in == 50
    # (-inf, mid) keeps only the first
    before = meter.summary("acct_1", until=mid)
    assert before.request_count == 1
    assert before.tokens_in == 10
    assert first.ts < mid


def test_events_are_ordered_oldest_first(meter: UsageMeter) -> None:
    a = meter.record("acct_1", "/v1/review")
    b = meter.record("acct_1", "/v1/process")
    ids = [e.event_id for e in meter.events("acct_1")]
    assert ids == [a.event_id, b.event_id]


def test_events_window_and_isolation(meter: UsageMeter) -> None:
    meter.record("acct_1", "/v1/review")
    cut = meter.record("acct_1", "/v1/review").ts
    meter.record("acct_1", "/v1/review")
    meter.record("acct_2", "/v1/review")
    assert all(e.account_id == "acct_1" for e in meter.events("acct_1"))
    assert len(meter.events("acct_1", since=cut)) == 2
    assert len(meter.events("acct_1", until=cut)) == 1


def test_meter_persists_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "managed.db"
    UsageMeter(path).record("acct_1", "/v1/review", tokens_in=7)
    assert UsageMeter(path).summary("acct_1").tokens_in == 7
