# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WebSocket Ticket Registry Tests
"""Tests for short-lived single-use WebSocket handshake tickets."""

from __future__ import annotations

import pytest

from director_ai.core.runtime.ws_ticket import (
    TicketBinding,
    WebSocketTicketRegistry,
)


class _FakeClock:
    """Manually advanced monotonic clock for deterministic expiry tests."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_issue_then_redeem_returns_binding():
    reg = WebSocketTicketRegistry(ttl_seconds=30.0)
    ticket = reg.issue("key-1", "tenant-a")
    binding = reg.redeem(ticket)
    assert binding == TicketBinding(api_key="key-1", tenant_id="tenant-a")


def test_ticket_is_single_use():
    reg = WebSocketTicketRegistry()
    ticket = reg.issue("key-1")
    assert reg.redeem(ticket) is not None
    # Second redemption fails — the ticket was consumed.
    assert reg.redeem(ticket) is None


def test_expired_ticket_is_rejected():
    clock = _FakeClock()
    reg = WebSocketTicketRegistry(ttl_seconds=10.0, clock=clock)
    ticket = reg.issue("key-1")
    clock.advance(10.001)
    assert reg.redeem(ticket) is None


def test_ticket_valid_just_before_expiry():
    clock = _FakeClock()
    reg = WebSocketTicketRegistry(ttl_seconds=10.0, clock=clock)
    ticket = reg.issue("key-1", "tenant-x")
    clock.advance(9.0)
    assert reg.redeem(ticket) == TicketBinding("key-1", "tenant-x")


def test_unknown_ticket_returns_none():
    reg = WebSocketTicketRegistry()
    assert reg.redeem("never-issued") is None


def test_empty_ticket_returns_none():
    reg = WebSocketTicketRegistry()
    assert reg.redeem("") is None


def test_issue_requires_api_key():
    reg = WebSocketTicketRegistry()
    with pytest.raises(ValueError, match="api_key is required"):
        reg.issue("")


def test_non_positive_ttl_rejected():
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        WebSocketTicketRegistry(ttl_seconds=0)
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        WebSocketTicketRegistry(ttl_seconds=-1)


def test_tickets_are_unique():
    reg = WebSocketTicketRegistry()
    tickets = {reg.issue("key-1") for _ in range(50)}
    assert len(tickets) == 50


def test_expired_tickets_are_pruned():
    clock = _FakeClock()
    reg = WebSocketTicketRegistry(ttl_seconds=5.0, clock=clock)
    reg.issue("key-1")
    reg.issue("key-2")
    assert len(reg) == 2
    clock.advance(6.0)
    # len() prunes expired entries so the registry cannot grow unbounded.
    assert len(reg) == 0


def test_ttl_seconds_property():
    reg = WebSocketTicketRegistry(ttl_seconds=42.0)
    assert reg.ttl_seconds == 42.0


class _QueueClock:
    """Clock returning a fixed sequence of timestamps, one per call."""

    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        return self._values.pop(0)


def test_redeem_rejects_ticket_expiring_between_prune_and_check():
    # The prune/check race: a ticket survives the in-redeem prune (clock=10, not
    # yet past expiry=10) but the wall clock advances to 11 before the explicit
    # expiry check, which must then reject it. Clock calls: issue prune+expiry,
    # redeem prune+check.
    clock = _QueueClock([0.0, 0.0, 10.0, 11.0])
    reg = WebSocketTicketRegistry(ttl_seconds=10.0, clock=clock)
    ticket = reg.issue("key-1")
    assert reg.redeem(ticket) is None
