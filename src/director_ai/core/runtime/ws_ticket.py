# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — WebSocket Handshake Tickets
"""Short-lived single-use tickets for browser WebSocket authentication.

Browsers cannot attach custom headers (``X-API-Key`` / ``Authorization:
Bearer``) to the WebSocket handshake, so a browser client first calls an
authenticated HTTP endpoint to exchange its API key for a short-lived,
single-use ticket, then opens the socket with ``?ticket=...``. Tickets are
bound to the issuing key and tenant, expire quickly, and are consumed on first
redemption — which is materially safer than placing a long-lived API key in the
WebSocket URL.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TicketBinding:
    """The key/tenant a redeemed ticket authenticates as."""

    api_key: str
    tenant_id: str


class WebSocketTicketRegistry:
    """In-process registry of short-lived single-use WebSocket tickets.

    Tickets live in memory bound to the issuing process. With multiple server
    workers the WebSocket must reach the same process that issued the ticket
    (sticky sessions); a shared backend (e.g. Redis) would be required for
    cross-worker redemption.
    """

    def __init__(
        self,
        ttl_seconds: float = 30.0,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = float(ttl_seconds)
        self._clock = clock
        self._tickets: dict[str, tuple[float, TicketBinding]] = {}

    def issue(self, api_key: str, tenant_id: str = "") -> str:
        """Mint a ticket bound to ``api_key`` and ``tenant_id``.

        Returns an opaque URL-safe token to be passed as the ``ticket`` query
        parameter on the WebSocket handshake.
        """
        if not api_key:
            raise ValueError("api_key is required to issue a ticket")
        self._prune()
        token = secrets.token_urlsafe(32)
        self._tickets[token] = (
            self._clock() + self._ttl,
            TicketBinding(api_key=api_key, tenant_id=tenant_id),
        )
        return token

    def redeem(self, ticket: str) -> TicketBinding | None:
        """Consume ``ticket`` and return its binding, or ``None``.

        A ticket is single-use: it is removed on the first redemption regardless
        of outcome, so a replayed or expired ticket never authenticates.
        """
        if not ticket:
            return None
        self._prune()
        entry = self._tickets.pop(ticket, None)
        if entry is None:
            return None
        expiry, binding = entry
        if self._clock() > expiry:
            return None
        return binding

    @property
    def ttl_seconds(self) -> float:
        """Configured ticket lifetime in seconds."""
        return self._ttl

    def _prune(self) -> None:
        """Drop expired tickets so the registry cannot grow unbounded."""
        now = self._clock()
        expired = [tok for tok, (exp, _) in self._tickets.items() if now > exp]
        for tok in expired:
            del self._tickets[tok]

    def __len__(self) -> int:
        """Return the count of live tickets after pruning expired ones."""
        self._prune()
        return len(self._tickets)
