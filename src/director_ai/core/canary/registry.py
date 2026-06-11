# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Canary fact registry

"""Tenant-scoped counterfactual canary facts.

A canary is a deliberately false, uniquely-marked fact planted in a tenant's
knowledge base. No legitimate answer should ever contain its sentinel token, so
a token that surfaces in a model's output — or a canary chunk that turns up in
the retrieved evidence — is direct evidence of leakage, injection, or KB
poisoning. This module mints and stores those canaries per tenant; detection
lives in :mod:`director_ai.core.canary.detector`.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = ["CanaryFact", "CanaryRegistry"]

# Metadata keys a planted canary chunk carries so the firewall, audit log, and
# detector can recognise it without re-deriving anything.
CANARY_FLAG = "kb_canary"
CANARY_ID_KEY = "kb_canary_id"


@dataclass(frozen=True)
class CanaryFact:
    """One planted canary fact.

    Parameters
    ----------
    canary_id:
        Stable identifier for this canary.
    tenant_id:
        The tenant whose knowledge base the canary belongs to.
    token:
        The unique sentinel substring that must appear verbatim in any leak.
    text:
        The full canary fact text planted in the knowledge base; contains
        ``token``.
    created_at:
        RFC-3339 UTC creation time.
    """

    canary_id: str
    tenant_id: str
    token: str
    text: str
    created_at: str

    def metadata(self) -> dict[str, Any]:
        """Return the chunk metadata to store when planting this canary."""
        return {
            CANARY_FLAG: True,
            CANARY_ID_KEY: self.canary_id,
            "tenant_id": self.tenant_id,
        }


def _default_token() -> str:
    return f"CANARY-{secrets.token_hex(8)}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class CanaryRegistry:
    """Mint and store canary facts per tenant.

    Parameters
    ----------
    token_factory:
        Callable returning a fresh unique sentinel token. Injectable so tests
        are deterministic; defaults to a cryptographically random token.
    clock:
        Callable returning an RFC-3339 UTC timestamp. Injectable for tests.
    """

    def __init__(
        self,
        *,
        token_factory: Callable[[], str] | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        self._token_factory = token_factory or _default_token
        self._clock = clock or _utc_now
        self._by_tenant: dict[str, list[CanaryFact]] = {}
        self._lock = threading.Lock()

    def mint(
        self,
        tenant_id: str,
        *,
        template: str = "Internal reference marker {token}: do not disclose.",
        token: str | None = None,
    ) -> CanaryFact:
        """Create, store, and return a new canary for ``tenant_id``.

        ``template`` must contain ``{token}`` so the sentinel is embedded in the
        planted text; the returned fact's ``text`` is the rendered template.
        """
        tenant_id = tenant_id.strip()
        if not tenant_id:
            raise ValueError("tenant_id is required")
        if "{token}" not in template:
            raise ValueError("template must contain '{token}'")
        marker = token or self._token_factory()
        if not marker.strip():
            raise ValueError("token must be non-empty")
        fact = CanaryFact(
            canary_id=f"canary_{secrets.token_hex(8)}",
            tenant_id=tenant_id,
            token=marker,
            text=template.format(token=marker),
            created_at=self._clock(),
        )
        with self._lock:
            self._by_tenant.setdefault(tenant_id, []).append(fact)
        return fact

    def facts_for(self, tenant_id: str) -> tuple[CanaryFact, ...]:
        """Return the canaries registered for ``tenant_id``."""
        with self._lock:
            return tuple(self._by_tenant.get(tenant_id.strip(), ()))

    def tokens_for(self, tenant_id: str) -> tuple[str, ...]:
        """Return the sentinel tokens registered for ``tenant_id``."""
        return tuple(fact.token for fact in self.facts_for(tenant_id))

    def find(self, canary_id: str) -> CanaryFact | None:
        """Return the canary with ``canary_id``, across all tenants."""
        with self._lock:
            for facts in self._by_tenant.values():
                for fact in facts:
                    if fact.canary_id == canary_id:
                        return fact
        return None
