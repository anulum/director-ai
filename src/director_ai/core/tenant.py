# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Multi-Tenant KB Isolation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Tenant-isolated knowledge base routing.

Each tenant gets its own GroundTruthStore. No data leaks between tenants.

Usage::

    router = TenantRouter()
    router.add_fact("acme", "capital", "Paris is the capital of France.")
    router.add_fact("globex", "hq", "Globex HQ is in Springfield.")

    store = router.get_store("acme")
    # store only sees acme's facts

    # Use with CoherenceScorer:
    scorer = router.get_scorer("acme", threshold=0.6)
"""

from __future__ import annotations

import threading

from .knowledge import GroundTruthStore
from .scorer import CoherenceScorer


class TenantRouter:
    """Routes requests to tenant-isolated GroundTruthStores.

    Thread-safe: stores are created lazily on first access.
    """

    def __init__(self) -> None:
        self._stores: dict[str, GroundTruthStore] = {}
        self._lock = threading.Lock()

    @property
    def tenant_ids(self) -> list[str]:
        with self._lock:
            return list(self._stores.keys())

    def get_store(self, tenant_id: str) -> GroundTruthStore:
        """Get or create an isolated store for this tenant."""
        with self._lock:
            if tenant_id not in self._stores:
                store = GroundTruthStore()
                store.facts = {}
                self._stores[tenant_id] = store
            return self._stores[tenant_id]

    def add_fact(self, tenant_id: str, key: str, value: str) -> None:
        """Add a fact to a specific tenant's store."""
        self.get_store(tenant_id).add(key, value)

    def remove_tenant(self, tenant_id: str) -> bool:
        """Remove a tenant and all its data. Returns True if existed."""
        with self._lock:
            return self._stores.pop(tenant_id, None) is not None

    def get_scorer(
        self,
        tenant_id: str,
        threshold: float = 0.6,
        use_nli: bool | None = None,
    ) -> CoherenceScorer:
        """Build a CoherenceScorer scoped to this tenant's KB."""
        return CoherenceScorer(
            threshold=threshold,
            ground_truth_store=self.get_store(tenant_id),
            use_nli=use_nli,
        )

    def fact_count(self, tenant_id: str) -> int:
        """Number of facts in a tenant's store."""
        with self._lock:
            store = self._stores.get(tenant_id)
            return len(store.facts) if store else 0
