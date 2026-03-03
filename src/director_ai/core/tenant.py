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
from .vector_store import (
    InMemoryBackend,
    VectorBackend,
    VectorGroundTruthStore,
)


class TenantRouter:
    """Routes requests to tenant-isolated GroundTruthStores.

    Thread-safe: stores are created lazily on first access.
    """

    def __init__(self) -> None:
        self._stores: dict[str, GroundTruthStore] = {}
        self._vector_stores: dict[tuple[str, str], VectorGroundTruthStore] = {}
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

    def get_vector_store(
        self,
        tenant_id: str,
        backend_type: str = "memory",
        **kwargs,
    ) -> VectorGroundTruthStore:
        """Get or create a tenant-isolated VectorGroundTruthStore.

        Supported backend_type values: "memory", "chroma", "pinecone", "qdrant".
        Extra kwargs are forwarded to the backend constructor.
        """
        key = (tenant_id, backend_type)
        with self._lock:
            if key in self._vector_stores:
                return self._vector_stores[key]
            backend: VectorBackend = self._build_vector_backend(
                tenant_id, backend_type, **kwargs
            )
            store = VectorGroundTruthStore(backend=backend, tenant_id=tenant_id)
            self._vector_stores[key] = store
            return store

    @staticmethod
    def _build_vector_backend(
        tenant_id: str, backend_type: str, **kwargs
    ) -> VectorBackend:
        if backend_type == "memory":
            return InMemoryBackend()
        if backend_type == "chroma":
            from .vector_store import ChromaBackend

            return ChromaBackend(
                collection_name=kwargs.get(
                    "collection_name", f"director_ai_{tenant_id}"
                ),
                **{k: v for k, v in kwargs.items() if k != "collection_name"},
            )
        if backend_type == "pinecone":
            from .vector_store import PineconeBackend

            return PineconeBackend(
                namespace=kwargs.pop("namespace", tenant_id),
                **kwargs,
            )
        if backend_type == "qdrant":
            from .vector_store import QdrantBackend

            return QdrantBackend(
                collection_name=kwargs.get(
                    "collection_name", f"director_facts_{tenant_id}"
                ),
                **{k: v for k, v in kwargs.items() if k != "collection_name"},
            )
        raise ValueError(f"Unknown vector backend_type: {backend_type!r}")

    def fact_count(self, tenant_id: str) -> int:
        """Number of facts in a tenant's store."""
        with self._lock:
            store = self._stores.get(tenant_id)
            return len(store.facts) if store else 0
