# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Multi-Tenant KB Isolation
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Tenant-isolated knowledge base routing + per-tenant model versioning.

Each tenant gets its own GroundTruthStore and optional fine-tuned model.
No data leaks between tenants.

Usage::

    router = TenantRouter()
    router.add_fact("acme", "capital", "Paris is the capital of France.")

    # Per-tenant fine-tuned model:
    router.set_model("acme", "ft-20260310-medical-v1", "./models/acme-v1")
    scorer = router.get_scorer("acme", threshold=0.6)
    # scorer uses acme's fine-tuned model, not the default
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .knowledge import GroundTruthStore
from .scorer import CoherenceScorer
from .vector_store import (
    InMemoryBackend,
    VectorBackend,
    VectorGroundTruthStore,
)


@dataclass
class ModelVersion:
    """Metadata for a tenant's fine-tuned model."""

    model_id: str
    model_path: str
    created_at: float = 0.0
    dataset_hash: str = ""
    balanced_accuracy: float = 0.0
    regression_pp: float = 0.0
    recommendation: str = ""
    active: bool = False


class TenantRouter:
    """Routes requests to tenant-isolated GroundTruthStores.

    Thread-safe: stores are created lazily on first access.
    Supports per-tenant fine-tuned model selection.
    """

    def __init__(self) -> None:
        self._stores: dict[str, GroundTruthStore] = {}
        self._vector_stores: dict[tuple[str, str], VectorGroundTruthStore] = {}
        self._models: dict[str, list[ModelVersion]] = {}
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

    # ── Per-tenant model versioning (Phase D) ──────────────────────

    def set_model(
        self,
        tenant_id: str,
        model_id: str,
        model_path: str,
        balanced_accuracy: float = 0.0,
        regression_pp: float = 0.0,
        recommendation: str = "",
        dataset_hash: str = "",
    ) -> ModelVersion:
        """Register a fine-tuned model for a tenant."""
        mv = ModelVersion(
            model_id=model_id,
            model_path=model_path,
            created_at=time.time(),
            balanced_accuracy=balanced_accuracy,
            regression_pp=regression_pp,
            recommendation=recommendation,
            dataset_hash=dataset_hash,
        )
        with self._lock:
            if tenant_id not in self._models:
                self._models[tenant_id] = []
            self._models[tenant_id].append(mv)
        return mv

    def activate_model(self, tenant_id: str, model_id: str) -> bool:
        """Activate a specific model version for a tenant. Deactivates others."""
        with self._lock:
            versions = self._models.get(tenant_id, [])
            found = False
            for mv in versions:
                if mv.model_id == model_id:
                    mv.active = True
                    found = True
                else:
                    mv.active = False
            return found

    def rollback_model(self, tenant_id: str) -> bool:
        """Deactivate all models for a tenant (revert to baseline)."""
        with self._lock:
            versions = self._models.get(tenant_id, [])
            if not versions:
                return False
            for mv in versions:
                mv.active = False
            return True

    def get_active_model(self, tenant_id: str) -> ModelVersion | None:
        """Return the active model for a tenant, or None for baseline."""
        with self._lock:
            for mv in self._models.get(tenant_id, []):
                if mv.active:
                    return mv
            return None

    def list_models(self, tenant_id: str) -> list[ModelVersion]:
        """List all model versions for a tenant."""
        with self._lock:
            return list(self._models.get(tenant_id, []))

    def delete_model(self, tenant_id: str, model_id: str) -> bool:
        """Remove a model version. Cannot delete active models."""
        with self._lock:
            versions = self._models.get(tenant_id, [])
            for i, mv in enumerate(versions):
                if mv.model_id == model_id:
                    if mv.active:
                        return False
                    versions.pop(i)
                    return True
            return False

    def get_scorer(
        self,
        tenant_id: str,
        threshold: float = 0.6,
        use_nli: bool | None = None,
    ) -> CoherenceScorer:
        """Build a CoherenceScorer scoped to this tenant's KB and model."""
        active = self.get_active_model(tenant_id)
        nli_model = active.model_path if active else None
        kwargs = {}
        if nli_model:
            kwargs["nli_model"] = nli_model
        return CoherenceScorer(
            threshold=threshold,
            ground_truth_store=self.get_store(tenant_id),
            use_nli=use_nli,
            **kwargs,
        )

    def save_manifest(self, path: str | Path) -> None:
        """Save all tenant model metadata to a JSON manifest."""
        manifest = {}
        with self._lock:
            for tid, versions in self._models.items():
                manifest[tid] = [asdict(mv) for mv in versions]
        Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def load_manifest(self, path: str | Path) -> int:
        """Load tenant model metadata from a JSON manifest. Returns count loaded."""
        p = Path(path)
        if not p.exists():
            return 0
        data = json.loads(p.read_text(encoding="utf-8"))
        count = 0
        with self._lock:
            for tid, versions in data.items():
                self._models[tid] = [ModelVersion(**v) for v in versions]
                count += len(versions)
        return count

    def fact_count(self, tenant_id: str) -> int:
        """Number of facts in a tenant's store."""
        with self._lock:
            store = self._stores.get(tenant_id)
            return len(store.facts) if store else 0
