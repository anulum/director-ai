# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — VectorGroundTruthStore

"""``VectorGroundTruthStore`` — semantic RAG store over any ``VectorBackend``.

Extends the keyword-based :class:`GroundTruthStore` with
embedding-based similarity search and exposes a ``grounded()``
factory that wires the recommended hybrid (BM25 + dense) recipe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ...evidence_firewall import EvidenceFirewall, FirewallContext
from ...metrics import metrics
from ...otel import trace_vector_add, trace_vector_query
from ...types import EvidenceChunk
from ..knowledge import GroundTruthStore, _require_non_empty_string
from ._conflicts import ConflictLedgerMixin
from ._versioning import VersionLedgerMixin
from .base import (
    RECOMMENDED_EMBEDDING_MODEL,
    InMemoryBackend,
    VectorBackend,
    logger,
)
from .composite import HybridBackend
from .embedding import SentenceTransformerBackend

__all__ = ["VectorGroundTruthStore"]


def _is_vector_backend_like(backend: object) -> bool:
    """Return whether an object exposes the VectorBackend method surface."""
    return all(
        callable(getattr(backend, method, None)) for method in ("add", "query", "count")
    )


def _require_non_negative_top_k(top_k: int) -> int:
    """Validate a non-negative integer retrieval limit."""
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError("top_k must be an integer")
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0; got {top_k!r}")
    return top_k


def _result_evidence_text(result: dict[str, Any]) -> str | None:
    """Return usable evidence text from a backend result, or ``None`` to skip it.

    The ``VectorBackend.query`` contract is loosely typed (``list[dict[str, Any]]``
    with no enforced keys), so a third-party backend such as ColBERT may return
    matches without a ``text`` field. A result carrying no non-empty string text
    holds no evidence to ground against, so it is skipped rather than crashing the
    whole review with a ``KeyError``.
    """
    text = result.get("text")
    if isinstance(text, str) and text:
        return text
    return None


def _result_source(result: dict[str, Any]) -> str:
    """Build the ``vector:<id>`` evidence source label, tolerating a missing id."""
    return f"vector:{result.get('id', '')}"


class VectorGroundTruthStore(
    VersionLedgerMixin,
    ConflictLedgerMixin,
    GroundTruthStore,
):
    """Ground truth store with vector-based semantic retrieval.

    Extends the keyword-based ``GroundTruthStore`` with embedding-based
    similarity search. Falls back to keyword matching when the vector
    backend returns no results.

    Parameters
    ----------
    backend : VectorBackend — vector DB backend (default: InMemoryBackend).

    """

    def __init__(
        self,
        backend: VectorBackend | None = None,
        tenant_id: str = "",
        evidence_firewall: EvidenceFirewall | None = None,
    ) -> None:
        super().__init__()
        if backend is not None and not _is_vector_backend_like(backend):
            raise ValueError("backend must provide add, query, and count methods")
        if not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a string")
        self.backend = backend if backend is not None else InMemoryBackend()
        self.tenant_id = tenant_id.strip()
        # Optional pre-model evidence firewall. When set, every retrieval batch
        # is screened so quarantined chunks never reach the grounding context.
        self.evidence_firewall = evidence_firewall
        self._version_records = {}
        self._retraction_records = []
        self._replacement_records = []
        self._conflict_records = []

    def _resolved_tenant_id(self, tenant_id: str = "") -> str:
        """Return the explicit tenant id or the store default tenant id."""
        if not isinstance(tenant_id, str):
            raise ValueError("tenant_id must be a string")
        return (tenant_id or self.tenant_id).strip()

    def kb_snapshot_records(self, tenant_id: str = "") -> list[dict[str, str]]:
        """Return canonical KB snapshot records visible to *tenant_id*."""
        records = []
        for record in self.version_manifest(tenant_id).values():
            snapshot_record = {
                "key": record.get("key", ""),
                "tenant_id": record.get("tenant_id", ""),
                "version": record.get("version", ""),
                "chunk_version": record.get("chunk_version", ""),
                "content_hash": record.get("content_hash", ""),
                "previous_hash": record.get("previous_hash", ""),
                "record_kind": record.get("record_kind", ""),
                "chunk_index": record.get("chunk_index", ""),
                "status": record.get("status", "active"),
                "retraction_reason": record.get("retraction_reason", ""),
                "replacement_reason": record.get("replacement_reason", ""),
                "source_id": record.get("source_id", ""),
                "external_id": record.get("external_id", ""),
                "source_timestamp": record.get("source_timestamp", ""),
                "updated_timestamp": record.get("updated_timestamp", ""),
                "citation_status": record.get("citation_status", ""),
                "status_source": record.get("status_source", ""),
                "status_observed_at": record.get("status_observed_at", ""),
                "claim_id": record.get("claim_id", ""),
                "claim_source": record.get("claim_source", ""),
                "signed_fact_id": record.get("signed_fact_id", ""),
                "passport_claim_id": record.get("passport_claim_id", ""),
            }
            records.append(snapshot_record)
        return sorted(
            records,
            key=lambda item: (
                item["tenant_id"],
                item["key"],
                item["record_kind"],
                item["chunk_index"],
            ),
        )

    def kb_snapshot_root(self, tenant_id: str = "") -> str:
        """Return a deterministic Merkle root for the current KB snapshot."""
        leaves = [
            self._snapshot_leaf(record)
            for record in self.kb_snapshot_records(tenant_id)
        ]
        return self._merkle_root_hex(leaves)

    def kb_snapshot_audit_record(self, tenant_id: str = "") -> dict[str, str | int]:
        """Return a compact audit payload for the current KB snapshot."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        records = self.kb_snapshot_records(tenant_id)
        return {
            "event": "kb_snapshot",
            "tenant_id": tenant_id,
            "revision": self.revision,
            "record_count": len(records),
            "retraction_count": len(self.retraction_records(tenant_id)),
            "replacement_count": len(self.replacement_records(tenant_id)),
            "conflict_count": len(self.conflict_reports(tenant_id)),
            "merkle_root": self._merkle_root_hex(
                [self._snapshot_leaf(record) for record in records]
            ),
        }

    def add_fact(
        self,
        key: str,
        value: str,
        tenant_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Alias for add() — also populates parent keyword store."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        key = _require_non_empty_string("key", key)
        value = _require_non_empty_string("value", value)
        self.add(key, value, metadata=metadata, tenant_id=tenant_id)
        fact_key = f"{tenant_id}:{key}" if tenant_id else key
        self.facts[fact_key] = value

    def ingest(self, texts: list[str], tenant_id: str = "") -> int:
        """Bulk-add plain text documents into the vector backend."""
        tenant_id = self._resolved_tenant_id(tenant_id)
        if not isinstance(texts, list):
            raise ValueError("texts must be a list of non-empty strings")
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("texts must contain only non-empty strings")
        for i, text in enumerate(texts):
            doc_id = f"ingest_{i}_{tenant_id}"
            metadata = {
                "source": "ingest",
                **self._build_version_metadata(
                    key=doc_id,
                    value=text,
                    tenant_id=tenant_id,
                    record_kind="derived_chunk",
                    requested_bump="patch",
                    chunk_index=i,
                ),
            }
            if tenant_id:
                metadata["tenant_id"] = tenant_id
            self.backend.add(
                doc_id=doc_id,
                text=text,
                metadata=metadata,
            )
            self._commit_version_metadata(doc_id, metadata, tenant_id)
        if texts:
            self._bump_revision()
        logger.info("Ingested %d documents into vector backend.", len(texts))
        return len(texts)

    def add(
        self,
        key: str,
        value: str,
        metadata: dict[str, Any] | None = None,
        tenant_id: str = "",
    ) -> None:
        """Add one fact to the vector backend with version metadata."""
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        key = _require_non_empty_string("key", key)
        value = _require_non_empty_string("value", value)
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        doc_id = f"{tenant_id}::{key}" if tenant_id else key
        combined_text = f"{key}: {value}"
        incoming = dict(metadata or {})
        requested_bump = str(incoming.pop("kb_version_bump", "patch"))
        meta = {
            **incoming,
            "key": key,
            "value": value,
            **self._build_version_metadata(
                key=key,
                value=value,
                tenant_id=tenant_id,
                record_kind="fact",
                requested_bump=requested_bump,
                chunk_index=0,
            ),
        }
        meta.update(self._normalised_claim_metadata(meta))
        if tenant_id:
            meta["tenant_id"] = tenant_id

        with trace_vector_add() as span:
            start_time = time.monotonic()
            metrics.inc("knowledge_adds_total")
            try:
                conflicts = self._build_conflict_reports(key, meta, tenant_id)
                self.backend.add(doc_id=doc_id, text=combined_text, metadata=meta)
                self._commit_version_metadata(key, meta, tenant_id)
                self._conflict_records.extend(conflicts)
                self._bump_revision()
                duration = time.monotonic() - start_time
                metrics.observe("knowledge_add_duration_seconds", duration)
                span.set_attribute("vector.doc_id", doc_id)
                span.set_attribute("vector.tenant_id", tenant_id)
            except Exception as e:
                metrics.inc("knowledge_add_errors")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                raise ValueError(f"Failed to add to vector store: {e}") from e

    @staticmethod
    def _snapshot_leaf(record: dict[str, str]) -> bytes:
        """Return a domain-separated Merkle leaf digest for a record."""
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(b"director-ai/kb-snapshot/v1/leaf\x00" + payload).digest()

    @staticmethod
    def _merkle_root_hex(leaves: list[bytes]) -> str:
        """Return the Merkle root for snapshot leaves as hex."""
        if not leaves:
            return hashlib.sha256(b"director-ai/kb-snapshot/v1/empty").hexdigest()
        level = list(leaves)
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [
                hashlib.sha256(
                    b"director-ai/kb-snapshot/v1/node\x00" + level[i] + level[i + 1]
                ).digest()
                for i in range(0, len(level), 2)
            ]
        return level[0].hex()

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> str | None:
        """Retrieve context as a string (matching parent interface).

        Falls back to keyword-based parent if vector search returns nothing.
        """
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        query = _require_non_empty_string("query", query)
        top_k = _require_non_negative_top_k(top_k)
        with trace_vector_query() as span:
            start_time = time.monotonic()
            metrics.inc("knowledge_queries_total")
            try:
                try:
                    results = self.backend.query(
                        query,
                        n_results=top_k,
                        tenant_id=tenant_id,
                    )
                except TypeError:
                    # Backend doesn't accept tenant_id
                    results = self.backend.query(query, n_results=top_k)
                span.set_attribute("vector.query.k", top_k)
                span.set_attribute("vector.tenant_id", tenant_id)

                if results:
                    active_results = self._active_results(results, tenant_id)
                    texts = [
                        text
                        for r in active_results
                        if (text := _result_evidence_text(r)) is not None
                    ]
                    if not texts:
                        return super().retrieve_context(
                            query,
                            tenant_id=tenant_id,
                            top_k=top_k,
                        )
                    duration = time.monotonic() - start_time
                    metrics.observe("knowledge_query_duration_seconds", duration)
                    return "; ".join(texts)

                duration = time.monotonic() - start_time
                metrics.observe("knowledge_query_duration_seconds", duration)
                # Fall back to keyword-based parent
                return super().retrieve_context(
                    query,
                    tenant_id=tenant_id,
                    top_k=top_k,
                )
            except Exception as e:
                metrics.inc("knowledge_query_errors")
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

                raise ValueError(f"Failed to query vector store: {e}") from e

    @classmethod
    def grounded(
        cls,
        embedding_model: str = RECOMMENDED_EMBEDDING_MODEL,
        use_hybrid: bool = True,
        rrf_k: int = 60,
        tenant_id: str = "",
    ) -> VectorGroundTruthStore:
        """Build the recommended grounded retrieval recipe.

        Sets up hybrid retrieval (BM25 + dense) with a sentence-transformer
        embedding model. This is the intended production path for domain
        profiles (medical, finance, legal) where NLI-only scoring has
        100% FPR without KB grounding.

        Usage::

            store = VectorGroundTruthStore.grounded()
            store.ingest(["Your product documentation...", ...])
            scorer = CoherenceScorer(ground_truth_store=store, use_nli=True)

        Parameters
        ----------
        embedding_model : str
            HuggingFace model ID for dense embeddings.
            Default: ``BAAI/bge-large-en-v1.5``.
        use_hybrid : bool
            Wrap dense backend with BM25 + RRF fusion (default True).
        rrf_k : int
            Reciprocal Rank Fusion parameter (default 60).
        tenant_id : str
            Default tenant scope for multi-tenant deployments.
        """
        dense: VectorBackend
        try:
            dense = SentenceTransformerBackend(model_name=embedding_model)
        except Exception:
            logger.warning(
                "sentence-transformers not available, falling back to InMemoryBackend. "
                "Install with: pip install director-ai[vector]"
            )
            dense = InMemoryBackend()

        backend = HybridBackend(base=dense, rrf_k=rrf_k) if use_hybrid else dense

        return cls(backend=backend, tenant_id=tenant_id)

    def retrieve_context_with_chunks(
        self,
        query: str,
        top_k: int = 3,
        tenant_id: str = "",
    ) -> list[EvidenceChunk]:
        """Retrieve context as EvidenceChunk objects."""
        import time

        tenant_id = self._resolved_tenant_id(tenant_id)
        query = _require_non_empty_string("query", query)
        top_k = _require_non_negative_top_k(top_k)
        with trace_vector_query() as span:
            start_time = time.monotonic()
            try:
                try:
                    results = self.backend.query(
                        query,
                        n_results=top_k,
                        tenant_id=tenant_id,
                    )
                except TypeError:
                    # Backend doesn't accept tenant_id
                    results = self.backend.query(query, n_results=top_k)
                chunks = []
                for r in self._active_results(results, tenant_id):
                    text = _result_evidence_text(r)
                    if text is None:
                        continue
                    chunks.append(
                        EvidenceChunk(
                            text=text,
                            distance=r.get("distance", 0.0),
                            source=_result_source(r),
                        ),
                    )
                duration = time.monotonic() - start_time
                metrics.observe("knowledge_query_duration_seconds", duration)
                if not chunks:
                    return super().retrieve_context_with_chunks(
                        query,
                        top_k=top_k,
                        tenant_id=tenant_id,
                    )
                return chunks
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
                raise ValueError(f"Failed to query vector store: {e}") from e

    def _active_results(
        self,
        results: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Filter stale or retracted vector results before returning evidence."""
        active: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict):
                active.append(result)
                continue
            source_key = str(metadata.get("kb_source_key", result.get("id", "")))
            result_tenant = str(metadata.get("tenant_id", tenant_id))
            version_key = self._version_key(source_key, result_tenant)
            record = self._version_records.get(version_key)
            if record is not None and record.get("status") == "retracted":
                continue
            result_hash = str(metadata.get("kb_content_hash", ""))
            if (
                record is not None
                and result_hash
                and result_hash != record["content_hash"]
            ):
                continue
            active.append(result)
        return self._firewall_screen(active, tenant_id)

    def _firewall_screen(
        self,
        results: list[dict[str, Any]],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Drop chunks the evidence firewall quarantines, if one is configured.

        A no-op when no firewall is attached, so retrieval behaviour is
        unchanged unless a deployment opts in.
        """
        if self.evidence_firewall is None:
            return results
        import time

        context = FirewallContext(tenant_id=tenant_id, now_unix=time.time())
        report = self.evidence_firewall.screen(results, context)
        # Verdicts are 1:1 with input order, so the original row dicts (with
        # their distance/score fields) are preserved for the admitted chunks.
        return [
            result
            for result, verdict in zip(results, report.verdicts, strict=True)
            if verdict.admitted
        ]
