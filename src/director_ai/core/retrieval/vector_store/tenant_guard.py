# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tenant isolation enforcement for vector backends

"""Bound-tenant enforcement wrapper for shared vector backends.

Metadata-filter tenancy (Pinecone ``$eq`` filters, FAISS post-filter,
BM25 metadata checks) is only as strong as every caller passing the
right ``tenant_id`` on every call — a forgotten or empty tenant runs
an unfiltered query over the shared index. ``TenantScopedBackend``
binds the tenant once at construction and turns that convention into
an enforced contract:

- ``add`` stamps the bound tenant into document metadata; a
  conflicting caller-supplied ``tenant_id`` raises
  :class:`TenantIsolationError` instead of silently mislabelling.
- ``query`` always filters by the bound tenant — an empty caller
  tenant can never widen the scope, and a different non-empty caller
  tenant raises.
- Returned rows are verified (defence in depth against a backend
  whose filter is broken or lossy): any row whose metadata carries a
  missing or foreign ``tenant_id`` is dropped and counted in the
  ``tenant_isolation_violations`` metric — or raises when
  ``strict=True``.

``count()`` delegates to the base backend and is therefore
index-wide on a shared index; it is a capacity signal, not a
tenant-scoped document count.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ...metrics import metrics
from .base import VectorBackend

__all__ = ["TenantIsolationError", "TenantScopedBackend"]

logger = logging.getLogger("DirectorAI.VectorStore")

# Mirrors the conservative tenant-id shape enforced by
# ``director_ai.core.tenant`` (which imports this package, so the
# pattern is duplicated here rather than imported to avoid a cycle).
_SAFE_TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class TenantIsolationError(ValueError):
    """A tenant-isolation contract was violated."""


class TenantScopedBackend(VectorBackend):
    """Bind any ``VectorBackend`` to one tenant and enforce the scope."""

    def __init__(
        self,
        base: VectorBackend,
        tenant_id: str,
        *,
        strict: bool = False,
    ) -> None:
        if base is None:
            raise ValueError("base backend is required")
        if not isinstance(tenant_id, str) or not tenant_id:
            raise TenantIsolationError(
                "TenantScopedBackend requires a non-empty tenant_id",
            )
        if not _SAFE_TENANT_RE.fullmatch(tenant_id):
            raise TenantIsolationError(
                "tenant_id must match ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$",
            )
        if not isinstance(strict, bool):
            raise ValueError("strict must be a boolean")
        self._base = base
        self._tenant = tenant_id
        self._strict = strict

    @property
    def tenant_id(self) -> str:
        """The tenant this backend is bound to."""
        return self._tenant

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Stamp the bound tenant into metadata and index the document."""
        stamped = dict(metadata or {})
        existing = stamped.get("tenant_id")
        if existing not in (None, "", self._tenant):
            raise TenantIsolationError(
                f"document tenant_id {existing!r} conflicts with the "
                f"bound tenant {self._tenant!r}",
            )
        stamped["tenant_id"] = self._tenant
        self._base.add(doc_id, text, stamped)

    def query(
        self,
        text: str,
        n_results: int = 3,
        tenant_id: str = "",
    ) -> list[dict[str, Any]]:
        """Query the bound tenant's scope and verify every returned row."""
        if tenant_id and tenant_id != self._tenant:
            raise TenantIsolationError(
                f"query tenant_id {tenant_id!r} conflicts with the "
                f"bound tenant {self._tenant!r}",
            )
        rows = self._base.query(
            text,
            n_results=n_results,
            tenant_id=self._tenant,
        )
        kept: list[dict[str, Any]] = []
        leaked = 0
        for row in rows:
            row_tenant = (row.get("metadata") or {}).get("tenant_id")
            if row_tenant == self._tenant:
                kept.append(row)
            else:
                leaked += 1
        if leaked:
            metrics.inc("tenant_isolation_violations", float(leaked))
            logger.warning(
                "Dropped %d result(s) outside tenant %r returned by %s",
                leaked,
                self._tenant,
                type(self._base).__name__,
            )
            if self._strict:
                raise TenantIsolationError(
                    f"backend returned {leaked} result(s) outside "
                    f"tenant {self._tenant!r}",
                )
        return kept

    def count(self) -> int:
        """Index-wide document count of the underlying (shared) base."""
        return self._base.count()
