# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Production Guard Canary Operations
"""Canary surface of the production guard.

:class:`CanaryOperationsMixin` mints tenant-scoped canary facts, plants
them in the guard's knowledge base, and scans responses (and cited
evidence) for tripped canaries — the data-exfiltration tripwire of
:class:`~director_ai.guard.ProductionGuard`. Registry and detector are
built lazily on first use and persist on the guard.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from director_ai.core import GroundTruthStore
    from director_ai.core.canary import (
        CanaryDetector,
        CanaryFact,
        CanaryRegistry,
        CanarySignal,
    )

logger = logging.getLogger("DirectorAI.Guard")

__all__ = ["CanaryOperationsMixin"]


class CanaryOperationsMixin:
    """Tenant-scoped canary minting, planting, and scanning.

    All state is initialised by :class:`~director_ai.guard.ProductionGuard`'s
    ``__init__``; the knowledge base comes from the composing guard through
    the ``_store`` contract declared below.
    """

    _canary_registry: CanaryRegistry | None
    _canary_detector: CanaryDetector | None

    if TYPE_CHECKING:
        # Provided by the composing ProductionGuard.
        _store: GroundTruthStore

    def _ensure_canary(self) -> CanaryDetector:
        if self._canary_detector is None:
            from director_ai.core.canary import CanaryDetector, CanaryRegistry

            self._canary_registry = CanaryRegistry()
            self._canary_detector = CanaryDetector(
                self._canary_registry,
                alert=lambda s: logger.warning(
                    "canary tripped: id=%s tenant=%s signal=%s",
                    s.canary_id,
                    s.tenant_id,
                    s.signal,
                ),
            )
        return self._canary_detector

    def plant_canary(
        self,
        tenant_id: str,
        *,
        template: str = "Internal reference marker {token}: do not disclose.",
        token: str | None = None,
    ) -> CanaryFact:
        """Mint a tenant-scoped canary, plant it in the KB, and return it.

        The canary text is added to the knowledge base so retrieval can surface
        it under attack; its sentinel token must never appear in a legitimate
        answer. Detect trips with :meth:`scan_canaries`.
        """
        self._ensure_canary()
        assert self._canary_registry is not None
        fact = self._canary_registry.mint(tenant_id, template=template, token=token)
        self._store.add(fact.canary_id, fact.text)
        return fact

    def scan_canaries(
        self,
        response: str,
        tenant_id: str,
        *,
        evidence: Iterable[Any] = (),
    ) -> list[CanarySignal]:
        """Scan a response (and optional evidence chunks) for tripped canaries.

        Returns a :class:`CanarySignal` for each canary token found in the
        response (leakage) and each canary chunk present in ``evidence``
        (citation).
        """
        detector = self._ensure_canary()
        return detector.scan(response, tenant_id, evidence=list(evidence))
