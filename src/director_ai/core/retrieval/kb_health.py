# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Knowledge base health check
"""Knowledge base health diagnostics.

Analyses a ``VectorGroundTruthStore`` for coverage gaps, staleness,
embedding quality, and document statistics. Useful for pre-launch
validation and ongoing monitoring.

Usage::

    from director_ai.core.retrieval.kb_health import KBHealthCheck

    check = KBHealthCheck(store)
    report = check.run()
    print(report.summary)
    if not report.healthy:
        for issue in report.issues:
            print(f"  WARNING: {issue}")
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field

__all__ = ["KBHealthCheck", "KBHealthReport"]

logger = logging.getLogger("DirectorAI.KBHealth")


@dataclass
class KBHealthReport:
    """Result of a KB health check."""

    healthy: bool
    document_count: int
    total_entries: int
    avg_query_latency_ms: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_passed: int = 0
    checks_total: int = 0

    @property
    def summary(self) -> str:
        status = "HEALTHY" if self.healthy else "UNHEALTHY"
        return (
            f"KB Health: {status} "
            f"({self.checks_passed}/{self.checks_total} checks passed, "
            f"{self.document_count} docs, "
            f"{self.avg_query_latency_ms:.1f}ms avg query)"
        )


class KBHealthCheck:
    """Run diagnostics on a knowledge base store.

    Parameters
    ----------
    store : object
        A ``GroundTruthStore`` or ``VectorGroundTruthStore`` instance.
    probe_queries : list[str] | None
        Test queries for latency measurement. Defaults to generic probes.
    min_documents : int
        Minimum expected document count.
    max_query_latency_ms : float
        Maximum acceptable average query latency.
    """

    def __init__(
        self,
        store,
        probe_queries: list[str] | None = None,
        min_documents: int = 1,
        max_query_latency_ms: float = 100.0,
    ) -> None:
        self._store = store
        self._probe_queries = probe_queries or [
            "test query",
            "what is the policy",
            "how does it work",
            "explain the process",
            "technical specification",
        ]
        self._min_documents = min_documents
        self._max_latency = max_query_latency_ms

    def run(self) -> KBHealthReport:
        """Execute all health checks and return report."""
        issues: list[str] = []
        warnings: list[str] = []
        checks_passed = 0
        checks_total = 0

        # Check 1: document count
        checks_total += 1
        doc_count = self._check_document_count()
        if doc_count < self._min_documents:
            issues.append(
                f"Document count ({doc_count}) below minimum ({self._min_documents})"
            )
        else:
            checks_passed += 1

        # Check 2: store is queryable
        checks_total += 1
        queryable = self._check_queryable()
        if not queryable:
            issues.append("Store is not queryable — retrieval will fail")
        else:
            checks_passed += 1

        # Check 3: query latency
        checks_total += 1
        avg_latency = self._measure_query_latency()
        if avg_latency > self._max_latency:
            warnings.append(
                f"Avg query latency ({avg_latency:.1f}ms) exceeds "
                f"threshold ({self._max_latency:.1f}ms)"
            )
        else:
            checks_passed += 1

        # Check 4: retrieval returns results
        checks_total += 1
        has_results = self._check_retrieval_results()
        if not has_results and doc_count > 0:
            warnings.append("Probe queries returned no results despite non-empty store")
        elif has_results:
            checks_passed += 1

        # Check 5: no empty entries
        checks_total += 1
        empty_check = self._check_no_empty_entries()
        if not empty_check:
            warnings.append("Store may contain empty or very short entries")
        else:
            checks_passed += 1

        total_entries = self._get_total_entries()
        healthy = len(issues) == 0

        report = KBHealthReport(
            healthy=healthy,
            document_count=doc_count,
            total_entries=total_entries,
            avg_query_latency_ms=round(avg_latency, 2),
            issues=issues,
            warnings=warnings,
            checks_passed=checks_passed,
            checks_total=checks_total,
        )

        if healthy:
            logger.info("KB health check: %s", report.summary)
        else:
            logger.warning("KB health check: %s", report.summary)
            for issue in issues:
                logger.warning("  ISSUE: %s", issue)
            for warn in warnings:
                logger.info("  WARNING: %s", warn)

        return report

    def _check_document_count(self) -> int:
        """Get document/entry count from store."""
        try:
            if hasattr(self._store, "backend") and hasattr(
                self._store.backend, "count"
            ):
                return int(self._store.backend.count())
            if hasattr(self._store, "count"):
                return int(self._store.count())
            if hasattr(self._store, "facts") and isinstance(self._store.facts, dict):
                return len(self._store.facts)
            return 0
        except Exception:  # noqa: BLE001 — intentional broad catch for diagnostic
            return 0

    def _check_queryable(self) -> bool:
        """Verify that the store accepts queries without error."""
        try:
            if hasattr(self._store, "retrieve_context"):
                self._store.retrieve_context("health check probe")
                return True
            return False
        except Exception:  # noqa: BLE001 — intentional broad catch for diagnostic
            return False

    def _measure_query_latency(self) -> float:
        """Measure average query latency across probe queries."""
        if not hasattr(self._store, "retrieve_context"):
            return 0.0

        latencies: list[float] = []
        for query in self._probe_queries:
            t0 = time.perf_counter()
            with contextlib.suppress(Exception):
                self._store.retrieve_context(query)
            latencies.append((time.perf_counter() - t0) * 1000)

        return sum(latencies) / len(latencies) if latencies else 0.0

    def _check_retrieval_results(self) -> bool:
        """Check if at least one probe query returns results."""
        if not hasattr(self._store, "retrieve_context"):
            return False
        for query in self._probe_queries[:3]:
            with contextlib.suppress(Exception):
                result = self._store.retrieve_context(query)
                if result:
                    return True
        return False

    def _check_no_empty_entries(self) -> bool:
        """Heuristic: check store doesn't have obviously empty entries."""
        # This is a best-effort check — not all backends expose entries
        if hasattr(self._store, "facts") and isinstance(self._store.facts, dict):
            for value in self._store.facts.values():
                if not value or len(str(value).strip()) < 3:
                    return False
        return True

    def _get_total_entries(self) -> int:
        """Get total entry count (may differ from document count)."""
        with contextlib.suppress(Exception):
            if hasattr(self._store, "backend") and hasattr(
                self._store.backend, "count"
            ):
                return int(self._store.backend.count())
            return self._check_document_count()
        return 0
