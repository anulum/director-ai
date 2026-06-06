# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — board-level KPI computation

"""Compute the board-level guardrail KPIs from labelled decisions + counters.

The external audit asked for the few "board-level" numbers a guardrail product is
actually steered by, rather than another dashboard of raw metrics. This module is
that data layer: it derives halt rate, halt precision, false-positive rate (and
per-domain FPR), and p95 scoring latency from the same reviewer-labelled records
the active-labelling cockpit produces, and passes through the operational
counters (tenant-boundary violations, unsigned-KB-writes rejected, security
exception debt) that the host already tracks. The presentation layer (web
dashboard) is separate; this is the deterministic, testable computation behind
it. Inputs that need an external source (pilot→production conversion, docs
freshness) are accepted as caller-supplied values, never fabricated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..labelling_cockpit.items import GROUNDED, HALLUCINATION, LabelItem

__all__ = ["KpiReport", "compute_kpis"]


def _p95(samples: Sequence[float]) -> float | None:
    """Return the 95th-percentile of ``samples`` (nearest-rank), or None."""
    ordered = sorted(float(s) for s in samples)
    if not ordered:
        return None
    rank = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return round(ordered[rank], 4)


@dataclass(frozen=True)
class KpiReport:
    """The board-level guardrail KPIs for one window."""

    labelled_total: int
    halt_rate: float
    halt_precision: float | None
    false_positive_rate: float | None
    per_domain_false_positive_rate: dict[str, float] = field(default_factory=dict)
    p95_scoring_latency_ms: float | None = None
    tenant_boundary_violations: int = 0
    unsigned_kb_writes_rejected: int = 0
    security_exception_debt: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "labelled_total": self.labelled_total,
            "halt_rate": self.halt_rate,
            "halt_precision": self.halt_precision,
            "false_positive_rate": self.false_positive_rate,
            "per_domain_false_positive_rate": dict(self.per_domain_false_positive_rate),
            "p95_scoring_latency_ms": self.p95_scoring_latency_ms,
            "tenant_boundary_violations": self.tenant_boundary_violations,
            "unsigned_kb_writes_rejected": self.unsigned_kb_writes_rejected,
            "security_exception_debt": self.security_exception_debt,
        }


def _false_positive_rate(items: Sequence[LabelItem]) -> float | None:
    """Fraction of grounded answers the guard wrongly blocked, or None."""
    grounded = [i for i in items if i.label == GROUNDED]
    if not grounded:
        return None
    false_halts = sum(1 for i in grounded if not i.guard_approved)
    return round(false_halts / len(grounded), 4)


def compute_kpis(
    items: Sequence[LabelItem],
    *,
    latency_ms_samples: Sequence[float] = (),
    tenant_boundary_violations: int = 0,
    unsigned_kb_writes_rejected: int = 0,
    security_exception_debt: int = 0,
) -> KpiReport:
    """Compute the board-level KPIs from labelled decisions + counters.

    Parameters
    ----------
    items:
        Reviewer-labelled guard decisions (see the active-labelling cockpit).
    latency_ms_samples:
        Per-request end-to-end scoring latencies, for the p95.
    tenant_boundary_violations, unsigned_kb_writes_rejected,
    security_exception_debt:
        Operational counters the host already tracks, passed through verbatim.
    """
    labelled = [i for i in items if i.labelled]
    total = len(labelled)
    halted = [i for i in labelled if not i.guard_approved]
    halt_rate = round(len(halted) / total, 4) if total else 0.0

    halt_precision: float | None = None
    if halted:
        correct_halts = sum(1 for i in halted if i.label == HALLUCINATION)
        halt_precision = round(correct_halts / len(halted), 4)

    per_domain: dict[str, float] = {}
    domains = {i.domain for i in labelled if i.domain}
    for domain in sorted(domains):
        rate = _false_positive_rate([i for i in labelled if i.domain == domain])
        if rate is not None:
            per_domain[domain] = rate

    return KpiReport(
        labelled_total=total,
        halt_rate=halt_rate,
        halt_precision=halt_precision,
        false_positive_rate=_false_positive_rate(labelled),
        per_domain_false_positive_rate=per_domain,
        p95_scoring_latency_ms=_p95(latency_ms_samples),
        tenant_boundary_violations=int(tenant_boundary_violations),
        unsigned_kb_writes_rejected=int(unsigned_kb_writes_rejected),
        security_exception_debt=int(security_exception_debt),
    )
