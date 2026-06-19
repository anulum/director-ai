# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Temporal freshness scoring — flag claims that may rely on stale knowledge.

Detects date-sensitive entity types (positions, prices, statistics, records)
and cross-references against source timestamps to assess staleness risk.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..mandatory import mandatory_execution

__all__ = [
    "CitationStatusSignal",
    "CitationStatusVerdict",
    "FreshnessClaim",
    "FreshnessResult",
    "score_temporal_freshness",
]

try:
    from backfire_kernel import rust_score_temporal_freshness

    _RUST_TEMPORAL = True
except ImportError:
    _RUST_TEMPORAL = True

    def rust_score_temporal_freshness(
        _claim_text: str,
        _source_timestamp: float | None,
        _now: float,
        _domain: str | None,
    ) -> tuple[float, str]:
        raise RuntimeError(
            "backfire_kernel rust_score_temporal_freshness is unavailable"
        )


_CLAIM_REASONS: dict[str, str] = {
    "position": "Leadership positions change frequently",
    "statistic": "Statistics are updated periodically",
    "current_reference": "Temporal claim may not reflect current state",
    "record": "Records and rankings change over time",
}

_STATUS_RISK: dict[str, float] = {
    "active": 0.0,
    "current": 0.0,
    "published": 0.0,
    "verified": 0.0,
    "unknown": 0.25,
    "corrected": 0.35,
    "updated": 0.45,
    "stale": 0.65,
    "superseded": 0.75,
    "expression_of_concern": 0.8,
    "concern": 0.8,
    "withdrawn": 1.0,
    "retracted": 1.0,
}

_DOMAIN_MAX_AGE_DAYS: dict[str, float] = {
    "clinical": 45.0,
    "medical": 45.0,
    "finance": 45.0,
    "financial": 45.0,
    "legal": 90.0,
    "science": 90.0,
    "scientific": 90.0,
}

_POSITION_PATTERN = re.compile(
    r"(?:the\s+)?(?:CEO|CTO|CFO|COO|president|prime\s+minister|chairman|"
    r"director|head|leader|secretary|minister|governor|mayor)\s+"
    r"(?:of\s+)?(\S+(?:\s+\S+){0,10})\s+(?:is|was)\b",
    re.IGNORECASE,
)

_STAT_PATTERN = re.compile(
    r"(?:population|GDP|revenue|market\s+cap|stock\s+price|unemployment|"
    r"inflation|interest\s+rate|exchange\s+rate|growth\s+rate)"
    r"(?:\s+\w+){0,5}\s+"
    r"([\d,.]+\s*(?:million|billion|trillion|%|percent)?)",
    re.IGNORECASE,
)

_CURRENT_PATTERN = re.compile(
    r"(?:currently|as of|right now|at present|today|this year|in \d{4})",
    re.IGNORECASE,
)

_RECORD_PATTERN = re.compile(
    r"(?:world\s+record|fastest|tallest|largest|smallest|highest|lowest|"
    r"most\s+\w+|best\s+selling|top\s+\w+|#1|number\s+one)",
    re.IGNORECASE,
)


@dataclass
class FreshnessClaim:
    """A claim identified as potentially date-sensitive."""

    text: str
    claim_type: str  # "position", "statistic", "record", "current_reference"
    staleness_risk: float  # 0 = fresh, 1 = likely stale
    reason: str
    source_id: str = ""
    external_status: str = ""


@dataclass
class CitationStatusSignal:
    """External source status used to adjust temporal freshness risk."""

    source_id: str
    status: str
    observed_at: float | None = None
    published_at: float | None = None
    updated_at: float | None = None
    status_source: str = ""
    note: str = ""
    weight: float = 1.0


@dataclass
class CitationStatusVerdict:
    """Per-source freshness contribution from an external status feed."""

    source_id: str
    status: str
    risk: float
    reason: str
    status_source: str = ""


@dataclass
class FreshnessResult:
    """Result of temporal freshness analysis."""

    claims: list[FreshnessClaim] = field(default_factory=list)
    citation_status_verdicts: list[CitationStatusVerdict] = field(default_factory=list)
    overall_staleness_risk: float = 0.0  # max risk across all claims
    external_status_risk: float = 0.0
    source_age_days: float | None = None
    has_temporal_claims: bool = False

    @property
    def stale_claims(self) -> list[FreshnessClaim]:
        """Return claims whose staleness risk exceeds 0.5."""
        return [c for c in self.claims if c.staleness_risk > 0.5]

    @property
    def risky_statuses(self) -> list[CitationStatusVerdict]:
        """Return citation-status verdicts whose risk exceeds 0.5."""
        return [v for v in self.citation_status_verdicts if v.risk > 0.5]


def score_temporal_freshness(
    text: str,
    source_timestamp: float | None = None,
    max_age_days: float = 180,
    citation_statuses: Sequence[CitationStatusSignal | Mapping[str, Any]] | None = None,
    domain: str = "",
) -> FreshnessResult:
    """Score text for temporal freshness risk.

    Parameters
    ----------
    text : str
        LLM-generated response to analyze.
    source_timestamp : float | None
        Unix timestamp of the source data. If None, assumes
        current time (maximum staleness for date-sensitive claims).
    max_age_days : float
        Number of days after which information is considered stale.
    citation_statuses : Sequence[CitationStatusSignal | Mapping[str, Any]] | None
        External source status signals, such as active, superseded, or retracted.
    domain : str
        Optional domain hint. High-stakes domains use a shorter default age
        window unless ``max_age_days`` is already lower.

    Returns
    -------
    FreshnessResult
        Per-claim staleness analysis.
    """
    effective_max_age_days = _effective_max_age_days(max_age_days, domain)
    status_verdicts = _status_verdicts(citation_statuses, effective_max_age_days)
    external_status_risk = max((v.risk for v in status_verdicts), default=0.0)

    # Rust fast path: regex extraction when no source metadata is supplied.
    if (
        _RUST_TEMPORAL
        and source_timestamp is None
        and not citation_statuses
        and not domain
    ):
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            raw_claims, _overall, _has = rust_score_temporal_freshness(text)
            rust_claims = [
                FreshnessClaim(
                    text=t,
                    claim_type=ct,
                    staleness_risk=risk,
                    reason=_CLAIM_REASONS.get(ct, "Temporal claim"),
                )
                for t, ct, risk in raw_claims
            ]
            rust_overall = max((c.staleness_risk for c in rust_claims), default=0.0)
            return FreshnessResult(
                claims=rust_claims,
                overall_staleness_risk=max(rust_overall, external_status_risk),
                external_status_risk=external_status_risk,
                citation_status_verdicts=status_verdicts,
                has_temporal_claims=len(rust_claims) > 0,
            )

    claims: list[FreshnessClaim] = []

    # Age factor: how old is the source data?
    if source_timestamp is not None:
        age_days = max(0.0, (time.time() - source_timestamp) / 86400)
        age_factor = min(1.0, age_days / effective_max_age_days)
    else:
        age_days = None
        age_factor = 0.5  # unknown source = moderate risk

    # 1. Position references (CEO, president, etc.)
    for m in _POSITION_PATTERN.finditer(text):
        risk = 0.6 + 0.4 * age_factor  # positions change; high base risk
        claims.append(
            FreshnessClaim(
                text=m.group(0).strip(),
                claim_type="position",
                staleness_risk=min(1.0, risk),
                reason="Leadership positions change frequently",
            )
        )

    # 2. Statistical claims (population, GDP, etc.)
    for m in _STAT_PATTERN.finditer(text):
        risk = 0.4 + 0.4 * age_factor
        claims.append(
            FreshnessClaim(
                text=m.group(0).strip(),
                claim_type="statistic",
                staleness_risk=min(1.0, risk),
                reason="Statistics are updated periodically",
            )
        )

    # 3. "Currently" / "as of" references
    for m in _CURRENT_PATTERN.finditer(text):
        risk = 0.5 + 0.5 * age_factor
        context = text[max(0, m.start() - 30) : m.end() + 50].strip()
        claims.append(
            FreshnessClaim(
                text=context,
                claim_type="current_reference",
                staleness_risk=min(1.0, risk),
                reason="Temporal claim may not reflect current state",
            )
        )

    # 4. Record/superlative claims
    for m in _RECORD_PATTERN.finditer(text):
        risk = 0.3 + 0.3 * age_factor
        context = text[max(0, m.start() - 20) : m.end() + 40].strip()
        claims.append(
            FreshnessClaim(
                text=context,
                claim_type="record",
                staleness_risk=min(1.0, risk),
                reason="Records and rankings change over time",
            )
        )

    overall = max(
        max((c.staleness_risk for c in claims), default=0.0),
        external_status_risk,
    )
    return FreshnessResult(
        claims=claims,
        citation_status_verdicts=status_verdicts,
        overall_staleness_risk=overall,
        external_status_risk=external_status_risk,
        source_age_days=age_days,
        has_temporal_claims=len(claims) > 0,
    )


def _effective_max_age_days(max_age_days: float, domain: str) -> float:
    if max_age_days <= 0:
        raise ValueError("max_age_days must be positive")
    domain_limit = _DOMAIN_MAX_AGE_DAYS.get(domain.strip().lower())
    if domain_limit is None:
        return max_age_days
    return min(max_age_days, domain_limit)


def _status_verdicts(
    citation_statuses: Sequence[CitationStatusSignal | Mapping[str, Any]] | None,
    max_age_days: float,
) -> list[CitationStatusVerdict]:
    if not citation_statuses:
        return []
    return [
        _status_verdict(_coerce_status_signal(signal), max_age_days)
        for signal in citation_statuses
    ]


def _status_verdict(
    signal: CitationStatusSignal,
    max_age_days: float,
) -> CitationStatusVerdict:
    status = signal.status.strip().lower() or "unknown"
    base_risk = _STATUS_RISK.get(status, _STATUS_RISK["unknown"])
    timestamp = (
        signal.updated_at if signal.updated_at is not None else signal.published_at
    )
    age_risk = 0.0
    if timestamp is not None:
        age_days = max(0.0, (time.time() - timestamp) / 86400)
        age_risk = min(0.5, age_days / max_age_days * 0.5)
    risk = max(base_risk, age_risk)
    risk = min(1.0, max(0.0, risk * max(0.0, signal.weight)))
    reason = signal.note or _status_reason(status, risk)
    return CitationStatusVerdict(
        source_id=signal.source_id,
        status=status,
        risk=risk,
        reason=reason,
        status_source=signal.status_source,
    )


def _coerce_status_signal(
    signal: CitationStatusSignal | Mapping[str, Any],
) -> CitationStatusSignal:
    if isinstance(signal, CitationStatusSignal):
        return signal
    return CitationStatusSignal(
        source_id=str(signal.get("source_id", "")),
        status=str(signal.get("status", "unknown")),
        observed_at=_optional_float(signal.get("observed_at")),
        published_at=_optional_float(signal.get("published_at")),
        updated_at=_optional_float(signal.get("updated_at")),
        status_source=str(signal.get("status_source", "")),
        note=str(signal.get("note", "")),
        weight=float(signal.get("weight", 1.0)),
    )


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _status_reason(status: str, risk: float) -> str:
    if status in {"retracted", "withdrawn"}:
        return "Source has been withdrawn or retracted"
    if status in {"superseded", "stale"}:
        return "Source has a newer external status"
    if status in {"corrected", "updated"}:
        return "Source changed after first publication"
    if risk > 0.0:
        return "Source age or status increases freshness risk"
    return "Source status does not increase freshness risk"
