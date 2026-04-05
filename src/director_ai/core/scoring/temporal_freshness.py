# SPDX-License-Identifier: AGPL-3.0-or-later
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
from dataclasses import dataclass, field

__all__ = ["FreshnessClaim", "FreshnessResult", "score_temporal_freshness"]

try:
    from backfire_kernel import rust_score_temporal_freshness

    _RUST_TEMPORAL = True
except ImportError:
    _RUST_TEMPORAL = False

_CLAIM_REASONS: dict[str, str] = {
    "position": "Leadership positions change frequently",
    "statistic": "Statistics are updated periodically",
    "current_reference": "Temporal claim may not reflect current state",
    "record": "Records and rankings change over time",
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


@dataclass
class FreshnessResult:
    """Result of temporal freshness analysis."""

    claims: list[FreshnessClaim] = field(default_factory=list)
    overall_staleness_risk: float = 0.0  # max risk across all claims
    has_temporal_claims: bool = False

    @property
    def stale_claims(self) -> list[FreshnessClaim]:
        return [c for c in self.claims if c.staleness_risk > 0.5]


def score_temporal_freshness(
    text: str,
    source_timestamp: float | None = None,
    max_age_days: float = 180,
) -> FreshnessResult:
    """Analyze text for temporal freshness risk.

    Parameters
    ----------
    text : str
        LLM-generated response to analyze.
    source_timestamp : float | None
        Unix timestamp of the source data. If None, assumes
        current time (maximum staleness for date-sensitive claims).
    max_age_days : float
        Number of days after which information is considered stale.

    Returns
    -------
    FreshnessResult
        Per-claim staleness analysis.
    """
    # Rust fast path: regex extraction when no source_timestamp
    if _RUST_TEMPORAL and source_timestamp is None:
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
            overall_staleness_risk=rust_overall,
            has_temporal_claims=len(rust_claims) > 0,
        )

    claims: list[FreshnessClaim] = []

    # Age factor: how old is the source data?
    if source_timestamp is not None:
        age_days = (time.time() - source_timestamp) / 86400
        age_factor = min(1.0, age_days / max_age_days)
    else:
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

    overall = max((c.staleness_risk for c in claims), default=0.0)
    return FreshnessResult(
        claims=claims,
        overall_staleness_risk=overall,
        has_temporal_claims=len(claims) > 0,
    )
