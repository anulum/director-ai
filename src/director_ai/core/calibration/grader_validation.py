# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — validate the honesty grader (WS-3, the meta-FPR contract)

"""Validate the honesty grader itself — measure the FPR of the trust gate.

A trust-rendering system is itself a guardrail, and every guardrail has a
measurable error rate. The honesty grade (``validated | bounded | refuted |
falsified | unknown``) is produced by a *classifier* — the honesty bridge — and a
classifier has its own false-positive and false-negative rates. A portal that
renders a grade it never validated is theatre: a beautiful render of an
uncalibrated verdict.

This module is the WS-3 release-blocker contract (SCPN-STUDIO Round 2): run a
grader against a labelled validation set and publish its drift-gated error rate,
so a *grade above its evidence* is a measurable, CI-failable event. The headline
measurable is the **overclaim rate** — P(the grader renders more support than the
ground truth warrants) — reported with a one-sided Hoeffding upper bound and,
crucially, **paired with coverage** so a grader that punts everything to
``unknown`` cannot game a perfect overclaim rate (the abstention-crutch failure;
WS-4 frontier discipline applied to the grader).

The grader is an injected callable, so this harness validates *any* grader — the
canonical ``@anulum/ui`` honesty bridge, a Director-AI reference grader, or a
candidate — uniformly. It reuses the shipped distribution-free
:func:`~director_ai.core.calibration.miscoverage.hoeffding_upper_bound`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .miscoverage import hoeffding_upper_bound

__all__ = [
    "GraderCase",
    "GraderReport",
    "GraderStatus",
    "GraderValidationError",
    "StatusMetrics",
    "assert_grader_admissible",
    "validate_grader",
]

#: The canonical honesty statuses (LOCK-4); ``falsified``/``refuted`` are
#: first-class negatives, never hidden.
GraderStatus = Literal["validated", "bounded", "refuted", "falsified", "unknown"]

#: How much *support* each status asserts. An overclaim is rendering a status that
#: asserts MORE support than the ground truth warrants — ``rank(pred) >
#: rank(truth)``. ``unknown`` asserts no support (1, above the negatives, below any
#: positive claim); ``refuted``/``falsified`` assert a negative (0). Rendering
#: ``unknown`` for a truly ``validated`` claim is an *under*-claim (conservative,
#: not an overclaim); rendering ``refuted`` for ``validated`` is a false
#: refutation, counted in that status's FPR, not as an overclaim.
_SUPPORT_RANK: Mapping[GraderStatus, int] = {
    "falsified": 0,
    "refuted": 0,
    "unknown": 1,
    "bounded": 2,
    "validated": 3,
}

_STATUSES: tuple[GraderStatus, ...] = (
    "validated",
    "bounded",
    "refuted",
    "falsified",
    "unknown",
)


class GraderValidationError(RuntimeError):
    """Raised when a grader report violates an admissibility (release-blocker) rule."""


@dataclass(frozen=True)
class GraderCase:
    """One labelled validation case: the grader's inputs and the true status.

    ``inputs`` is whatever the grader consumes (evidence kind/level, claim
    boundary, the raw signal vector); ``ground_truth`` is the status an oracle
    adjudicated. The harness never inspects ``inputs`` — only the grader does.
    """

    case_id: str
    inputs: Mapping[str, object]
    ground_truth: GraderStatus


@dataclass(frozen=True)
class StatusMetrics:
    """Per-status error rates: a confusion slice, not one scalar."""

    status: GraderStatus
    support: int  # cases whose ground truth IS this status
    predicted: int  # cases the grader assigned this status
    false_positive_rate: float  # P(predict status | truth != status)
    false_positive_upper: float  # one-sided Hoeffding upper bound on the FPR
    false_negative_rate: float  # P(not predict status | truth == status)

    def to_dict(self) -> dict[str, object]:
        """Serialise for the published ``studio.grader-report.v1`` payload."""
        return {
            "status": self.status,
            "support": self.support,
            "predicted": self.predicted,
            "fpr": round(self.false_positive_rate, 4),
            "fpr_upper": round(self.false_positive_upper, 4),
            "fnr": round(self.false_negative_rate, 4),
        }


@dataclass(frozen=True)
class GraderReport:
    """The published, gate-able verdict on a grader (``studio.grader-report.v1``).

    ``headline_overclaim_rate`` is the FPR of the honesty gate itself; it is
    reported with ``headline_overclaim_upper`` (one-sided bound) and ``coverage``
    so it can never be gamed by abstaining. ``per_status`` carries the full
    confusion so a single benign aggregate cannot hide a bad slice (e.g. a missed
    ``falsified``).
    """

    n: int
    per_status: tuple[StatusMetrics, ...]
    headline_overclaim_rate: float
    headline_overclaim_upper: float
    coverage: float

    def status(self, status: GraderStatus) -> StatusMetrics:
        """Return the metrics slice for ``status``."""
        for metrics in self.per_status:
            if metrics.status == status:
                return metrics
        raise KeyError(status)

    def to_dict(self) -> dict[str, object]:
        """Serialise the whole report for publication + the CI drift gate."""
        return {
            "schema": "studio.grader-report.v1",
            "n": self.n,
            "headline_overclaim_rate": round(self.headline_overclaim_rate, 4),
            "headline_overclaim_upper": round(self.headline_overclaim_upper, 4),
            "coverage": round(self.coverage, 4),
            "per_status": [m.to_dict() for m in self.per_status],
        }

    def canonical_bytes(self) -> bytes:
        """Return the byte-stable canonical form — the unit a seal signs (WS-1).

        Deterministic, sorted-key, tight-separator JSON, so the grader-report can
        be a *signed unit* in the verifiable-honesty contract: a portal that
        strips the report's grade is detectable because the rendered grade no
        longer matches the digest of this canonical form. Recompute-verification
        re-runs :func:`validate_grader` and checks the digests are equal.
        """
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def content_digest(self) -> str:
        """Return the ``sha256:`` digest of the canonical form (H2 reproduction)."""
        return "sha256:" + hashlib.sha256(self.canonical_bytes()).hexdigest()


def validate_grader(
    grader: Callable[[Mapping[str, object]], GraderStatus],
    cases: Sequence[GraderCase],
    *,
    confidence: float = 0.95,
) -> GraderReport:
    """Run ``grader`` over ``cases`` and measure its error rates.

    Returns a :class:`GraderReport` carrying the per-status confusion, the
    headline overclaim rate (rendering more support than the truth warrants) with
    its one-sided Hoeffding upper bound, and the coverage (the non-``unknown``
    fraction). ``confidence`` sets the bound level (default 0.95 → delta 0.05).

    Raises ``ValueError`` on an empty case set (an unmeasured grader cannot be
    certified) or an out-of-range confidence.
    """
    if not cases:
        raise ValueError("cannot validate a grader on an empty case set")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    delta = 1.0 - confidence

    n = len(cases)
    predictions = [(grader(case.inputs), case.ground_truth) for case in cases]

    per_status: list[StatusMetrics] = []
    for status in _STATUSES:
        support = sum(1 for _, truth in predictions if truth == status)
        predicted = sum(1 for pred, _ in predictions if pred == status)
        # FP: predicted this status while the truth was a different status.
        fp = sum(1 for pred, truth in predictions if pred == status and truth != status)
        negatives = n - support
        fpr = fp / negatives if negatives else 0.0
        fpr_upper = hoeffding_upper_bound(fp, negatives, delta) if negatives else 0.0
        # FN: truth was this status but the grader assigned something else.
        fn = sum(1 for pred, truth in predictions if truth == status and pred != status)
        fnr = fn / support if support else 0.0
        per_status.append(
            StatusMetrics(
                status=status,
                support=support,
                predicted=predicted,
                false_positive_rate=fpr,
                false_positive_upper=fpr_upper,
                false_negative_rate=fnr,
            )
        )

    overclaims = sum(
        1 for pred, truth in predictions if _SUPPORT_RANK[pred] > _SUPPORT_RANK[truth]
    )
    covered = sum(1 for pred, _ in predictions if pred != "unknown")

    return GraderReport(
        n=n,
        per_status=tuple(per_status),
        headline_overclaim_rate=overclaims / n,
        headline_overclaim_upper=hoeffding_upper_bound(overclaims, n, delta),
        coverage=covered / n,
    )


def assert_grader_admissible(
    report: GraderReport,
    *,
    overclaim_alpha: float = 0.05,
    coverage_floor: float = 0.70,
) -> None:
    """Enforce the WS-3 release-blocker rules; raise on any violation.

    1. The overclaim **upper bound** must not exceed ``overclaim_alpha`` — a grade
       above its evidence, beyond the tolerated rate, fails (the meta-overclaim).
    2. A missed ``falsified`` (non-zero FNR) fails unconditionally — a refutation
       must never be silently downgraded.
    3. Coverage must meet ``coverage_floor`` — a grader that abstains its way to a
       perfect overclaim rate is the abstention crutch, and is rejected.
    """
    if report.headline_overclaim_upper > overclaim_alpha:
        raise GraderValidationError(
            f"grader overclaim upper bound {report.headline_overclaim_upper:.4f} "
            f"exceeds alpha {overclaim_alpha} (a grade above its evidence)"
        )
    falsified_fnr = report.status("falsified").false_negative_rate
    if falsified_fnr > 0.0:
        raise GraderValidationError(
            f"grader missed a falsification (falsified FNR {falsified_fnr:.4f} > 0); "
            "a refutation must never be downgraded"
        )
    if report.coverage < coverage_floor:
        raise GraderValidationError(
            f"grader coverage {report.coverage:.4f} below floor {coverage_floor} "
            "(abstention crutch: honesty without an answer rate)"
        )
