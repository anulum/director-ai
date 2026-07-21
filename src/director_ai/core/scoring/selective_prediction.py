# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Selective Prediction Metrics

"""Risk-coverage metrics for a guardrail that can abstain.

A guardrail that abstains (KIMI3-abstain: a configured store produced no
usable context, so the factual signal is the neutral mid-score) trades
coverage for accuracy: the verdicts it *does* make should be more reliable
because it declined the ungrounded ones. This module turns a batch of
:class:`~director_ai.core.types.CoherenceScore` verdicts, each paired with its
ground-truth hallucination label, into the standard selective-prediction pair:

* **coverage** — the fraction of inputs the guardrail actually judged (did not
  abstain on);
* **selective accuracy** — accuracy over that covered subset.

Reporting both keeps an abstaining guardrail honest: a high selective accuracy
is only meaningful alongside the coverage it was achieved at.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..types import CoherenceScore


@dataclass(frozen=True)
class SelectivePredictionReport:
    """Risk-coverage summary for a batch of abstention-aware verdicts."""

    total: int
    covered: int  # verdicts actually made (not abstained)
    abstained: int
    coverage: float  # covered / total
    selective_accuracy: float | None  # accuracy on the covered subset
    selective_error: float | None  # 1 - selective_accuracy
    overall_accuracy: float  # accuracy over all inputs (abstentions = errors)

    def to_dict(self) -> dict[str, float | int | None]:
        """Render to a JSON-serialisable summary."""
        return {
            "total": self.total,
            "covered": self.covered,
            "abstained": self.abstained,
            "coverage": round(self.coverage, 6),
            "selective_accuracy": (
                None
                if self.selective_accuracy is None
                else round(self.selective_accuracy, 6)
            ),
            "selective_error": (
                None if self.selective_error is None else round(self.selective_error, 6)
            ),
            "overall_accuracy": round(self.overall_accuracy, 6),
        }


def selective_prediction_metrics(
    results: Sequence[tuple[CoherenceScore, bool]],
) -> SelectivePredictionReport:
    """Compute risk-coverage metrics over ``(score, is_hallucinated)`` pairs.

    A verdict is *correct* when the guardrail's approval matches the ground
    truth — it approved a grounded response (``is_hallucinated`` false) or
    rejected a hallucination (``is_hallucinated`` true). Abstained verdicts
    (``score.abstained``) are excluded from selective accuracy but still count
    against coverage; ``overall_accuracy`` treats every abstention as a miss so
    a guardrail cannot inflate its score by abstaining on the hard cases.

    Parameters
    ----------
    results:
        ``(CoherenceScore, is_hallucinated)`` pairs. ``is_hallucinated`` is the
        ground-truth label: ``True`` when the response should be rejected.

    Returns
    -------
    SelectivePredictionReport
    """
    if not results:
        raise ValueError("selective_prediction_metrics requires at least one result")

    total = len(results)
    covered_correct = 0
    covered = 0
    overall_correct = 0
    for score, is_hallucinated in results:
        correct = bool(score.approved) == (not is_hallucinated)
        if score.abstained:
            continue
        covered += 1
        if correct:
            covered_correct += 1
            overall_correct += 1

    abstained = total - covered
    coverage = covered / total
    selective_accuracy = covered_correct / covered if covered else None
    selective_error = None if selective_accuracy is None else 1.0 - selective_accuracy
    overall_accuracy = overall_correct / total
    return SelectivePredictionReport(
        total=total,
        covered=covered,
        abstained=abstained,
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        selective_error=selective_error,
        overall_accuracy=overall_accuracy,
    )
