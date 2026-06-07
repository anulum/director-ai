# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Self-Healing Threshold Controller
"""Holdout-validated, auto-rolling-back online threshold adaptation.

Where ``OnlineCalibrator`` computes a recommended threshold from the durable
feedback store, this controller closes the loop *safely*: it proposes a new
operating threshold from a window of recent labelled outcomes, but only deploys
it after it beats the current threshold on a held-out split (so an update that
overfits the support window is rejected), and it rolls the policy back
automatically when a deployed update later regresses against the policy it
replaced. Every proposal, acceptance, rejection, and rollback is recorded for
audit — there is no blind self-mutation.

The adaptation is a meta-learning-style train/validate split: the best threshold
is found on the support split (the "adaptation") and validated on the holdout
split (the "query") before it is allowed to take effect. The split is
deterministic (by observation index), so the controller is fully reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ACCEPT = "accept"
REJECT = "reject"
ROLLBACK = "rollback"
INSUFFICIENT_DATA = "insufficient_data"
STABLE = "stable"
NO_PRIOR = "no_prior"


@dataclass(frozen=True)
class LabelledOutcome:
    """One labelled guard decision used to tune the operating threshold.

    ``score`` is the coherence score in ``[0, 1]``; ``grounded`` is the
    ground-truth label (``True`` when the answer was actually grounded). A
    threshold ``T`` approves when ``score >= T``.
    """

    score: float
    grounded: bool
    tenant_id: str = ""
    domain: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be a finite value in [0, 1]")


@dataclass(frozen=True)
class TuningConfig:
    """Policy for proposing, validating, and rolling back threshold updates."""

    holdout_fraction: float = 0.34
    false_halt_weight: float = 1.0
    missed_hallucination_weight: float = 1.0
    min_samples: int = 12
    regression_tolerance: float = 0.0
    threshold_min: float = 0.0
    threshold_max: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.holdout_fraction < 1.0:
            raise ValueError("holdout_fraction must be in (0, 1)")
        if self.min_samples < 2:
            raise ValueError("min_samples must be >= 2")
        if self.regression_tolerance < 0.0:
            raise ValueError("regression_tolerance must be >= 0")
        if not 0.0 <= self.threshold_min < self.threshold_max <= 1.0:
            raise ValueError("require 0 <= threshold_min < threshold_max <= 1")


@dataclass(frozen=True)
class PolicyUpdate:
    """The audited result of one propose/regression-check cycle."""

    action: str
    old_threshold: float
    new_threshold: float
    holdout_error_old: float
    holdout_error_new: float
    sample_count: int
    reason: str

    @property
    def changed(self) -> bool:
        """True when the controller's active threshold actually moved."""
        return self.action in (ACCEPT, ROLLBACK)

    def to_dict(self) -> dict[str, object]:
        """Serialisable audit record."""
        return {
            "action": self.action,
            "old_threshold": self.old_threshold,
            "new_threshold": self.new_threshold,
            "holdout_error_old": self.holdout_error_old,
            "holdout_error_new": self.holdout_error_new,
            "sample_count": self.sample_count,
            "reason": self.reason,
        }


class SelfHealingThresholdController:
    """Online threshold controller with a holdout deploy-gate and auto-rollback.

    Feed labelled outcomes with :meth:`observe`; call :meth:`propose` to evaluate
    the accumulated window and (only on a holdout improvement) move the threshold;
    call :meth:`evaluate_regression` with fresh outcomes to roll back a deployed
    update that has since regressed. The active threshold is :attr:`threshold`.
    """

    def __init__(
        self,
        initial_threshold: float,
        config: TuningConfig | None = None,
    ) -> None:
        cfg = config or TuningConfig()
        if not math.isfinite(initial_threshold) or not (
            cfg.threshold_min <= initial_threshold <= cfg.threshold_max
        ):
            raise ValueError("initial_threshold must lie within the configured bounds")
        self._config = cfg
        self._threshold = float(initial_threshold)
        self._previous: float | None = None
        self._buffer: list[LabelledOutcome] = []
        self._history: list[PolicyUpdate] = []

    @property
    def threshold(self) -> float:
        """The active operating threshold."""
        return self._threshold

    @property
    def previous_threshold(self) -> float | None:
        """The threshold this one replaced (the rollback target), if any."""
        return self._previous

    @property
    def pending(self) -> int:
        """Number of outcomes accumulated since the last proposal."""
        return len(self._buffer)

    @property
    def history(self) -> tuple[PolicyUpdate, ...]:
        """Every audited update decision, oldest first."""
        return tuple(self._history)

    def observe(self, outcome: LabelledOutcome) -> None:
        """Accumulate one labelled outcome for the next proposal."""
        self._buffer.append(outcome)

    def observe_many(self, outcomes: Iterable[LabelledOutcome]) -> None:
        """Accumulate several labelled outcomes."""
        self._buffer.extend(outcomes)

    def _weighted_error(
        self, threshold: float, outcomes: Sequence[LabelledOutcome]
    ) -> float:
        if not outcomes:
            return 0.0
        cfg = self._config
        penalty = 0.0
        for o in outcomes:
            approved = o.score >= threshold
            if o.grounded and not approved:
                penalty += cfg.false_halt_weight
            elif not o.grounded and approved:
                penalty += cfg.missed_hallucination_weight
        return penalty / len(outcomes)

    def _split(self) -> tuple[list[LabelledOutcome], list[LabelledOutcome]]:
        """Deterministic support/holdout split by observation index."""
        stride = max(2, round(1.0 / self._config.holdout_fraction))
        support: list[LabelledOutcome] = []
        holdout: list[LabelledOutcome] = []
        for i, outcome in enumerate(self._buffer):
            (holdout if i % stride == 0 else support).append(outcome)
        return support, holdout

    def _best_threshold(self, support: Sequence[LabelledOutcome]) -> float:
        cfg = self._config
        scores = sorted({o.score for o in support})
        candidates = {cfg.threshold_min, cfg.threshold_max, self._threshold}
        for score in scores:
            candidates.add(min(max(score, cfg.threshold_min), cfg.threshold_max))
        for a, b in zip(scores, scores[1:], strict=False):
            candidates.add((a + b) / 2.0)
        # Lowest support error; ties resolved toward the current threshold to
        # avoid needless churn.
        return min(
            sorted(candidates),
            key=lambda t: (self._weighted_error(t, support), abs(t - self._threshold)),
        )

    def propose(self) -> PolicyUpdate:
        """Evaluate the accumulated window and deploy a better threshold.

        Returns the audited decision. A window with fewer than ``min_samples``
        outcomes (or an empty support/holdout split) yields no change. Otherwise
        the best support-split threshold is deployed only if it strictly lowers
        the holdout error; either way the window is consumed.
        """
        count = len(self._buffer)
        if count < self._config.min_samples:
            return self._record(
                INSUFFICIENT_DATA,
                self._threshold,
                self._threshold,
                0.0,
                0.0,
                count,
                f"need >= {self._config.min_samples} outcomes, have {count}",
            )
        # min_samples >= 2 and the holdout stride >= 2 guarantee both splits are
        # non-empty (index 0 → holdout, index 1 → support).
        support, holdout = self._split()
        candidate = self._best_threshold(support)
        err_old = self._weighted_error(self._threshold, holdout)
        err_new = self._weighted_error(candidate, holdout)
        old = self._threshold
        self._buffer.clear()
        if candidate != old and err_new < err_old:
            self._previous = old
            self._threshold = candidate
            return self._record(
                ACCEPT,
                old,
                candidate,
                err_old,
                err_new,
                count,
                "holdout error improved",
            )
        return self._record(
            REJECT,
            old,
            old,
            err_old,
            err_new,
            count,
            "candidate did not improve holdout error",
        )

    def evaluate_regression(
        self, fresh_outcomes: Sequence[LabelledOutcome]
    ) -> PolicyUpdate:
        """Roll back the last deployed update if it regressed on fresh data.

        Compares the active threshold against the one it replaced on
        ``fresh_outcomes``; if the active policy is worse by more than
        ``regression_tolerance`` it is rolled back to its predecessor.
        """
        count = len(fresh_outcomes)
        if self._previous is None:
            return self._record(
                NO_PRIOR,
                self._threshold,
                self._threshold,
                0.0,
                0.0,
                count,
                "no deployed update to roll back",
            )
        err_current = self._weighted_error(self._threshold, fresh_outcomes)
        err_previous = self._weighted_error(self._previous, fresh_outcomes)
        if err_current > err_previous + self._config.regression_tolerance:
            current = self._threshold
            self._threshold = self._previous
            self._previous = None
            return self._record(
                ROLLBACK,
                current,
                self._threshold,
                err_current,
                err_previous,
                count,
                "deployed update regressed against its predecessor",
            )
        return self._record(
            STABLE,
            self._threshold,
            self._threshold,
            err_previous,
            err_current,
            count,
            "deployed update is holding up",
        )

    def audit(self) -> list[dict[str, object]]:
        """Return the full audit trail as serialisable records."""
        return [update.to_dict() for update in self._history]

    def _record(
        self,
        action: str,
        old: float,
        new: float,
        err_old: float,
        err_new: float,
        count: int,
        reason: str,
    ) -> PolicyUpdate:
        update = PolicyUpdate(
            action=action,
            old_threshold=old,
            new_threshold=new,
            holdout_error_old=err_old,
            holdout_error_new=err_new,
            sample_count=count,
            reason=reason,
        )
        self._history.append(update)
        return update
