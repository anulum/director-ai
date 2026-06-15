# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — runtime threshold governor (live segmented-threshold wiring)

"""Apply learned per-segment thresholds to the live runtime, under control.

:class:`~director_ai.core.calibration.segmented_threshold.SegmentedThresholdLearner`
learns a halt threshold per domain/model/tenant from feedback, but a recommendation
is not something to slam into production the instant it appears: an unguarded
auto-tune oscillates, over-fits a noisy hour, and offers no audit trail. This
governor is the change-management overlay between the learner and the live
decision path.

It holds the threshold each segment is *currently* using and changes it only
through a controlled process:

* a change is proposed only once a segment has its own evidence (the
  recommendation's source is ``"segment"``, not the global cold-start pool) and
  the learner actually recommends a different threshold;
* each applied change moves at most ``max_step`` toward the recommendation, so the
  live threshold ramps rather than jumps;
* a recommendation that ``requires_human_approval`` is held as pending until
  :meth:`apply` is called with ``approve=True`` (unless the governor is
  constructed with ``auto_apply=True`` for self-tuning deployments);
* every applied change is recorded in an audit :meth:`history`.

At decision time :meth:`effective_threshold` returns the live segment threshold,
optionally tightened by a conformal-uncertainty tie-in: when a
:class:`~director_ai.core.calibration.conformal.PredictionInterval` is supplied
and an :class:`~director_ai.core.routing.uncertainty_router.UncertaintyRouter`
flags the estimate as unreliable or too wide, the threshold is lowered by
``uncertainty_penalty`` (halt more readily) and the router's action is surfaced —
so an uncertain request is judged more conservatively than a confident one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from director_ai.core.calibration.segmented_threshold import SegmentedThresholdLearner

__all__ = [
    "ThresholdChange",
    "EffectiveThreshold",
    "RuntimeThresholdGovernor",
]


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class ThresholdChange:
    """One audited threshold-change decision for a segment."""

    segment: str
    from_threshold: float
    to_threshold: float
    recommended_threshold: float | None
    applied: bool
    requires_approval: bool
    source: str
    reason: str
    at: float


@dataclass(frozen=True)
class EffectiveThreshold:
    """The threshold to use for a request, after the uncertainty tie-in."""

    segment: str
    base_threshold: float
    threshold: float
    uncertainty_adjusted: bool
    action: str | None
    reason: str


@dataclass
class RuntimeThresholdGovernor:
    """Controlled live application of learned per-segment thresholds.

    Parameters
    ----------
    learner:
        The :class:`SegmentedThresholdLearner` supplying recommendations.
    current_threshold:
        The threshold a segment uses before any change is applied (the same value
        the learner was built with).
    max_step:
        Maximum absolute move of the live threshold per applied change, in
        ``(0, 1]``; bounds how fast the runtime can drift.
    auto_apply:
        When ``True``, recommendations that would otherwise require human approval
        are applied automatically (self-tuning deployments). Default ``False``.
    uncertainty_router:
        Optional :class:`UncertaintyRouter` for the conformal tie-in.
    uncertainty_penalty:
        How much to lower the threshold when the conformal interval is uncertain,
        in ``[0, 1]``.
    clock:
        Injectable ``() -> float`` for audit timestamps (tests pass a fake).
    """

    learner: SegmentedThresholdLearner
    current_threshold: float
    max_step: float = 0.05
    auto_apply: bool = False
    uncertainty_router: object | None = None
    uncertainty_penalty: float = 0.1
    clock: Callable[[], float] = time.monotonic
    _live: dict[str, float] = field(default_factory=dict, repr=False)
    _history: list[ThresholdChange] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.current_threshold <= 1.0:
            raise ValueError("current_threshold must be in [0, 1]")
        if not 0.0 < self.max_step <= 1.0:
            raise ValueError("max_step must be in (0, 1]")
        if not 0.0 <= self.uncertainty_penalty <= 1.0:
            raise ValueError("uncertainty_penalty must be in [0, 1]")

    def live_threshold(self, segment: str) -> float:
        """The threshold *segment* is currently using."""
        return self._live.get(segment, self.current_threshold)

    def observe(self, *, segment: str, score: float, human_approved: bool) -> None:
        """Forward one labelled outcome to the learner for *segment*."""
        self.learner.observe(score, human_approved, segment=segment)

    def _step_toward(self, current: float, target: float) -> float:
        delta = target - current
        if abs(delta) > self.max_step:
            delta = self.max_step if delta > 0 else -self.max_step
        return _clamp_unit(current + delta)

    def propose(self, segment: str) -> ThresholdChange:
        """Compute (and, when permitted, apply) the next change for *segment*."""
        rec = self.learner.recommend(segment=segment)
        recommendation = rec.recommendation
        current = self.live_threshold(segment)
        target = recommendation.recommended_threshold

        if rec.source != "segment" or target is None:
            reason = (
                "insufficient segment evidence"
                if rec.source != "segment"
                else "no change recommended"
            )
            return self._record(
                segment,
                current,
                current,
                target,
                False,
                requires_approval=False,
                source=rec.source,
                reason=reason,
            )

        requires_approval = (
            recommendation.requires_human_approval and not self.auto_apply
        )
        stepped = self._step_toward(current, target)
        if requires_approval:
            return self._record(
                segment,
                current,
                current,
                target,
                False,
                requires_approval=True,
                source=rec.source,
                reason="pending human approval",
            )
        self._live[segment] = stepped
        return self._record(
            segment,
            current,
            stepped,
            target,
            True,
            requires_approval=False,
            source=rec.source,
            reason=recommendation.reason,
        )

    def apply(self, segment: str, *, approve: bool = False) -> ThresholdChange:
        """Apply the current recommendation for *segment* with explicit approval."""
        rec = self.learner.recommend(segment=segment)
        recommendation = rec.recommendation
        current = self.live_threshold(segment)
        target = recommendation.recommended_threshold
        if rec.source != "segment" or target is None:
            return self._record(
                segment,
                current,
                current,
                target,
                False,
                requires_approval=False,
                source=rec.source,
                reason="nothing to apply",
            )
        if recommendation.requires_human_approval and not (approve or self.auto_apply):
            return self._record(
                segment,
                current,
                current,
                target,
                False,
                requires_approval=True,
                source=rec.source,
                reason="approval required",
            )
        stepped = self._step_toward(current, target)
        self._live[segment] = stepped
        return self._record(
            segment,
            current,
            stepped,
            target,
            True,
            requires_approval=False,
            source=rec.source,
            reason=recommendation.reason,
        )

    def effective_threshold(
        self, segment: str, *, interval: object | None = None
    ) -> EffectiveThreshold:
        """Return the threshold to use now, tightened on conformal uncertainty."""
        base = self.live_threshold(segment)
        if interval is None or self.uncertainty_router is None:
            return EffectiveThreshold(
                segment=segment,
                base_threshold=base,
                threshold=base,
                uncertainty_adjusted=False,
                action=None,
                reason="no uncertainty signal",
            )
        decision = self.uncertainty_router.route(interval)
        if decision.action in ("escalate_human", "escalate_model"):
            tightened = _clamp_unit(base - self.uncertainty_penalty)
            return EffectiveThreshold(
                segment=segment,
                base_threshold=base,
                threshold=tightened,
                uncertainty_adjusted=True,
                action=decision.action,
                reason=f"uncertain ({decision.reason}); tightened by "
                f"{self.uncertainty_penalty}",
            )
        return EffectiveThreshold(
            segment=segment,
            base_threshold=base,
            threshold=base,
            uncertainty_adjusted=False,
            action=decision.action,
            reason=decision.reason,
        )

    def history(self, segment: str | None = None) -> tuple[ThresholdChange, ...]:
        """Audit trail of changes, all segments or one."""
        if segment is None:
            return tuple(self._history)
        return tuple(c for c in self._history if c.segment == segment)

    def _record(
        self,
        segment: str,
        from_t: float,
        to_t: float,
        recommended: float | None,
        applied: bool,
        *,
        requires_approval: bool,
        source: str,
        reason: str,
    ) -> ThresholdChange:
        change = ThresholdChange(
            segment=segment,
            from_threshold=round(from_t, 6),
            to_threshold=round(to_t, 6),
            recommended_threshold=(
                None if recommended is None else round(recommended, 6)
            ),
            applied=applied,
            requires_approval=requires_approval,
            source=source,
            reason=reason,
            at=self.clock(),
        )
        if applied:
            self._history.append(change)
        return change
