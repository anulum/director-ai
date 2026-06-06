# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — uncertainty-aware router

"""Route a calibrated hallucination interval to a downstream action.

Where :class:`~director_ai.core.routing.router.RiskRouter` routes *inputs* to a
scoring backend, this router acts on the *output*: it consumes the conformal
:class:`~director_ai.core.calibration.conformal.PredictionInterval` over
hallucination probability and turns its risk bounds into one of four actions.

* ``allow`` — the whole interval sits below ``allow_upper``: confidently
  low-risk, ship it.
* ``reject`` — the whole interval sits at or above ``reject_lower``:
  confidently high-risk, block it.
* ``escalate_human`` — the interval straddles the bounds and is too wide to
  resolve automatically (or the calibration is not yet reliable): send it to
  human review.
* ``escalate_model`` — the interval is uncertain but narrow enough to defer to
  a stronger model (LLM judge / ensemble) rather than a human.

The router is side-effect free and deterministic; dispatching the action to a
review queue or an LLM judge is the caller's job. Each decision records the
bounds it used so the routing rationale is auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..calibration.conformal import PredictionInterval

__all__ = ["UncertaintyAction", "UncertaintyDecision", "UncertaintyRouter"]

UncertaintyAction = Literal["allow", "reject", "escalate_human", "escalate_model"]


def _unit_interval(name: str, value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]; got {value!r}")
    return float(value)


@dataclass(frozen=True)
class UncertaintyDecision:
    """One uncertainty-routing outcome with the bounds that produced it."""

    action: UncertaintyAction
    point_estimate: float
    lower: float
    upper: float
    width: float
    is_reliable: bool
    reason: str


class UncertaintyRouter:
    """Map a conformal interval to an allow/reject/escalate action.

    Parameters
    ----------
    allow_upper :
        Interval upper bound at or below which the response is allowed.
        Default 0.2.
    reject_lower :
        Interval lower bound at or above which the response is rejected.
        Must be strictly greater than ``allow_upper``. Default 0.8.
    escalate_human_width :
        In the uncertain band, intervals at least this wide go to human
        review; narrower ones go to a stronger model. Default 0.5.
    """

    def __init__(
        self,
        *,
        allow_upper: float = 0.2,
        reject_lower: float = 0.8,
        escalate_human_width: float = 0.5,
    ) -> None:
        self._allow_upper = _unit_interval("allow_upper", allow_upper)
        self._reject_lower = _unit_interval("reject_lower", reject_lower)
        if self._allow_upper >= self._reject_lower:
            raise ValueError("allow_upper must be < reject_lower")
        if not 0.0 < escalate_human_width <= 1.0:
            raise ValueError(
                f"escalate_human_width must be in (0, 1]; got {escalate_human_width!r}"
            )
        self._escalate_human_width = float(escalate_human_width)

    def route(self, interval: PredictionInterval) -> UncertaintyDecision:
        """Return the routing decision for one conformal interval."""
        width = max(0.0, interval.upper - interval.lower)
        action, reason = self._classify(interval, width)
        return UncertaintyDecision(
            action=action,
            point_estimate=interval.point_estimate,
            lower=interval.lower,
            upper=interval.upper,
            width=width,
            is_reliable=interval.is_reliable,
            reason=reason,
        )

    def _classify(
        self, interval: PredictionInterval, width: float
    ) -> tuple[UncertaintyAction, str]:
        if not interval.is_reliable:
            return "escalate_human", (
                f"calibration unreliable ({interval.calibration_size} samples); "
                "defer to human review"
            )
        if interval.upper <= self._allow_upper:
            return "allow", (
                f"upper={interval.upper:.3f} <= allow_upper ({self._allow_upper:.3f})"
            )
        if interval.lower >= self._reject_lower:
            return "reject", (
                f"lower={interval.lower:.3f} >= reject_lower ({self._reject_lower:.3f})"
            )
        if width >= self._escalate_human_width:
            return "escalate_human", (
                f"uncertain interval width={width:.3f} >= "
                f"escalate_human_width ({self._escalate_human_width:.3f})"
            )
        return "escalate_model", (
            f"uncertain interval width={width:.3f} < "
            f"escalate_human_width ({self._escalate_human_width:.3f}); "
            "defer to stronger model"
        )
