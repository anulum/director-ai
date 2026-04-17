# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — EconomicRiskScorer

"""Compose the pool + bargaining + tragedy signals into one
score.

Three sub-signals enter the composite:

* ``exhaustion_headroom`` — ``1 - balance / capacity``. High
  means the pool is close to empty.
* ``fairness_gap`` — taken from the Nash bargaining solution if
  one was supplied. High means the bargained allocation favours
  one agent over others.
* ``tragedy_pressure`` — the detector's pressure signal.

The composite is a caller-weighted sum, clamped to ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bargaining import BargainingSolution
from .detector import TragedyDetector, TragedySignal
from .pool import ResourcePool


@dataclass(frozen=True)
class EconomicVerdict:
    """Outcome of one :meth:`EconomicRiskScorer.score` call."""

    risk: float
    exhaustion_headroom: float
    fairness_gap: float
    tragedy_pressure: float
    tragedy_firing: bool

    @property
    def safe(self) -> bool:
        return self.risk < 0.5


class EconomicRiskScorer:
    """Weighted-sum composite.

    Parameters
    ----------
    pool :
        The resource pool.
    detector :
        The tragedy detector.
    weight_exhaustion, weight_fairness, weight_tragedy :
        Non-negative weights that must sum to 1.0. Defaults
        0.4 / 0.2 / 0.4.
    """

    def __init__(
        self,
        *,
        pool: ResourcePool,
        detector: TragedyDetector,
        weight_exhaustion: float = 0.4,
        weight_fairness: float = 0.2,
        weight_tragedy: float = 0.4,
    ) -> None:
        weights = (weight_exhaustion, weight_fairness, weight_tragedy)
        for w in weights:
            if w < 0:
                raise ValueError("weights must be non-negative")
        total = sum(weights)
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"weights must sum to 1.0; got {total}")
        self._pool = pool
        self._detector = detector
        self._w_exh = weight_exhaustion
        self._w_fair = weight_fairness
        self._w_trag = weight_tragedy

    def score(
        self,
        *,
        bargaining: BargainingSolution | None = None,
    ) -> EconomicVerdict:
        exhaustion = 1.0 - (self._pool.balance() / self._pool.capacity)
        fairness_gap = bargaining.fairness_gap if bargaining is not None else 0.0
        tragedy: TragedySignal = self._detector.check()
        risk = (
            self._w_exh * exhaustion
            + self._w_fair * fairness_gap
            + self._w_trag * tragedy.pressure
        )
        risk = max(0.0, min(1.0, risk))
        return EconomicVerdict(
            risk=risk,
            exhaustion_headroom=exhaustion,
            fairness_gap=fairness_gap,
            tragedy_pressure=tragedy.pressure,
            tragedy_firing=tragedy.firing,
        )
