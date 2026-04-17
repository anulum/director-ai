# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ThresholdAdjuster

"""Rule-based threshold adjustment with hysteresis.

The adjuster is deliberately mechanical — no learning loop. It
consumes a :class:`MetaAnalysis` and returns a new
:class:`ThresholdBundle` when the observed drift justifies one.
The hysteresis interval prevents oscillation: a threshold can
move at most ``max_step`` per call and is held steady until
consecutive calls agree on direction.
"""

from __future__ import annotations

from dataclasses import dataclass

from .analyzer import MetaAnalysis


@dataclass(frozen=True)
class ThresholdBundle:
    """Two-band threshold pair used by the scoring layer.

    ``warn_threshold`` is the score above which a prompt warns.
    ``halt_threshold`` is the score above which the prompt halts.
    Must satisfy ``0 <= warn < halt <= 1``.
    """

    warn_threshold: float
    halt_threshold: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.warn_threshold < self.halt_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= warn < halt <= 1; got "
                f"({self.warn_threshold}, {self.halt_threshold})"
            )


class ThresholdAdjuster:
    """Produce a new :class:`ThresholdBundle` when drift demands it.

    Policy:

    * Mean score drifting *above* the reference with a
      Page-Hinkley alarm → tighten (raise) warn + halt by up to
      ``max_step`` each.
    * Mean score drifting *below* with a Page-Hinkley alarm →
      loosen (lower) warn + halt.
    * Brier alarm without a direction signal → narrow the
      warn-halt gap by ``max_step / 2`` on each side so the
      "uncertain" band shrinks while calibration is restored.
    * Action-rate alarm alone is reported but does not move
      thresholds — the policy is to page the operator rather
      than mutate thresholds on a distribution shift that could
      be legitimate traffic change.

    Parameters
    ----------
    initial :
        Starting thresholds. Kept as the hysteresis anchor.
    max_step :
        Largest single-call threshold move. Default 0.02 —
        conservative; combined with the two-strike hysteresis,
        the adjuster takes four drifting windows to move
        ``max_step * 2`` total.
    hysteresis_strikes :
        Consecutive analyses with the same drift direction
        required before the adjuster acts. Default 2 — one-off
        spikes pass through untouched.
    floor_warn :
        Minimum value for the warn threshold. Default 0.05.
    ceiling_halt :
        Maximum value for the halt threshold. Default 0.98.
    """

    def __init__(
        self,
        *,
        initial: ThresholdBundle,
        max_step: float = 0.02,
        hysteresis_strikes: int = 2,
        floor_warn: float = 0.05,
        ceiling_halt: float = 0.98,
    ) -> None:
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        if hysteresis_strikes <= 0:
            raise ValueError("hysteresis_strikes must be positive")
        if not 0.0 <= floor_warn < ceiling_halt <= 1.0:
            raise ValueError(
                "floor_warn / ceiling_halt must satisfy "
                "0 <= floor_warn < ceiling_halt <= 1"
            )
        self._max_step = max_step
        self._strike_target = hysteresis_strikes
        self._floor_warn = floor_warn
        self._ceiling_halt = ceiling_halt
        self._current = initial
        self._last_direction: int = 0
        self._consecutive_strikes: int = 0

    @property
    def current(self) -> ThresholdBundle:
        return self._current

    def observe(self, analysis: MetaAnalysis) -> ThresholdBundle | None:
        """Fold ``analysis`` into the hysteresis state machine and
        return a new :class:`ThresholdBundle` when the adjuster
        has crossed its strike target; otherwise ``None``."""
        direction = self._classify(analysis)
        if direction == 0:
            self._last_direction = 0
            self._consecutive_strikes = 0
            return None
        if direction == self._last_direction:
            self._consecutive_strikes += 1
        else:
            self._last_direction = direction
            self._consecutive_strikes = 1
        if self._consecutive_strikes < self._strike_target:
            return None
        new_bundle = self._move(direction, analysis)
        self._current = new_bundle
        self._consecutive_strikes = 0
        return new_bundle

    def _classify(self, analysis: MetaAnalysis) -> int:
        """Return +1 to tighten, -1 to loosen, 0 to hold."""
        if analysis.page_hinkley_alarm:
            if analysis.mean_score > 0.0:
                # The Page-Hinkley alarm fires regardless of
                # direction; compare the observed mean to the
                # current warn threshold to decide.
                return 1 if analysis.mean_score >= self._current.warn_threshold else -1
        if analysis.brier_alarm:
            # Narrow the band — direction +2 is the signal for a
            # centre squeeze.
            return 2
        return 0

    def _move(self, direction: int, analysis: MetaAnalysis) -> ThresholdBundle:
        warn = self._current.warn_threshold
        halt = self._current.halt_threshold
        if direction == 1:  # tighten: raise warn + halt
            warn = min(warn + self._max_step, self._ceiling_halt - 1e-6)
            halt = min(halt + self._max_step, self._ceiling_halt)
        elif direction == -1:  # loosen: lower warn + halt
            warn = max(warn - self._max_step, self._floor_warn)
            halt = max(halt - self._max_step, self._floor_warn + 1e-6)
        elif direction == 2:  # centre squeeze
            centre = 0.5 * (warn + halt)
            half_step = 0.5 * self._max_step
            warn = min(warn + half_step, centre - 1e-6)
            halt = max(halt - half_step, centre + 1e-6)
        # Bound and order-enforce after every move.
        warn = max(self._floor_warn, min(warn, self._ceiling_halt - 1e-6))
        halt = max(warn + 1e-6, min(halt, self._ceiling_halt))
        return ThresholdBundle(warn_threshold=warn, halt_threshold=halt)
