# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — long-context intent-drift interlock

"""Catch slow-burn jailbreaks that no single turn reveals.

A crescendo attack never trips a per-turn guard: each request is a little
further from the declared intent than the last, every step individually benign.
The signal lives in the *trajectory* — sustained drift away from the opening
intent, a rising slope, accumulating cross-turn contradiction — not in any one
turn.

:class:`IntentDriftInterlock` folds three per-turn safety signals into a bounded,
fixed-size state that survives a 100k-token conversation without keeping the raw
history:

* **intent divergence** — how far this turn drifts from the running context
  (the cross-turn NLI divergence the scorer already computes);
* **injection risk** — the per-turn intent-grounded injection score;
* **contradiction trend** — the self-contradiction slope from
  :class:`~director_ai.core.runtime.contradiction_tracker.ContradictionTracker`.

It keeps an exponential moving average of the first two (so old turns decay
rather than being dropped), a windowed slope of the intent divergence (the
gradual-escalation signal), and trips when the combined ``drift_risk`` crosses
``trigger_threshold`` after at least ``min_turns`` turns — so a conversation that
creeps from 0.3 to 0.6 divergence over six turns is halted even though no single
turn ever clears a per-turn block.

The interlock is pure and numeric: the scorer feeds it pre-computed signals, so
it is fully deterministic and testable without any model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

__all__ = ["DriftState", "IntentDriftInterlock"]

# Weights for the combined drift risk (sum to 1.0).
_W_SUSTAINED = 0.4  # sustained divergence from intent (EMA)
_W_ESCALATION = 0.3  # rising slope of divergence (the crescendo signal)
_W_INJECTION = 0.2  # sustained injection pressure (EMA)
_W_CONTRADICTION = 0.1  # cross-turn self-contradiction trend
# Normalisers map a raw slope/trend onto [0, 1] before weighting.
_SLOPE_NORM = 0.3
_TREND_NORM = 0.3


@dataclass
class DriftState:
    """Compressed safety state after folding one turn (JSON-safe)."""

    turn_count: int
    sustained_divergence: float  # EMA of intent divergence
    escalation: float  # normalised positive slope of divergence [0, 1]
    injection_pressure: float  # EMA of injection risk
    contradiction_pressure: float  # normalised positive contradiction trend
    drift_risk: float  # combined [0, 1]
    triggered: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "turn_count": self.turn_count,
            "sustained_divergence": round(self.sustained_divergence, 4),
            "escalation": round(self.escalation, 4),
            "injection_pressure": round(self.injection_pressure, 4),
            "contradiction_pressure": round(self.contradiction_pressure, 4),
            "drift_risk": round(self.drift_risk, 4),
            "triggered": self.triggered,
        }


class IntentDriftInterlock:
    """Accumulate per-turn safety signals into a bounded drift state.

    Parameters
    ----------
    half_life_turns : float
        Turns over which an EMA signal decays to half weight (default 4).
    window : int
        Number of recent intent-divergence values kept for the slope estimate
        (default 8). Bounds memory to a fixed size regardless of conversation
        length.
    trigger_threshold : float
        ``drift_risk`` at or above which the interlock trips (default 0.6).
    min_turns : int
        Turns required before the interlock can trip, so a single off-topic
        opening turn never fires it (default 3).
    """

    def __init__(
        self,
        *,
        half_life_turns: float = 4.0,
        window: int = 8,
        trigger_threshold: float = 0.6,
        min_turns: int = 3,
    ) -> None:
        if half_life_turns <= 0:
            raise ValueError("half_life_turns must be positive")
        if window < 2:
            raise ValueError("window must be >= 2")
        if not 0.0 < trigger_threshold <= 1.0:
            raise ValueError("trigger_threshold must be in (0, 1]")
        if min_turns < 1:
            raise ValueError("min_turns must be >= 1")
        self._decay: float = 0.5 ** (1.0 / half_life_turns)
        self._window = window
        self._trigger_threshold = trigger_threshold
        self._min_turns = min_turns
        self._turn_count = 0
        self._ema_divergence = 0.0
        self._ema_injection = 0.0
        self._divergence_window: deque[float] = deque(maxlen=window)

    def update(
        self,
        *,
        intent_divergence: float,
        injection_risk: float = 0.0,
        contradiction_trend: float = 0.0,
    ) -> DriftState:
        """Fold one turn's signals and return the current drift state."""
        divergence = _clamp(intent_divergence)
        injection = _clamp(injection_risk)
        if self._turn_count == 0:
            self._ema_divergence = divergence
            self._ema_injection = injection
        else:
            self._ema_divergence = self._ema(self._ema_divergence, divergence)
            self._ema_injection = self._ema(self._ema_injection, injection)
        self._divergence_window.append(divergence)
        self._turn_count += 1

        escalation = _clamp(self._slope() / _SLOPE_NORM)
        contradiction_pressure = _clamp(max(0.0, contradiction_trend) / _TREND_NORM)
        drift_risk = _clamp(
            _W_SUSTAINED * self._ema_divergence
            + _W_ESCALATION * escalation
            + _W_INJECTION * self._ema_injection
            + _W_CONTRADICTION * contradiction_pressure
        )
        triggered = (
            self._turn_count >= self._min_turns
            and drift_risk >= self._trigger_threshold
        )
        return DriftState(
            turn_count=self._turn_count,
            sustained_divergence=self._ema_divergence,
            escalation=escalation,
            injection_pressure=self._ema_injection,
            contradiction_pressure=contradiction_pressure,
            drift_risk=drift_risk,
            triggered=triggered,
        )

    def reset(self) -> None:
        """Clear all accumulated state (e.g. on a new conversation)."""
        self._turn_count = 0
        self._ema_divergence = 0.0
        self._ema_injection = 0.0
        self._divergence_window.clear()

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def _ema(self, previous: float, signal: float) -> float:
        return self._decay * previous + (1.0 - self._decay) * signal

    def _slope(self) -> float:
        """Recent-minus-old mean of the divergence window (positive = rising)."""
        values = list(self._divergence_window)
        if len(values) < 4:
            return 0.0
        mid = len(values) // 2
        old_avg = sum(values[:mid]) / mid
        new_avg = sum(values[mid:]) / (len(values) - mid)
        return float(new_avg - old_avg)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
