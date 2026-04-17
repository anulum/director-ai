# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — recursive self-referential meta-guard

"""Monitor the guardrail's own decisions and auto-adjust its
thresholds when its scoring behaviour drifts away from the
calibration distribution.

Four pieces:

* :class:`ScoringDecision` — one record: prompt hash, raw score,
  action, calibrated ground-truth label if available, timestamp.
  The prompt text itself is not retained by default — the
  default privacy mode hashes it so the log can be audited
  without storing user content.
* :class:`DecisionLog` — thread-safe bounded append-only log.
  Windowed query helpers power the analyser.
* :class:`MetaAnalyzer` — three statistics over the current
  window: Page-Hinkley change-point detection on the score mean,
  Brier-score calibration drift against ground-truth labels when
  the operator supplies them, and per-action-rate drift
  (allow / warn / halt). Reports a
  :class:`MetaAnalysis` with every signal so the caller can
  decide whether to auto-adjust, page, or ignore.
* :class:`ThresholdAdjuster` — deterministic rule set for moving
  thresholds in response to drift, with a hysteresis interval
  that prevents the adjuster from oscillating on noise.
* :class:`MetaGuard` — orchestrator that binds a :class:`DecisionLog`,
  :class:`MetaAnalyzer`, and :class:`ThresholdAdjuster` into a
  single ``.record(decision)`` entry point. Callers fold new
  decisions in, and the guard emits a :class:`MetaVerdict` that
  names the new thresholds (or ``None`` when no adjustment is
  due).
"""

from .adjuster import ThresholdAdjuster, ThresholdBundle
from .analyzer import MetaAnalysis, MetaAnalyzer
from .guard import MetaGuard, MetaVerdict
from .log import DecisionLog, ScoringAction, ScoringDecision

__all__ = [
    "DecisionLog",
    "MetaAnalysis",
    "MetaAnalyzer",
    "MetaGuard",
    "MetaVerdict",
    "ScoringAction",
    "ScoringDecision",
    "ThresholdAdjuster",
    "ThresholdBundle",
]
