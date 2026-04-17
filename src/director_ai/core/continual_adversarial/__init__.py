# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — continual adversarial evolution

"""Continually mined adversarial test suites distilled from
production failure events.

* :class:`FailureEvent` — one production failure record: the
  prompt that slipped past the guardrail, the expected label,
  and a timestamp.
* :class:`FailureStore` — thread-safe bounded log.
* :class:`PatternMiner` — extracts recurring patterns via two
  passes: token-n-gram frequency with a minimum support
  threshold, and normalised-edit-distance clustering for prompts
  that share structure without sharing exact n-grams.
* :class:`AdversarialSuite` — versioned test-case container. Each
  version carries its generation timestamp + mined patterns +
  promotion reason, so operators can diff suites across time.
* :class:`AdversaryScorer` — perceptron distilled from the mined
  pattern labels; returns the probability a fresh prompt
  matches the adversarial distribution.
* :class:`ContinualEngine` — the cycle that turns new failures
  into a new suite + a retrained scorer.
"""

from .engine import ContinualEngine, EvolveReport
from .failure import FailureEvent, FailureStore
from .miner import FailurePattern, PatternMiner
from .scorer import (
    AdversaryScorer,
    PerceptronAdversaryScorer,
    TrainedAdversaryScorer,
)
from .suite import AdversarialCase, AdversarialSuite, SuiteVersion

__all__ = [
    "AdversaryScorer",
    "AdversarialCase",
    "AdversarialSuite",
    "ContinualEngine",
    "EvolveReport",
    "FailureEvent",
    "FailurePattern",
    "FailureStore",
    "PatternMiner",
    "PerceptronAdversaryScorer",
    "SuiteVersion",
    "TrainedAdversaryScorer",
]
