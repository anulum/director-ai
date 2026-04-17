# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — self-evolving online guardrail

"""Close the loop on a production guardrail: collect real failures,
synthesise adversarial variants, run a LoRA micro-fine-tune, and
update the conformal threshold — all atomic, all versioned.

Three Protocol boundaries let operators swap implementations
without touching the orchestrator:

* :class:`FeedbackStore` — append-only event log. The shipped
  :class:`InMemoryFeedbackStore` is thread-safe and indexed by
  label so adversarial generation can sample failures cheaply.
  A :class:`JSONLFeedbackStore` mirrors the same API but persists
  to disk so long-lived deployments survive a restart.
* :class:`AdversarialGenerator` — produce adversarial prompt
  variants from a seed set of observed failures. The shipped
  :class:`PerturbativeAdversarialGenerator` runs ten deterministic
  mutations (character swap, token drop, casing flip, marker
  injection, paraphrase scaffolds) with a seeded RNG so CI runs
  are reproducible.
* :class:`GuardrailTrainer` — drop-in for a LoRA micro-fine-tune.
  The shipped :class:`PerceptronGuardrailTrainer` is a fully
  functional online perceptron that trains on the collected
  feedback, returns a :class:`TrainedGuardrail` with a callable
  ``.score(text)`` method, and exposes the trained weights for
  audit. :class:`LoraGuardrailTrainer` slots in the ``peft``
  stack via lazy import for production deployments with a GPU.

The :class:`SelfEvolver` orchestrator runs one round:

1. Pull recent failures from the store.
2. Generate adversarial variants.
3. Train a new guardrail.
4. Calibrate a conformal threshold on a held-out fold.
5. Atomically hot-swap into the caller-supplied registry.
"""

from .adversarial import (
    AdversarialGenerator,
    PerturbativeAdversarialGenerator,
)
from .calibration import ConformalCalibrator
from .evolver import EvolutionReport, SelfEvolver
from .feedback import (
    FeedbackEvent,
    FeedbackLabel,
    FeedbackStore,
    InMemoryFeedbackStore,
    JSONLFeedbackStore,
)
from .trainer import (
    GuardrailTrainer,
    LoraGuardrailTrainer,
    PerceptronGuardrailTrainer,
    TrainedGuardrail,
)

__all__ = [
    "AdversarialGenerator",
    "ConformalCalibrator",
    "EvolutionReport",
    "FeedbackEvent",
    "FeedbackLabel",
    "FeedbackStore",
    "GuardrailTrainer",
    "InMemoryFeedbackStore",
    "JSONLFeedbackStore",
    "LoraGuardrailTrainer",
    "PerceptronGuardrailTrainer",
    "PerturbativeAdversarialGenerator",
    "SelfEvolver",
    "TrainedGuardrail",
]
