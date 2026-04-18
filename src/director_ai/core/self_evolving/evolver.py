# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — SelfEvolver orchestrator

"""Close the loop — pull recent failures, synthesise adversarial
variants, retrain the guardrail, re-calibrate, hot-swap.

The orchestrator is deliberately deterministic: every stage is
seeded, every stage exposes its result on the returned
:class:`EvolutionReport`, and the hot-swap is atomic (delegated
to the caller's registry, which is assumed to be thread-safe —
the shipped :class:`~director_ai.core.policy_compiler.PolicyRegistry`
is one such registry).
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass, field

from .adversarial import AdversarialGenerator, PerturbativeAdversarialGenerator
from .calibration import ConformalCalibrator, ConformalResult
from .feedback import FeedbackEvent, FeedbackLabel, FeedbackStore
from .trainer import (
    GuardrailTrainer,
    PerceptronGuardrailTrainer,
    TrainedGuardrail,
)


@dataclass(frozen=True)
class EvolutionReport:
    """Full trace of one :meth:`SelfEvolver.evolve` call."""

    guardrail: TrainedGuardrail
    threshold: float
    adversarial_samples: tuple[str, ...]
    feedback_seen: int
    calibration_size: int
    failure_labels: tuple[FeedbackLabel, ...] = field(default_factory=tuple)

    @property
    def conformal(self) -> ConformalResult:
        return ConformalResult(
            threshold=self.threshold,
            coverage_target=0.90,
            calibration_size=self.calibration_size,
        )


class SelfEvolver:
    """Orchestrate one evolution round.

    Parameters
    ----------
    store :
        The :class:`FeedbackStore` to pull training data from.
    trainer :
        The :class:`GuardrailTrainer`. Default
        :class:`PerceptronGuardrailTrainer`.
    adversarial :
        The :class:`AdversarialGenerator`. Default
        :class:`PerturbativeAdversarialGenerator`.
    calibrator :
        Optional :class:`ConformalCalibrator`. Default targets
        0.90 coverage.
    failure_labels :
        Labels that count as "failures" for the adversarial
        generator's seed set. Default
        ``("unsafe", "false_negative")``.
    adversarial_per_evolution :
        Upper bound on adversarial prompts generated per round.
        Default 64.
    min_feedback :
        Minimum number of labelled events the store must hold
        before :meth:`evolve` will run. Default 16.
    """

    def __init__(
        self,
        *,
        store: FeedbackStore,
        trainer: GuardrailTrainer | None = None,
        adversarial: AdversarialGenerator | None = None,
        calibrator: ConformalCalibrator | None = None,
        failure_labels: Iterable[FeedbackLabel] = ("unsafe", "false_negative"),
        adversarial_per_evolution: int = 64,
        min_feedback: int = 16,
    ) -> None:
        if adversarial_per_evolution <= 0:
            raise ValueError(
                f"adversarial_per_evolution must be positive; "
                f"got {adversarial_per_evolution!r}"
            )
        if min_feedback <= 0:
            raise ValueError(f"min_feedback must be positive; got {min_feedback!r}")
        self._store = store
        self._trainer: GuardrailTrainer = trainer or PerceptronGuardrailTrainer()
        self._adversarial: AdversarialGenerator = (
            adversarial or PerturbativeAdversarialGenerator()
        )
        self._calibrator = calibrator or ConformalCalibrator()
        self._failure_labels = tuple(failure_labels)
        self._adv_budget = adversarial_per_evolution
        self._min_feedback = min_feedback
        self._version = 0
        self._lock = threading.Lock()

    def evolve(self, *, seed: int = 0) -> EvolutionReport:
        """Run one evolution round and return the report.

        Raises :class:`ValueError` when the feedback store holds
        fewer than ``min_feedback`` events — the trainer will not
        produce a useful guardrail on a microscopic set."""
        if len(self._store) < self._min_feedback:
            raise ValueError(
                f"feedback store holds {len(self._store)} events; "
                f"need at least {self._min_feedback}"
            )
        failures: list[FeedbackEvent] = []
        for label in self._failure_labels:
            failures.extend(self._store.iter_labelled(label))
        adversarial = self._adversarial.generate(
            failures, max_samples=self._adv_budget, seed=seed
        )
        all_events = tuple(self._store.iter_all())
        synthesised = tuple(
            FeedbackEvent(prompt=p, response="", label="unsafe") for p in adversarial
        )
        training_set = all_events + synthesised

        # Split 80 / 20 deterministically on the combined set so the
        # conformal calibration sees unseen data.
        split = max(int(0.8 * len(training_set)), 1)
        train_events = training_set[:split]
        calibration_events = training_set[split:]
        if not calibration_events:
            # Tiny store — reuse the training fold for calibration.
            # The coverage guarantee weakens but the loop stays live.
            calibration_events = train_events

        with self._lock:
            self._version += 1
            version = self._version
        guardrail = self._trainer.train(train_events, version=version)
        conformal = self._calibrator.calibrate(guardrail, calibration_events)
        return EvolutionReport(
            guardrail=guardrail,
            threshold=conformal.threshold,
            adversarial_samples=adversarial,
            feedback_seen=len(all_events),
            calibration_size=conformal.calibration_size,
            failure_labels=self._failure_labels,
        )
