# SPDX-License-Identifier: LicenseRef-BUSL-1.1-ANULUM-TDM
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — SelfEvolver orchestrator

"""Orchestrate one guardrail evolution round end to end.

Split the feedback by source cluster, synthesise adversarial variants
from the training clusters, retrain the guardrail, calibrate on the
held-out real clusters, and hot-swap.

The orchestrator is deliberately deterministic: every stage is
seeded, every stage exposes its result on the returned
:class:`EvolutionReport`, and the hot-swap is atomic (delegated
to the caller's registry, which is assumed to be thread-safe —
the shipped :class:`~director_ai.core.policy_compiler.PolicyRegistry`
is one such registry).

Calibration independence
------------------------

Events from one source cluster depend on each other, so the independent
unit of calibration is the cluster group, not the event. Under cluster
exchangeability, one uniformly chosen score per held-out group is exactly
exchangeable with a uniformly chosen event of a new cluster, and the
split-conformal coverage guarantee holds with ``n`` equal to the number of
held-out groups. The guarantee is cluster-weighted: it covers an
event drawn by first choosing a new cluster and then an event in it, and
per-event coverage can differ when a few long conversations dominate
traffic. Each round therefore:

* splits whole source clusters (``metadata["source_cluster"]``), never
  single rows, and merges clusters whose prompts coincide after case
  folding and whitespace normalisation;
* holds out groups whose events all carry a recorded cluster, while
  events without a recorded cluster stay in training;
* calibrates on one seeded choice of event per held-out group;
* generates adversarial rows only from training-fold failures, keeps
  them in training, and drops any variant that coincides with a
  held-out prompt;
* refuses before any training or version change when fewer held-out
  groups exist than the calibrator's coverage target needs, or when no
  real event is left for training.

Two residuals remain. Prompts that are near-duplicates but differ after
normalisation, such as an edit or a paraphrase, are not merged across
clusters. A synthetic variant close to a held-out prompt, but not equal
to it after normalisation, can still train.
"""

from __future__ import annotations

import math
import random
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import get_args

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
    """Full trace of one :meth:`SelfEvolver.evolve` call.

    ``calibration_clusters`` counts the independent source-cluster
    groups held out for calibration. Each contributes one score, so it
    equals ``calibration_size``. ``unclustered_training_events`` counts
    real events without a recorded cluster, which stayed in training.
    ``coverage_target`` is the calibrator's ``1 - α``.
    """

    guardrail: TrainedGuardrail
    threshold: float
    adversarial_samples: tuple[str, ...]
    feedback_seen: int
    calibration_size: int
    failure_labels: tuple[FeedbackLabel, ...] = field(default_factory=tuple)
    calibration_clusters: int = 0
    unclustered_training_events: int = 0
    coverage_target: float = 0.90

    @property
    def conformal(self) -> ConformalResult:
        """Return this round's calibration as a ConformalResult view."""
        return ConformalResult(
            threshold=self.threshold,
            coverage_target=self.coverage_target,
            calibration_size=self.calibration_size,
        )


@dataclass(frozen=True)
class _FoldPlan:
    """Cluster-level split of the real feedback for one round.

    ``calibration`` holds one representative event per held-out group;
    ``calibration_keys`` covers the prompts of every held-out event.
    """

    training: tuple[FeedbackEvent, ...]
    calibration: tuple[FeedbackEvent, ...]
    calibration_clusters: int
    unclustered: int
    calibration_keys: frozenset[str]


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
        0.90 coverage. Its ``minimum_calibration_size`` sets the
        smallest number of independent held-out cluster groups a
        round accepts.
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
    calibration_fraction :
        Share of the clustered real events to hold out for
        calibration, in ``(0, 1)``. Whole cluster groups are held out
        until both this share and the calibrator's minimum number of
        groups are met. Default 0.2.
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
        calibration_fraction: float = 0.2,
    ) -> None:
        if adversarial_per_evolution <= 0:
            raise ValueError(
                f"adversarial_per_evolution must be positive; "
                f"got {adversarial_per_evolution!r}"
            )
        if min_feedback <= 0:
            raise ValueError(f"min_feedback must be positive; got {min_feedback!r}")
        if not 0.0 < calibration_fraction < 1.0:
            raise ValueError(
                f"calibration_fraction must be in (0, 1); got {calibration_fraction!r}"
            )
        labels = tuple(failure_labels)
        known = get_args(FeedbackLabel)
        unknown = sorted(label for label in labels if label not in known)
        if unknown:
            raise ValueError(f"unknown failure labels: {unknown}")
        self._store = store
        self._trainer: GuardrailTrainer = trainer or PerceptronGuardrailTrainer()
        self._adversarial: AdversarialGenerator = (
            adversarial or PerturbativeAdversarialGenerator()
        )
        self._calibrator = calibrator or ConformalCalibrator()
        self._failure_labels = labels
        self._adv_budget = adversarial_per_evolution
        self._min_feedback = min_feedback
        self._calibration_fraction = calibration_fraction
        self._version = 0
        self._lock = threading.Lock()

    def evolve(self, *, seed: int = 0) -> EvolutionReport:
        """Run one evolution round and return the report.

        Raises :class:`ValueError` when the feedback store holds
        fewer than ``min_feedback`` events, or when the cluster split
        cannot hold out as many independent cluster groups as the
        calibrator's coverage target needs while keeping real events for
        training. Both refusals happen before any training or version
        change.
        """
        if len(self._store) < self._min_feedback:
            raise ValueError(
                f"feedback store holds {len(self._store)} events; "
                f"need at least {self._min_feedback}"
            )
        all_events = tuple(self._store.iter_all())
        plan = _plan_folds(
            all_events,
            fraction=self._calibration_fraction,
            min_groups=self._calibrator.minimum_calibration_size,
            seed=seed,
        )
        failures = [
            event
            for label in self._failure_labels
            for event in plan.training
            if event.label == label
        ]
        generated = self._adversarial.generate(
            failures, max_samples=self._adv_budget, seed=seed
        )
        adversarial = tuple(
            prompt
            for prompt in generated
            if _visible_key(prompt) not in plan.calibration_keys
        )
        synthesised = tuple(
            FeedbackEvent(prompt=p, response="", label="unsafe") for p in adversarial
        )
        # Seeded shuffle keeps the round reproducible while mixing real and
        # synthetic rows for order-sensitive trainers. Synthetic rows never
        # enter calibration.
        training_set = list(plan.training + synthesised)
        random.Random(seed).shuffle(training_set)

        with self._lock:
            self._version += 1
            version = self._version
        guardrail = self._trainer.train(training_set, version=version)
        conformal = self._calibrator.calibrate(guardrail, plan.calibration)
        return EvolutionReport(
            guardrail=guardrail,
            threshold=conformal.threshold,
            adversarial_samples=adversarial,
            feedback_seen=len(all_events),
            calibration_size=conformal.calibration_size,
            failure_labels=self._failure_labels,
            calibration_clusters=plan.calibration_clusters,
            unclustered_training_events=plan.unclustered,
            coverage_target=conformal.coverage_target,
        )


def _visible_key(prompt: str) -> str:
    """Return the case-folded, whitespace-normalised prompt text."""
    return " ".join(prompt.casefold().split())


def _plan_folds(
    events: Sequence[FeedbackEvent],
    *,
    fraction: float,
    min_groups: int,
    seed: int,
) -> _FoldPlan:
    """Hold out whole source-cluster groups and pick one event from each.

    Each event joins the node of its recorded cluster, or a node of its
    own when it has none, and the node of its normalised prompt. Joined
    nodes form one group, so clusters that share a prompt stay together.
    Only groups whose events all carry a recorded cluster may be held out.
    Groups are taken in a seeded order until the held-out events reach
    ``fraction`` of the clustered events and the groups reach
    ``min_groups``. The calibration fold then holds one event per held-out
    group, chosen with the same seeded generator, so its size equals the
    number of independent groups.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        """Return the group root of ``node`` and compress its path."""
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        """Join two groups under the lexicographically smaller root."""
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            low, high = sorted((left_root, right_root))
            parent[high] = low

    source_nodes: list[str] = []
    unclustered = 0
    for index, event in enumerate(events):
        cluster = event.source_cluster
        if cluster is None:
            unclustered += 1
            node = f"event:{index}"
        else:
            node = f"cluster:{cluster}"
        source_nodes.append(node)
        union(node, f"prompt:{_visible_key(event.prompt)}")

    groups: dict[str, list[int]] = {}
    for index, node in enumerate(source_nodes):
        groups.setdefault(find(node), []).append(index)
    eligible = sorted(
        root
        for root, members in groups.items()
        if all(events[i].source_cluster is not None for i in members)
    )
    rng = random.Random(seed)
    rng.shuffle(eligible)
    clustered_events = len(events) - unclustered
    target_events = math.ceil(fraction * clustered_events)

    held_out: list[str] = []
    held_out_events = 0
    for root in eligible:
        if held_out_events >= target_events and len(held_out) >= min_groups:
            break
        held_out.append(root)
        held_out_events += len(groups[root])

    if len(held_out) < min_groups:
        raise ValueError(
            f"calibration needs at least {min_groups} independent source-cluster "
            f"groups for the coverage target; the feedback holds {len(eligible)}. "
            f"Record metadata['source_cluster'] on real feedback events"
        )
    held_out_indices = {i for root in held_out for i in groups[root]}
    if len(held_out_indices) == len(events):
        raise ValueError(
            "no real feedback remains for training after holding out "
            "the calibration clusters"
        )
    representatives = sorted(rng.choice(groups[root]) for root in held_out)
    return _FoldPlan(
        training=tuple(e for i, e in enumerate(events) if i not in held_out_indices),
        calibration=tuple(events[i] for i in representatives),
        calibration_clusters=len(held_out),
        unclustered=unclustered,
        calibration_keys=frozenset(
            _visible_key(events[i].prompt) for i in held_out_indices
        ),
    )
