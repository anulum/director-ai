# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — auto-redteam defence loop

"""Reviewed auto-redteam loop for defence-genome promotions.

This module joins the continual adversarial miner with the reviewed
defence update gate. One cycle:

1. Builds a bounded failure store from fresh production failures.
2. Mines and promotes an adversarial suite version.
3. Measures the active and candidate defences against the mined cases.
4. Promotes the candidate only if the candidate improves detection and
   the reviewed ``DefenseUpdatePipeline`` gates pass.

Reports are tenant-safe: raw prompts, raw feedback text, and defence
objects never appear in the serialised payload.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from director_ai.core.continual_adversarial import (
    AdversarialCase,
    ContinualEngine,
    FailureEvent,
    FailurePattern,
    FailureStore,
)

from .registry import Defense, DefenseRegistry
from .update_pipeline import DefenseUpdatePipeline

if TYPE_CHECKING:  # full-tier type — annotations only (ladder P2.4)
    from director_ai.core.self_evolving import GuardLoopProposal


@dataclass(frozen=True)
class AutoRedteamCycleInput:
    """Inputs for one reviewed auto-redteam promotion cycle."""

    failures: Sequence[FailureEvent]
    safe_corpus: Sequence[str]
    proposal: GuardLoopProposal
    candidate_defence: Defense
    version: int
    label: str
    baseline_score: float
    candidate_score: float

    def __post_init__(self) -> None:
        """Reject empty corpora, a bad version/label, or out-of-range scores."""
        failures = tuple(self.failures)
        safe_corpus = tuple(self.safe_corpus)
        if not failures:
            raise ValueError("failures must be non-empty")
        if not safe_corpus:
            raise ValueError("safe_corpus must be non-empty")
        if self.version <= 0:
            raise ValueError("version must be positive")
        if not self.label.strip():
            raise ValueError("label is required")
        _validate_unit_interval("baseline_score", self.baseline_score)
        _validate_unit_interval("candidate_score", self.candidate_score)
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "safe_corpus", safe_corpus)


@dataclass(frozen=True)
class AutoRedteamCycleReport:
    """Tenant-safe result for one auto-redteam cycle."""

    suite_version: int
    promoted_version: int
    label: str
    proposal_id: str
    adversarial_case_count: int
    mined_pattern_count: int
    baseline_detection_rate: float
    candidate_detection_rate: float
    detection_uplift: float
    holdout_delta: float
    pattern_digest: str
    promoted: bool
    metadata: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise without raw prompts, raw feedback, or defence objects."""
        return {
            "suite_version": self.suite_version,
            "promoted_version": self.promoted_version,
            "label": self.label,
            "proposal_id": self.proposal_id,
            "adversarial_case_count": self.adversarial_case_count,
            "mined_pattern_count": self.mined_pattern_count,
            "baseline_detection_rate": self.baseline_detection_rate,
            "candidate_detection_rate": self.candidate_detection_rate,
            "detection_uplift": self.detection_uplift,
            "holdout_delta": self.holdout_delta,
            "pattern_digest": self.pattern_digest,
            "promoted": self.promoted,
            "metadata": dict(self.metadata),
        }


class AutoRedteamDefenceLoop:
    """Run repeated adversarial-mining cycles before defence promotion."""

    def __init__(
        self,
        *,
        registry: DefenseRegistry,
        pipeline: DefenseUpdatePipeline | None = None,
        min_failures: int = 16,
        window_last_n: int = 512,
        block_threshold: float = 0.5,
        min_detection_uplift: float = 0.01,
        min_adversarial_cases: int = 1,
        min_holdout_improvement: float = 0.0,
    ) -> None:
        if min_failures <= 0:
            raise ValueError("min_failures must be positive")
        if window_last_n <= 0:
            raise ValueError("window_last_n must be positive")
        _validate_unit_interval("block_threshold", block_threshold)
        if not math.isfinite(min_detection_uplift) or min_detection_uplift < 0.0:
            raise ValueError("min_detection_uplift must be finite and non-negative")
        self._registry = registry
        self._pipeline = pipeline or DefenseUpdatePipeline(
            registry=registry,
            min_adversarial_cases=min_adversarial_cases,
            min_holdout_improvement=min_holdout_improvement,
        )
        self._min_failures = min_failures
        self._window_last_n = window_last_n
        self._block_threshold = block_threshold
        self._min_detection_uplift = min_detection_uplift

    def run(
        self,
        cycles: Sequence[AutoRedteamCycleInput],
    ) -> tuple[AutoRedteamCycleReport, ...]:
        """Run cycles sequentially so each promotion becomes the next baseline."""
        if not cycles:
            raise ValueError("cycles must be non-empty")
        return tuple(self.run_cycle(cycle) for cycle in cycles)

    def run_cycle(self, cycle: AutoRedteamCycleInput) -> AutoRedteamCycleReport:
        """Run one reviewed mining + detection-uplift + promotion cycle."""
        active = self._registry.active()
        if active is None:
            raise ValueError("registry must contain an active baseline defence")

        store = FailureStore(capacity=max(len(cycle.failures), self._min_failures))
        for event in cycle.failures:
            store.append(event)

        evolve_report = ContinualEngine(
            store=store,
            min_failures=self._min_failures,
            window_last_n=self._window_last_n,
        ).evolve(safe_corpus=cycle.safe_corpus)

        cases = evolve_report.version.cases
        baseline_detection_rate = _detection_rate(
            active.defense,
            cases,
            threshold=self._block_threshold,
        )
        candidate_detection_rate = _detection_rate(
            cycle.candidate_defence,
            cases,
            threshold=self._block_threshold,
        )
        detection_uplift = candidate_detection_rate - baseline_detection_rate
        if detection_uplift < self._min_detection_uplift:
            raise ValueError(
                "candidate detection uplift does not clear the redteam gate: "
                f"{detection_uplift:.6f} < {self._min_detection_uplift:.6f}"
            )

        update_report = self._pipeline.review_and_promote(
            proposal=cycle.proposal,
            evolve_report=evolve_report,
            defense=cycle.candidate_defence,
            version=cycle.version,
            label=cycle.label,
            baseline_score=cycle.baseline_score,
            candidate_score=cycle.candidate_score,
        )
        metadata = {
            **dict(update_report.metadata),
            "redteam_detection_uplift": f"{detection_uplift:.6f}",
            "redteam_pattern_digest": _pattern_digest(evolve_report.version.patterns),
        }
        return AutoRedteamCycleReport(
            suite_version=evolve_report.version.version,
            promoted_version=update_report.snapshot.version,
            label=cycle.label,
            proposal_id=update_report.proposal_id,
            adversarial_case_count=evolve_report.adversarial_case_count,
            mined_pattern_count=evolve_report.mined_pattern_count,
            baseline_detection_rate=round(baseline_detection_rate, 6),
            candidate_detection_rate=round(candidate_detection_rate, 6),
            detection_uplift=round(detection_uplift, 6),
            holdout_delta=round(cycle.candidate_score - cycle.baseline_score, 6),
            pattern_digest=metadata["redteam_pattern_digest"],
            promoted=update_report.promoted,
            metadata=metadata,
        )


def _detection_rate(
    defence: Defense,
    cases: Sequence[AdversarialCase],
    *,
    threshold: float,
) -> float:
    if not cases:
        raise ValueError("cases must be non-empty")
    detected = 0
    for case in cases:
        if _score_defence(defence, case.prompt) <= threshold:
            detected += 1
    return detected / len(cases)


def _score_defence(defence: Defense, prompt: str) -> float:
    score = float(defence.score(prompt))
    _validate_unit_interval("defence score", score)
    return score


def _pattern_digest(patterns: Sequence[FailurePattern]) -> str:
    digest = hashlib.sha256()
    for pattern in patterns:
        digest.update(pattern.kind.encode("utf-8"))
        digest.update(b"\0")
        digest.update(pattern.label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(pattern.support).encode("ascii"))
        digest.update(b"\0")
        digest.update(pattern.signature.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_unit_interval(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
