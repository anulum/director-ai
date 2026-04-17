# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TrajectorySimulator

"""Monte-Carlo trajectory simulator.

The flow is:

1. :meth:`TrajectorySimulator.preflight` receives a prompt.
2. For each of ``n_simulations`` seeds, the injected
   :class:`Actor` returns a token list — a plausible response
   draw.
3. The injected :class:`VerdictProducer` scores the joined
   response; its verdict contributes to the aggregate.
4. The aggregate — halt rate, mean coherence, 95 % credible
   interval over the simulated scores, and a ``proceed`` /
   ``warn`` / ``halt`` recommendation — is returned as a
   :class:`PreflightVerdict`.

Seeds are deterministic (``base_seed + i``) so reruns produce
identical verdicts — useful for regression testing and for
forensic reconstructions of a preflight decision.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

logger = logging.getLogger("DirectorAI.Trajectory")

Action = Literal["proceed", "warn", "halt"]


class Actor(Protocol):
    """Any callable that produces a sampled trajectory.

    A real implementation typically wraps a distilled lightweight
    policy model, but the protocol is content-agnostic: the
    simulator only requires a deterministic, seedable token
    generator. Tests inject fixed draws.
    """

    def sample(self, prompt: str, seed: int) -> list[str]:
        """Draw a token list for ``prompt``. Must be deterministic
        given ``seed`` so the preflight decision is reproducible."""
        ...  # pragma: no cover


class VerdictProducer(Protocol):
    """Any object exposing ``.review(prompt, action)`` returning
    ``(approved: bool, CoherenceScore-like)``. The existing
    :class:`director_ai.core.scoring.scorer.CoherenceScorer` fits
    the protocol without modification.
    """

    def review(self, prompt: str, action: str, tenant_id: str = "") -> tuple[bool, object]:
        ...  # pragma: no cover


@dataclass(frozen=True)
class TrajectoryResult:
    """One simulated draw and its verdict."""

    trajectory_id: int
    seed: int
    tokens: tuple[str, ...]
    final_coherence: float
    approved: bool

    @property
    def text(self) -> str:
        return "".join(self.tokens)


@dataclass(frozen=True)
class PreflightVerdict:
    """Aggregate of N simulated trajectories."""

    n_simulations: int
    halt_rate: float
    mean_coherence: float
    std_coherence: float
    ci_low: float
    ci_high: float
    recommended: Action
    reason: str
    trajectories: tuple[TrajectoryResult, ...] = field(default_factory=tuple)

    @property
    def min_coherence(self) -> float:
        if not self.trajectories:
            return 0.0
        return min(t.final_coherence for t in self.trajectories)

    @property
    def max_coherence(self) -> float:
        if not self.trajectories:
            return 0.0
        return max(t.final_coherence for t in self.trajectories)


class TrajectorySimulator:
    """Pre-execution Monte-Carlo halt check.

    Parameters
    ----------
    actor :
        A :class:`Actor` that samples token sequences.
    scorer :
        A :class:`VerdictProducer` that returns ``(approved, score)``.
    n_simulations :
        Number of independent draws per :meth:`preflight` call.
        Defaults to 8 — enough for a useful halt-rate estimate
        without turning preflight into the slow path.
    halt_rate_halt :
        Halt-rate threshold above which :meth:`preflight` returns
        ``action="halt"``.
    halt_rate_warn :
        Halt-rate threshold for ``action="warn"``. Between this and
        ``halt_rate_halt`` the caller is expected to escalate
        (e.g. route to a stronger model).
    base_seed :
        Seed offset; per-trajectory seeds are ``base_seed + i``.
    ci_level :
        Credible-interval level for the ``ci_low`` / ``ci_high``
        fields; 0.95 is the default.
    """

    def __init__(
        self,
        *,
        actor: Actor,
        scorer: VerdictProducer,
        n_simulations: int = 8,
        halt_rate_halt: float = 0.5,
        halt_rate_warn: float = 0.25,
        base_seed: int = 17,
        ci_level: float = 0.95,
    ) -> None:
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive; got {n_simulations}")
        if not 0.0 < halt_rate_warn < halt_rate_halt <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < warn < halt <= 1; got "
                f"({halt_rate_warn}, {halt_rate_halt})"
            )
        if not 0.5 <= ci_level < 1.0:
            raise ValueError(f"ci_level must be in [0.5, 1); got {ci_level}")
        self._actor = actor
        self._scorer = scorer
        self._n_simulations = n_simulations
        self._halt_rate_halt = halt_rate_halt
        self._halt_rate_warn = halt_rate_warn
        self._base_seed = base_seed
        self._ci_level = ci_level

    def preflight(
        self,
        prompt: str,
        *,
        tenant_id: str = "",
        on_trajectory: Callable[[TrajectoryResult], None] | None = None,
    ) -> PreflightVerdict:
        """Run the Monte-Carlo loop and return an aggregate verdict.

        ``on_trajectory`` is an optional per-draw callback — handy
        for live observability sinks that want to stream
        intermediate results without waiting for the aggregate.
        """
        trajectories: list[TrajectoryResult] = []
        coherences: list[float] = []
        halts = 0
        for i in range(self._n_simulations):
            seed = self._base_seed + i
            tokens = tuple(self._actor.sample(prompt, seed))
            text = "".join(tokens)
            approved, score = self._scorer.review(
                prompt=prompt, action=text, tenant_id=tenant_id
            )
            coherence = float(getattr(score, "score", 0.0))
            result = TrajectoryResult(
                trajectory_id=i,
                seed=seed,
                tokens=tokens,
                final_coherence=coherence,
                approved=bool(approved),
            )
            trajectories.append(result)
            coherences.append(coherence)
            if not approved:
                halts += 1
            if on_trajectory is not None:
                try:
                    on_trajectory(result)
                except Exception as exc:  # pragma: no cover — defensive
                    logger.debug(
                        "on_trajectory callback failed for #%d: %s", i, exc
                    )

        halt_rate = halts / float(self._n_simulations)
        mean = statistics.fmean(coherences) if coherences else 0.0
        std = (
            statistics.stdev(coherences) if len(coherences) >= 2 else 0.0
        )
        ci_low, ci_high = _credible_interval(coherences, self._ci_level)
        action, reason = self._decide(halt_rate, mean)
        return PreflightVerdict(
            n_simulations=self._n_simulations,
            halt_rate=halt_rate,
            mean_coherence=mean,
            std_coherence=std,
            ci_low=ci_low,
            ci_high=ci_high,
            recommended=action,
            reason=reason,
            trajectories=tuple(trajectories),
        )

    def _decide(self, halt_rate: float, mean: float) -> tuple[Action, str]:
        if halt_rate >= self._halt_rate_halt:
            return (
                "halt",
                f"halt_rate={halt_rate:.3f} >= halt_threshold "
                f"({self._halt_rate_halt:.3f})",
            )
        if halt_rate >= self._halt_rate_warn:
            return (
                "warn",
                f"halt_rate={halt_rate:.3f} >= warn_threshold "
                f"({self._halt_rate_warn:.3f})",
            )
        return (
            "proceed",
            f"halt_rate={halt_rate:.3f}, mean_coh={mean:.3f}",
        )


def _credible_interval(samples: list[float], level: float) -> tuple[float, float]:
    """Empirical ``(low, high)`` quantiles for the given level.

    For small-N preflight runs the true distribution is unknown,
    so we return the clamped empirical quantiles rather than
    assuming normality. Operators who need tighter guarantees
    should calibrate a conformal band from historical traces and
    feed it in separately.
    """
    if not samples:
        return (0.0, 0.0)
    lo = (1.0 - level) / 2.0
    hi = 1.0 - lo
    ordered = sorted(samples)
    n = len(ordered)
    return (
        ordered[max(0, min(n - 1, math.floor(lo * n)))],
        ordered[max(0, min(n - 1, math.ceil(hi * n) - 1))],
    )
