# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — L16 Physics → Consumer Agent Bridge
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""
Bridge connecting the research L16 physics modules to the consumer
CoherenceAgent scoring pipeline.

When research dependencies are installed, ``PhysicsBackedScorer``
uses the SEC Lyapunov functional to provide physics-grounded
coherence scores.  When research deps are absent, it falls back
to the standard heuristic scorer.

Usage::

    from director_ai.core.bridge import PhysicsBackedScorer

    scorer = PhysicsBackedScorer(threshold=0.6, ground_truth_store=store)
    approved, score = scorer.review("prompt", "response")
"""

from __future__ import annotations

import logging

import numpy as np

from .scorer import CoherenceScorer
from .types import CoherenceScore

logger = logging.getLogger("DirectorAI.Bridge")

_HAS_RESEARCH = False
try:
    from ..research.physics import L16OversightLoop, SECFunctional

    _HAS_RESEARCH = True
except ImportError:
    pass


class PhysicsBackedScorer(CoherenceScorer):
    """Coherence scorer with optional physics-backed validation.

    When the research extensions are available, runs a short L16
    oversight simulation to produce a physics-grounded coherence score.
    The final score blends the heuristic score with the physics score.

    Parameters
    ----------
    physics_weight : float — weight of physics score in blend (0-1).
    simulation_steps : int — UPDE integration steps per review.
    threshold, ground_truth_store, use_nli : inherited from CoherenceScorer.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ground_truth_store=None,
        use_nli: bool = False,
        physics_weight: float = 0.3,
        simulation_steps: int = 10,
    ) -> None:
        super().__init__(
            threshold=threshold,
            ground_truth_store=ground_truth_store,
            use_nli=use_nli,
        )
        self.physics_weight = physics_weight if _HAS_RESEARCH else 0.0
        self.simulation_steps = simulation_steps

        self._oversight: L16OversightLoop | None = None
        self._sec: SECFunctional | None = None
        self._last_physics_score: float | None = None

        if _HAS_RESEARCH:
            self._sec = SECFunctional()
            self._oversight = L16OversightLoop()
            logger.info("Physics-backed scoring enabled (weight=%.2f)", physics_weight)
        else:
            logger.info("Research deps unavailable — using heuristic scoring only")

    @property
    def has_physics(self) -> bool:
        return _HAS_RESEARCH and self._sec is not None

    def physics_score(self) -> float:
        """Run a short L16 oversight simulation and return coherence score.

        The simulation starts from random phases and runs for
        ``simulation_steps`` integration steps. The final SEC coherence
        score is returned (higher = more coherent).
        """
        if not self.has_physics or self._oversight is None:
            return 0.5

        snapshots = self._oversight.run(n_steps=self.simulation_steps)
        if not snapshots:
            return 0.5

        final = snapshots[-1]
        self._last_physics_score = final.coherence_score
        return final.coherence_score

    def review(self, prompt: str, action: str) -> tuple[bool, CoherenceScore]:
        """Score an action using blended heuristic + physics scoring.

        Returns (approved, CoherenceScore) with the blended score.
        """
        # Get heuristic score
        h_logic = self.calculate_logical_divergence(prompt, action)
        h_fact = self.calculate_factual_divergence(prompt, action)
        heuristic_coherence = 1.0 - (0.6 * h_logic + 0.4 * h_fact)

        # Blend with physics score if available
        if self.has_physics and self.physics_weight > 0:
            p_score = self.physics_score()
            w = self.physics_weight
            blended = (1 - w) * heuristic_coherence + w * p_score
        else:
            blended = heuristic_coherence

        approved = bool(blended >= self.threshold)

        if not approved:
            self.logger.critical(
                "COHERENCE FAILURE. Blended: %.4f < Threshold: %s",
                blended,
                self.threshold,
            )
        else:
            self.history.append(action)
            if len(self.history) > self.window:
                self.history.pop(0)

        score = CoherenceScore(
            score=blended,
            approved=approved,
            h_logical=h_logic,
            h_factual=h_fact,
        )
        return approved, score
