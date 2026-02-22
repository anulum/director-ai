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
import threading

from .scorer import CoherenceScorer
from .types import CoherenceScore

logger = logging.getLogger("DirectorAI.Bridge")

_HAS_RESEARCH = False
_RESEARCH_IMPORT_ERROR: str | None = None
try:
    from ..research.physics import L16OversightLoop, SECFunctional

    _HAS_RESEARCH = True
except ImportError:
    _RESEARCH_IMPORT_ERROR = "research extensions not installed"
except Exception as _exc:  # noqa: BLE001
    _RESEARCH_IMPORT_ERROR = f"research module broken: {_exc}"
    logger.warning("Research physics import failed (non-ImportError): %s", _exc)


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
        if not (0.0 <= physics_weight <= 1.0):
            raise ValueError(f"physics_weight must be in [0, 1], got {physics_weight}")
        if simulation_steps < 1:
            raise ValueError(f"simulation_steps must be >= 1, got {simulation_steps}")
        self.physics_weight = physics_weight if _HAS_RESEARCH else 0.0
        self.simulation_steps = simulation_steps

        self._oversight: L16OversightLoop | None = None
        self._sec: SECFunctional | None = None
        self._last_physics_score: float | None = None
        self._physics_lock = threading.Lock()

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

        with self._physics_lock:
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
        h_logic, h_fact, heuristic_coherence = self._heuristic_coherence(prompt, action)

        # Blend with physics score if available
        if self.has_physics and self.physics_weight > 0:
            p_score = self.physics_score()
            w = self.physics_weight
            blended = (1 - w) * heuristic_coherence + w * p_score
        else:
            blended = heuristic_coherence

        return self._finalise_review(blended, h_logic, h_fact, action)
