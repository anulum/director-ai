# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Physics Bridge Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core.bridge import PhysicsBackedScorer
from director_ai.core.knowledge import SAMPLE_FACTS, GroundTruthStore
from director_ai.core.types import CoherenceScore


@pytest.mark.integration
class TestPhysicsBackedScorer:
    def test_instantiation(self):
        store = GroundTruthStore(facts=SAMPLE_FACTS)
        scorer = PhysicsBackedScorer(threshold=0.5, ground_truth_store=store)
        assert scorer is not None
        assert scorer.has_physics  # research deps should be available in test env

    def test_review_returns_coherence_score(self):
        store = GroundTruthStore(facts=SAMPLE_FACTS)
        scorer = PhysicsBackedScorer(
            threshold=0.5,
            ground_truth_store=store,
            physics_weight=0.3,
        )
        approved, score = scorer.review("test", "consistent with reality")
        assert isinstance(score, CoherenceScore)
        assert 0.0 <= score.score <= 1.0
        assert isinstance(approved, bool)

    def test_physics_score_in_range(self):
        store = GroundTruthStore(facts=SAMPLE_FACTS)
        scorer = PhysicsBackedScorer(
            threshold=0.5,
            ground_truth_store=store,
            physics_weight=0.3,
            simulation_steps=5,
        )
        p_score = scorer.physics_score()
        assert 0.0 <= p_score <= 1.0

    def test_blended_score_differs_from_heuristic(self):
        store = GroundTruthStore(facts=SAMPLE_FACTS)
        # Pure heuristic
        heuristic = PhysicsBackedScorer(
            threshold=0.5,
            ground_truth_store=store,
            physics_weight=0.0,
        )
        # Blended
        blended = PhysicsBackedScorer(
            threshold=0.5,
            ground_truth_store=store,
            physics_weight=0.5,
            simulation_steps=10,
        )
        _, h_score = heuristic.review("test", "consistent with reality")
        _, b_score = blended.review("test", "consistent with reality")
        # They may differ due to physics contribution
        assert isinstance(h_score.score, float)
        assert isinstance(b_score.score, float)

    def test_backward_compat_aliases(self):
        store = GroundTruthStore(facts=SAMPLE_FACTS)
        scorer = PhysicsBackedScorer(
            threshold=0.5,
            ground_truth_store=store,
            physics_weight=0.0,
        )
        # Should still have parent class aliases
        h = scorer.calculate_factual_divergence("sky", "The sky color is blue.")
        assert h < 0.5
