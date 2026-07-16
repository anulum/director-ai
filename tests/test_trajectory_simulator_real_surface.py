# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — trajectory simulator real-surface tests
"""Real public-surface coverage for trajectory preflight decisions."""

from __future__ import annotations

import pytest

from director_ai.core import GroundTruthStore
from director_ai.core.actor import MockGenerator
from director_ai.core.config import DirectorConfig
from director_ai.core.scoring.scorer import CoherenceScorer
from director_ai.core.trajectory import TrajectorySimulator
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class MockGeneratorActor:
    """Adapt the shipped mock generator to the trajectory actor protocol."""

    def __init__(self) -> None:
        self._generator = MockGenerator()

    def sample(self, prompt: str, seed: int) -> list[str]:
        """Return the generated candidate as a deterministic token list."""
        candidate = self._generator.generate_candidates(f"{prompt}\nseed={seed}", n=1)[
            0
        ]
        text = str(candidate.get("text", ""))
        return [f"{token} " for token in text.split()]


class PublicScorerVerdictProducer:
    """Expose a public scorer through the simulator verdict protocol."""

    def __init__(self, scorer: CoherenceScorer) -> None:
        self._scorer = scorer

    def review(
        self, prompt: str, action: str, tenant_id: str = ""
    ) -> tuple[bool, object]:
        """Return the public scorer decision for a sampled trajectory."""
        return self._scorer.review(prompt, action, tenant_id=tenant_id)


def _build_public_simulator(*, threshold: float) -> TrajectorySimulator:
    """Build a documented trajectory preflight stack with real scorer wiring."""
    store = GroundTruthStore()
    store.add("capital", "Paris is the capital of France.", tenant_id="tenant-a")
    scorer = DirectorConfig(
        mode="general",
        use_nli=False,
        scorer_backend="lite",
        coherence_threshold=threshold,
        hard_limit=threshold,
        soft_limit=threshold,
        adaptive_threshold_enabled=False,
        cache_size=0,
    ).build_scorer(store=store)
    return TrajectorySimulator(
        actor=MockGeneratorActor(),
        scorer=PublicScorerVerdictProducer(scorer),
        n_simulations=3,
        halt_rate_warn=0.25,
        halt_rate_halt=0.5,
        base_seed=101,
    )


def test_trajectory_simulator_unit_guard_has_real_surface_companion() -> None:
    """The helper-heavy trajectory guard needs public preflight coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_trajectory_simulator.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_trajectory_simulator_real_surface.py" in category


def test_public_preflight_allows_low_risk_mock_generator_draws() -> None:
    """The documented simulator stack should emit an allow safety event."""
    simulator = _build_public_simulator(threshold=0.3)

    verdict = simulator.preflight(
        "What is the capital of France?",
        tenant_id="tenant-a",
    )

    assert verdict.recommended == "proceed"
    assert verdict.n_simulations == 3
    assert verdict.halt_rate == pytest.approx(0.0)
    # KIMI2-K: the lite backend is a model-backed entailment scorer, so its
    # grounded logical signal now scores against the retrieved context rather
    # than the raw question — a small, deterministic shift in mean coherence.
    assert verdict.mean_coherence == pytest.approx(0.34727272727272734)
    assert verdict.ci_low <= verdict.mean_coherence <= verdict.ci_high
    assert [trajectory.seed for trajectory in verdict.trajectories] == [
        101,
        102,
        103,
    ]
    assert verdict.safety_event is not None
    assert verdict.safety_event.hook_id == "trajectory.preflight"
    assert verdict.safety_event.hook_scope == "trajectory"
    assert verdict.safety_event.policy_decision == "allow"
    assert verdict.safety_event.evidence_refs == ()
    assert verdict.safety_event.attributes["halt_rate"] == "0.000000"
    assert verdict.safety_event.attributes["n_simulations"] == "3"


def test_public_preflight_fails_closed_when_all_draws_cross_threshold() -> None:
    """The same public stack should halt when configured above its score band."""
    simulator = _build_public_simulator(threshold=0.4)

    verdict = simulator.preflight(
        "What is the capital of France?",
        tenant_id="tenant-a",
    )

    assert verdict.recommended == "halt"
    assert verdict.halt_rate == pytest.approx(1.0)
    # KIMI2-K: lite backend now scores the grounded logical signal against the
    # retrieved context (see the sibling allow-path test for the rationale).
    assert verdict.mean_coherence == pytest.approx(0.34727272727272734)
    assert [trajectory.approved for trajectory in verdict.trajectories] == [
        False,
        False,
        False,
    ]
    assert verdict.safety_event is not None
    assert verdict.safety_event.policy_decision == "halt"
    assert verdict.safety_event.halt_reason == "trajectory_halt"
    assert verdict.safety_event.threshold == pytest.approx(0.5)
    assert verdict.safety_event.observed_score == pytest.approx(0.0)
    assert verdict.safety_event.evidence_refs == (
        "trajectory:0",
        "trajectory:1",
        "trajectory:2",
    )
    assert "What is the capital" not in verdict.safety_event.tenant_safe_explanation
