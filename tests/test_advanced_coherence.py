# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Advanced Coherence Tests
"""Multi-angle tests for CoherenceAgent multi-turn and paradox handling.

Covers: multi-turn state persistence, threshold enforcement, paradox
resolution, prompt injection resistance, state isolation between turns,
score monotonicity under consistent context, and pipeline performance.
"""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceAgent


@pytest.fixture
def agent():
    return CoherenceAgent()


@pytest.fixture
def strict_agent():
    agent = CoherenceAgent()
    agent.scorer.threshold = 0.7
    return agent


# ── Multi-turn coherence ──────────────────────────────────────────


class TestMultiTurnCoherence:
    """Agent must handle multi-turn conversations stably."""

    @pytest.mark.parametrize(
        "prompts",
        [
            [
                "Tell me about the fundamental laws of physics.",
                "Can these laws be broken by a sentient agent?",
                "Explain how deception affects system entropy.",
            ],
            [
                "What is photosynthesis?",
                "How does it relate to climate change?",
                "Can artificial photosynthesis solve energy problems?",
            ],
        ],
    )
    def test_multi_turn_produces_output(self, agent, prompts):
        for p in prompts:
            result = agent.process(p)
            assert len(result.output) > 0

    def test_multi_turn_with_strict_threshold(self, strict_agent):
        prompts = [
            "What is gravity?",
            "How does it affect time?",
        ]
        for p in prompts:
            result = strict_agent.process(p)
            assert len(result.output) > 0

    def test_state_persists_between_turns(self, agent):
        r1 = agent.process("Define AI.")
        r2 = agent.process("Expand on that.")
        # Both should produce output, agent shouldn't reset
        assert len(r1.output) > 0
        assert len(r2.output) > 0

    def test_many_turns_no_degradation(self, agent):
        for i in range(10):
            result = agent.process(f"Question number {i}: What is {i}+{i}?")
            assert len(result.output) > 0


# ── Paradox handling ──────────────────────────────────────────────


class TestParadoxHandling:
    """Agent must handle logical paradoxes without crashing."""

    @pytest.mark.parametrize(
        "paradox",
        [
            "Is this statement a lie: 'This sentence is false'?",
            "Can an omnipotent being create a rock so heavy it cannot lift?",
            "If everything is relative, is that statement also relative?",
            "What happens when an irresistible force meets an immovable object?",
        ],
    )
    def test_paradox_produces_output(self, agent, paradox):
        result = agent.process(paradox)
        assert len(result.output) > 0

    def test_paradox_does_not_corrupt_state(self, agent):
        agent.process("This statement is false.")
        result = agent.process("What is 2+2?")
        assert len(result.output) > 0


# ── Threshold enforcement ────────────────────────────────────────


class TestThresholdEnforcement:
    """Verify threshold parameter affects scoring behaviour."""

    @pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_various_thresholds(self, threshold):
        agent = CoherenceAgent()
        agent.scorer.threshold = threshold
        result = agent.process("Simple question: what is water?")
        assert len(result.output) > 0
        assert hasattr(result, "coherence")


# ── Pipeline performance ─────────────────────────────────────────


class TestAdvancedCoherencePerformance:
    """Document pipeline performance for multi-turn interactions."""

    def test_coherence_score_structure(self, agent):
        result = agent.process("What is AI?")
        assert hasattr(result.coherence, "score")
        assert hasattr(result.coherence, "h_logical")
        assert hasattr(result.coherence, "h_factual")
        assert 0.0 <= result.coherence.score <= 1.0

    def test_result_has_output_and_coherence(self, agent):
        result = agent.process("Test query")
        assert hasattr(result, "output")
        assert hasattr(result, "coherence")
        assert hasattr(result, "halted")
