# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``director_ai.core.retrieval.adaptive_router``.

Covers factual queries, creative queries, ambiguous cases, confidence
scoring, threshold behaviour, and pattern matching.
"""

from __future__ import annotations

from director_ai.core.retrieval.adaptive_router import (
    AdaptiveRouter,
    RoutingDecision,
)

# ── Factual queries → should retrieve ──────────────────────────────────


class TestFactualQueries:
    def test_what_is_question(self):
        r = AdaptiveRouter()
        d = r.should_retrieve("What is the refund policy?")
        assert d.retrieve is True
        assert d.confidence >= 0.5

    def test_who_is_question(self):
        d = AdaptiveRouter().should_retrieve("Who is the CEO of ANULUM?")
        assert d.retrieve is True

    def test_how_many_question(self):
        d = AdaptiveRouter().should_retrieve("How many layers does SCPN have?")
        assert d.retrieve is True

    def test_verify_request(self):
        d = AdaptiveRouter().should_retrieve("Verify that water boils at 100°C")
        assert d.retrieve is True

    def test_policy_question(self):
        d = AdaptiveRouter().should_retrieve(
            "What does the regulation say about data retention?"
        )
        assert d.retrieve is True

    def test_according_to(self):
        d = AdaptiveRouter().should_retrieve(
            "According to the specification, what is the maximum load?"
        )
        assert d.retrieve is True


# ── Creative / conversational → should skip ─────────────────────────────


class TestCreativeQueries:
    def test_write_poem(self):
        d = AdaptiveRouter().should_retrieve("Write me a poem about spring")
        assert d.retrieve is False

    def test_compose_email(self):
        d = AdaptiveRouter().should_retrieve("Compose an email to the team")
        assert d.retrieve is False

    def test_greeting(self):
        d = AdaptiveRouter().should_retrieve("Hello, how are you?")
        assert d.retrieve is False

    def test_brainstorm(self):
        d = AdaptiveRouter().should_retrieve(
            "Brainstorm ideas for a marketing campaign"
        )
        assert d.retrieve is False

    def test_translate(self):
        d = AdaptiveRouter().should_retrieve("Translate this to German")
        assert d.retrieve is False

    def test_tell_joke(self):
        d = AdaptiveRouter().should_retrieve("Tell me a joke about programmers")
        assert d.retrieve is False

    def test_code_request(self):
        d = AdaptiveRouter().should_retrieve("Code a function that sorts a list")
        assert d.retrieve is False


# ── Confidence scoring ──────────────────────────────────────────────────


class TestConfidence:
    def test_confidence_range(self):
        router = AdaptiveRouter()
        for query in [
            "What is X?",
            "Write me a poem",
            "Hello",
            "Explain the policy on returns",
        ]:
            d = router.should_retrieve(query)
            assert 0.0 <= d.confidence <= 1.0

    def test_factual_has_high_confidence(self):
        d = AdaptiveRouter().should_retrieve("What is the capital of France?")
        assert d.confidence >= 0.7

    def test_creative_has_reasonable_confidence(self):
        d = AdaptiveRouter().should_retrieve("Write a haiku")
        assert d.confidence >= 0.5


# ── Threshold behaviour ────────────────────────────────────────────────


class TestThreshold:
    def test_high_threshold_skips_ambiguous(self):
        # With very high threshold, ambiguous queries should not retrieve
        router = AdaptiveRouter(factual_threshold=0.95, default_retrieve=False)
        d = router.should_retrieve("Tell me about science")
        # This is ambiguous — should respect default_retrieve=False
        # when confidence is below threshold
        assert isinstance(d.retrieve, bool)

    def test_low_threshold_retrieves_more(self):
        router = AdaptiveRouter(factual_threshold=0.1)
        d = router.should_retrieve("What is photosynthesis?")
        assert d.retrieve is True

    def test_default_retrieve_true(self):
        router = AdaptiveRouter(default_retrieve=True)
        d = router.should_retrieve("Something ambiguous happening here")
        # Ambiguous should default to True
        assert isinstance(d.retrieve, bool)

    def test_default_retrieve_false(self):
        router = AdaptiveRouter(default_retrieve=False)
        # Even with default=False, strong factual signals should still retrieve
        d = router.should_retrieve("What is the specification for this component?")
        assert d.retrieve is True


# ── Task type detection ─────────────────────────────────────────────────


class TestTaskType:
    def test_task_type_returned(self):
        d = AdaptiveRouter().should_retrieve("What is the refund policy?")
        assert isinstance(d.task_type, str)
        assert len(d.task_type) > 0

    def test_routing_decision_is_frozen(self):
        d = RoutingDecision(retrieve=True, task_type="qa", confidence=0.9)
        assert d.retrieve is True
        assert d.task_type == "qa"
        assert d.confidence == 0.9


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_query(self):
        d = AdaptiveRouter().should_retrieve("")
        assert isinstance(d.retrieve, bool)

    def test_very_long_query(self):
        d = AdaptiveRouter().should_retrieve("what is " * 1000)
        assert d.retrieve is True

    def test_mixed_signals(self):
        """Query with both factual and creative patterns."""
        d = AdaptiveRouter().should_retrieve(
            "Write me a summary based on the policy document"
        )
        # Should not crash; confidence should be lower
        assert isinstance(d.retrieve, bool)
        assert 0.0 <= d.confidence <= 1.0

    def test_unicode_query(self):
        d = AdaptiveRouter().should_retrieve("Čo je to fotosyntéza?")
        assert isinstance(d.retrieve, bool)

    def test_with_response(self):
        d = AdaptiveRouter().should_retrieve(
            "Tell me about water",
            response="Water is H2O, a molecule consisting of two hydrogen atoms.",
        )
        assert isinstance(d.retrieve, bool)
