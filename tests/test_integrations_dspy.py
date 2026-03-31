# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DSPy Integration Tests
from __future__ import annotations

from unittest.mock import patch

import pytest

from director_ai.core import GroundTruthStore
from director_ai.core.exceptions import HallucinationError
from director_ai.core.types import CoherenceScore
from director_ai.integrations.dspy import coherence_check, director_assert


def _fake_review_pass(self, prompt, action, session=None, tenant_id=""):
    cs = CoherenceScore(score=0.95, approved=True, h_logical=0.02, h_factual=0.03)
    return True, cs


def _fake_review_fail(self, prompt, action, session=None, tenant_id=""):
    cs = CoherenceScore(score=0.15, approved=False, h_logical=0.8, h_factual=0.7)
    return False, cs


class TestCoherenceCheck:
    def test_returns_dict_with_required_keys(self):
        result = coherence_check("The sky is blue.", prompt="What colour is the sky?")
        assert isinstance(result, dict)
        assert "approved" in result
        assert "score" in result
        assert "evidence" in result

    @patch("director_ai.core.CoherenceScorer.review", _fake_review_pass)
    def test_approved_when_coherent(self):
        result = coherence_check(
            "Team plan costs $19/user/month.",
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.3,
        )
        assert result["approved"] is True
        assert result["score"] >= 0.3

    @patch("director_ai.core.CoherenceScorer.review", _fake_review_fail)
    def test_rejected_when_incoherent(self):
        result = coherence_check(
            "Random unrelated text.",
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.5,
        )
        assert result["approved"] is False
        assert result["score"] < 0.5

    def test_uses_provided_store(self):
        store = GroundTruthStore()
        store.add("pricing", "Team plan costs $19/user/month.")
        result = coherence_check(
            "Team plan costs $19/user/month.",
            store=store,
        )
        assert isinstance(result, dict)
        assert "approved" in result

    @patch("director_ai.core.CoherenceScorer.review", _fake_review_pass)
    def test_store_overrides_facts(self):
        store = GroundTruthStore()
        store.add("data", "Cats are mammals.")
        result = coherence_check(
            "Cats are mammals.",
            facts={"data": "This should be ignored when store is provided."},
            store=store,
            threshold=0.3,
        )
        assert result["approved"] is True

    def test_empty_response(self):
        result = coherence_check("", prompt="test")
        assert isinstance(result["score"], float)

    def test_no_facts_no_store(self):
        result = coherence_check("Some text.", prompt="query")
        assert isinstance(result, dict)

    def test_use_nli_none(self):
        result = coherence_check("Text.", use_nli=None)
        assert isinstance(result["approved"], bool)

    def test_use_nli_false(self):
        result = coherence_check("Text.", use_nli=False)
        assert isinstance(result["approved"], bool)

    def test_score_is_float(self):
        result = coherence_check("Answer.", prompt="Question?")
        assert isinstance(result["score"], float)


class TestDirectorAssert:
    @patch("director_ai.integrations.dspy.CoherenceScorer.review", _fake_review_pass)
    def test_passes_when_coherent(self):
        director_assert(
            "Team plan costs $19/user/month.",
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.3,
        )

    @patch("director_ai.integrations.dspy.CoherenceScorer.review", _fake_review_fail)
    def test_raises_hallucination_error(self):
        with pytest.raises(HallucinationError):
            director_assert(
                "Fabricated nonsense.",
                facts={"pricing": "Team plan costs $19/user/month."},
                threshold=0.5,
            )

    @patch("director_ai.integrations.dspy.CoherenceScorer.review", _fake_review_fail)
    def test_error_has_attributes(self):
        with pytest.raises(HallucinationError) as exc_info:
            director_assert(
                "Wrong answer.",
                prompt="What is the pricing?",
                facts={"pricing": "Team plan costs $19/user/month."},
                threshold=0.5,
            )
        err = exc_info.value
        assert err.query == "What is the pricing?"
        assert err.response == "Wrong answer."
        assert hasattr(err, "score")
        assert err.score.score == 0.15

    @patch("director_ai.integrations.dspy.CoherenceScorer.review", _fake_review_pass)
    def test_custom_message_ignored_on_pass(self):
        director_assert(
            "Team plan costs $19/user/month.",
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.3,
            message="Should not raise",
        )

    @patch("director_ai.integrations.dspy.CoherenceScorer.review", _fake_review_pass)
    def test_with_store(self):
        store = GroundTruthStore()
        store.add("data", "Paris is in France.")
        director_assert("Paris is in France.", store=store, threshold=0.3)

    @patch("director_ai.integrations.dspy.CoherenceScorer.review", _fake_review_pass)
    def test_with_use_nli_false(self):
        director_assert(
            "Team plan costs $19/user/month.",
            facts={"pricing": "Team plan costs $19/user/month."},
            threshold=0.3,
            use_nli=False,
        )
