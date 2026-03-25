# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Phase 5 Gem 10: Cross-Model Consensus."""

from __future__ import annotations

import pytest

from director_ai.core.scoring.consensus import (
    ConsensusScorer,
    ModelResponse,
)


class TestConsensusScorer:
    def test_identical_responses_full_agreement(self):
        responses = [
            ModelResponse(model="a", response="The capital of France is Paris."),
            ModelResponse(model="b", response="The capital of France is Paris."),
        ]
        scorer = ConsensusScorer(models=["a", "b"])
        result = scorer.score_responses(responses)
        assert result.agreement_score == 1.0
        assert result.has_consensus
        assert len(result.disagreement_pairs) == 0

    def test_completely_different_responses(self):
        responses = [
            ModelResponse(model="a", response="AAAA BBBB CCCC DDDD"),
            ModelResponse(model="b", response="xxxx yyyy zzzz wwww"),
        ]
        scorer = ConsensusScorer(models=["a", "b"])
        result = scorer.score_responses(responses)
        assert result.agreement_score < 0.3
        assert not result.has_consensus

    def test_three_models_mixed(self):
        responses = [
            ModelResponse(model="a", response="Paris is the capital of France."),
            ModelResponse(model="b", response="The capital of France is Paris."),
            ModelResponse(model="c", response="Tokyo is the capital of Japan."),
        ]
        scorer = ConsensusScorer(models=["a", "b", "c"])
        result = scorer.score_responses(responses)
        assert result.num_models == 3
        assert len(result.pairs) == 3  # C(3,2) = 3

    def test_single_response(self):
        responses = [ModelResponse(model="a", response="test")]
        scorer = ConsensusScorer(models=["a", "b"])
        result = scorer.score_responses(responses)
        assert result.agreement_score == 1.0
        assert result.num_models == 1


class TestConsensusWithGenerate:
    def test_generate_fn(self):
        def mock_gen(prompt, model):
            return f"{model} says: the answer is 42"

        scorer = ConsensusScorer(
            models=["a", "b"],
            generate_fn=mock_gen,
        )
        result = scorer.score("What is the answer?")
        assert result.num_models == 2
        assert result.agreement_score > 0.5

    def test_no_generate_fn_raises(self):
        scorer = ConsensusScorer(models=["a", "b"])
        with pytest.raises(ValueError, match="generate_fn"):
            scorer.score("test")


class TestConsensusCustomScorer:
    def test_custom_score_fn(self):
        def always_agree(a, b):
            return 0.0

        responses = [
            ModelResponse(model="a", response="anything"),
            ModelResponse(model="b", response="completely different"),
        ]
        scorer = ConsensusScorer(models=["a", "b"], score_fn=always_agree)
        result = scorer.score_responses(responses)
        assert result.agreement_score == 1.0

    def test_custom_score_fn_disagree(self):
        def always_disagree(a, b):
            return 1.0

        responses = [
            ModelResponse(model="a", response="same text"),
            ModelResponse(model="b", response="same text"),
        ]
        scorer = ConsensusScorer(models=["a", "b"], score_fn=always_disagree)
        result = scorer.score_responses(responses)
        assert result.agreement_score == 0.0


class TestValidation:
    def test_needs_two_models(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConsensusScorer(models=["only_one"])

    def test_pairwise_count(self):
        responses = [
            ModelResponse(model="a", response="x"),
            ModelResponse(model="b", response="x"),
            ModelResponse(model="c", response="x"),
            ModelResponse(model="d", response="x"),
        ]
        scorer = ConsensusScorer(models=["a", "b", "c", "d"])
        result = scorer.score_responses(responses)
        assert len(result.pairs) == 6  # C(4,2)
