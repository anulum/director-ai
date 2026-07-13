# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multi-angle tests for the self-consistency / semantic-entropy signal.

Covers: constructor contracts, input validation, lexical and NLI
clustering, entropy normalisation edges (full agreement, full
disagreement, mixed), primary-agreement semantics, to_dict projection,
and the review_with_samples fusion (blend arithmetic, approval
revocation, no silent fallback, field attachment).
"""

from __future__ import annotations

import pytest

from director_ai.core import CoherenceScorer, SelfConsistencyScorer
from director_ai.core.scoring.self_consistency import ConsistencyResult


class _FakeNLI:
    """Deterministic divergence oracle keyed by unordered text pairs."""

    def __init__(self, divergences: dict[frozenset[str], float]) -> None:
        self._d = divergences

    def score(self, premise: str, hypothesis: str) -> float:
        if premise == hypothesis:
            return 0.0
        return self._d.get(frozenset((premise, hypothesis)), 1.0)


class TestConstructorContracts:
    @pytest.mark.parametrize("field", ["entail_divergence", "lexical_overlap"])
    @pytest.mark.parametrize("value", [0.0, 1.0, -0.2, 1.5])
    def test_rejects_out_of_range_thresholds(self, field, value):
        with pytest.raises(ValueError, match=field):
            SelfConsistencyScorer(**{field: value})

    def test_rejects_nli_without_score_method(self):
        with pytest.raises(ValueError, match="callable score method"):
            SelfConsistencyScorer(nli_scorer=object())

    def test_backend_labels(self):
        assert SelfConsistencyScorer().entailment_backend == "lexical"
        assert (
            SelfConsistencyScorer(nli_scorer=_FakeNLI({})).entailment_backend == "nli"
        )


class TestInputValidation:
    def setup_method(self):
        self.scorer = SelfConsistencyScorer()

    @pytest.mark.parametrize("primary", ["", "   ", None, 42])
    def test_rejects_bad_primary(self, primary):
        with pytest.raises(ValueError, match="primary"):
            self.scorer.score(primary, ["a sample"])

    @pytest.mark.parametrize("samples", [[], None, "not a list"])
    def test_rejects_empty_or_non_list_samples(self, samples):
        with pytest.raises(ValueError, match="samples"):
            self.scorer.score("primary answer", samples)

    def test_rejects_blank_sample_member(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            self.scorer.score("primary answer", ["fine", "  "])


class TestLexicalClustering:
    def setup_method(self):
        self.scorer = SelfConsistencyScorer()

    def test_full_agreement_zero_entropy(self):
        result = self.scorer.score(
            "The warranty period is 24 months.",
            [
                "The warranty period is 24 months.",
                "Warranty period: 24 months.",
            ],
        )

        assert result.n_clusters == 1
        assert result.semantic_entropy == 0.0
        assert result.primary_agreement == 1.0
        assert result.consistency_score == 1.0

    def test_full_disagreement_max_entropy(self):
        result = self.scorer.score(
            "The capital of Australia is Canberra.",
            [
                "Paris hosted the 2024 summer olympic games.",
                "Quantum tunnelling enables flash memory cells.",
            ],
        )

        assert result.n_clusters == 3
        assert result.semantic_entropy == pytest.approx(1.0)
        assert result.primary_agreement == pytest.approx(1 / 3)
        assert result.consistency_score == pytest.approx(1 / 6)

    def test_majority_cluster_with_outlier(self):
        result = self.scorer.score(
            "The hub listens on port 8876.",
            [
                "The hub listens on port 8876.",
                "The hub listens on port 8876 today.",
                "Bananas are botanically berries grown in tropical zones.",
            ],
        )

        assert result.n_clusters == 2
        assert result.primary_agreement == 0.75
        assert 0.0 < result.semantic_entropy < 1.0

    def test_jaccard_identical_empty_token_sets(self):
        assert SelfConsistencyScorer._jaccard("!!!", "???") == 1.0


class TestNliClustering:
    def test_bidirectional_entailment_required(self):
        a = "The device ships with a two-year warranty."
        b = "A 24-month warranty covers the device."
        c = "The device has no warranty at all."
        nli = _FakeNLI(
            {
                frozenset((a, b)): 0.1,  # mutual entailment
                frozenset((a, c)): 0.9,
                frozenset((b, c)): 0.9,
            },
        )
        scorer = SelfConsistencyScorer(nli_scorer=nli)

        result = scorer.score(a, [b, c])

        assert result.n_clusters == 2
        assert result.clusters[0] == [0, 1]
        assert result.entailment_backend == "nli"

    def test_asymmetric_divergence_splits_cluster(self):
        a = "All seats require an identity key."
        b = "Some seats require an identity key."

        class _Asymmetric:
            def score(self, premise: str, hypothesis: str) -> float:
                # a -> b entails (specialisation), b -> a does not.
                return 0.1 if premise == a else 0.8

        scorer = SelfConsistencyScorer(nli_scorer=_Asymmetric())

        result = scorer.score(a, [b])

        assert result.n_clusters == 2


class TestResultProjection:
    def test_to_dict_shapes_and_rounding(self):
        result = ConsistencyResult(
            semantic_entropy=0.123456,
            primary_agreement=0.666666,
            consistency_score=0.771605,
            n_samples=3,
            n_clusters=2,
            entailment_backend="lexical",
            clusters=[[0, 1, 2], [3]],
        )

        payload = result.to_dict()

        assert payload["semantic_entropy"] == 0.1235
        assert payload["primary_agreement"] == 0.6667
        assert payload["consistency_score"] == 0.7716
        assert payload["clusters"] == [[0, 1, 2], [3]]


class TestReviewWithSamplesFusion:
    def _scorer(self, threshold: float = 0.55) -> CoherenceScorer:
        scorer = CoherenceScorer(use_nli=False, threshold=threshold)
        scorer.enable_self_consistency(weight=0.5)
        return scorer

    def test_requires_enable_first(self):
        scorer = CoherenceScorer(use_nli=False)

        with pytest.raises(RuntimeError, match="enable_self_consistency"):
            scorer.review_with_samples("q", "a plausible answer", ["a sample"])

    def test_enable_rejects_bad_weight(self):
        scorer = CoherenceScorer(use_nli=False)

        with pytest.raises(ValueError, match="weight"):
            scorer.enable_self_consistency(weight=1.0)

    def test_consistent_samples_attach_fields_and_keep_score(self):
        scorer = self._scorer()
        answer = "The warranty period is 24 months."

        approved, score = scorer.review_with_samples(
            "What is the warranty period?",
            answer,
            [answer, "Warranty period: 24 months."],
        )

        assert score.self_consistency_score == 1.0
        assert score.semantic_entropy == 0.0
        assert score.self_consistency_backend == "lexical"

    def test_disagreeing_samples_revoke_approval(self):
        scorer = self._scorer(threshold=0.5)
        prompt = "What is the warranty period?"
        answer = "The warranty period is 24 months."
        divergent = [
            "The warranty lasts six months only.",
            "There is no warranty coverage offered.",
        ]

        base_approved, base_score = scorer.review(prompt, answer)
        approved, fused_score = scorer.review_with_samples(
            prompt,
            answer,
            divergent,
        )

        assert base_approved is True
        # Three-way disagreement: entropy 1.0, agreement 1/3 -> signal 1/6.
        assert fused_score.self_consistency_score == pytest.approx(1 / 6, abs=1e-4)
        expected = 0.5 * base_score.score + 0.5 * fused_score.self_consistency_score
        assert fused_score.score == pytest.approx(expected, abs=1e-4)
        assert fused_score.score < scorer.threshold
        assert approved is False
        assert fused_score.approved is False

    def test_fusion_never_approves_a_rejected_review(self):
        scorer = CoherenceScorer(use_nli=False, threshold=0.99)
        scorer.enable_self_consistency(weight=0.5)
        answer = "The warranty period is 24 months."

        approved, score = scorer.review_with_samples(
            "What is the warranty period?",
            answer,
            [answer, answer],
        )

        # Plain review rejects at threshold 0.99; perfect consistency
        # (signal 1.0) must not flip the decision.
        assert approved is False

    def test_enable_reuses_model_backed_nli_when_available(self):
        scorer = CoherenceScorer(use_nli=False)

        consistency = scorer.enable_self_consistency()

        # use_nli=False -> lexical fallback, never a phantom NLI handle.
        assert consistency.entailment_backend == "lexical"
