# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Conformal intervals on the review path

"""Conformal hallucination-risk intervals emitted by ``review()`` (WCA-4).

``CoherenceScorer.enable_conformal()`` opts the review path into
distribution-free risk intervals from
:class:`~director_ai.core.calibration.conformal.ConformalPredictor`.
These tests lock the opt-in contract: the default path carries no
conformal fields, an uncalibrated predictor is marked unreliable with
vacuous bounds, a calibrated predictor yields a coverage-stamped
interval around the point risk, and the cache-hit and batch paths carry
the same fields as the direct path.
"""

from __future__ import annotations

from director_ai.core.calibration.conformal import ConformalPredictor
from director_ai.core.scoring.scorer import CoherenceScorer

_PROMPT = "What colour is the clear daytime sky?"
_ACTION = "The clear daytime sky is blue."


def _calibrated_scorer(threshold: float = 0.3) -> CoherenceScorer:
    scorer = CoherenceScorer(threshold=threshold, use_nli=False)
    predictor = scorer.enable_conformal(coverage=0.9, min_samples=5)
    predictor.calibrate(
        scores=[0.95, 0.9, 0.85, 0.2, 0.15, 0.1],
        labels=[False, False, False, True, True, True],
    )
    return scorer


class TestOptInContract:
    def test_default_review_carries_no_conformal_fields(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        _approved, score = scorer.review(_PROMPT, _ACTION)
        assert score.conformal_risk_lower is None
        assert score.conformal_risk_upper is None
        assert score.conformal_coverage is None
        assert score.conformal_reliable is None

    def test_enable_conformal_returns_the_attached_predictor(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        predictor = scorer.enable_conformal(coverage=0.9, min_samples=5)
        assert isinstance(predictor, ConformalPredictor)
        assert scorer._conformal_predictor is predictor

    def test_supplied_predictor_is_used_verbatim(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        own = ConformalPredictor(coverage=0.8, min_samples=2)
        assert scorer.enable_conformal(own) is own
        _approved, score = scorer.review(_PROMPT, _ACTION)
        assert score.conformal_coverage == 0.8


class TestIntervalEmission:
    def test_uncalibrated_predictor_emits_vacuous_unreliable_interval(self):
        scorer = CoherenceScorer(threshold=0.3, use_nli=False)
        scorer.enable_conformal(coverage=0.9, min_samples=5)
        _approved, score = scorer.review(_PROMPT, _ACTION)
        assert (score.conformal_risk_lower, score.conformal_risk_upper) == (0.0, 1.0)
        assert score.conformal_calibration_size == 0
        assert score.conformal_reliable is False

    def test_calibrated_interval_bounds_the_point_risk(self):
        scorer = _calibrated_scorer()
        _approved, score = scorer.review(_PROMPT, _ACTION)
        point_risk = max(0.0, min(1.0, 1.0 - score.score))
        assert score.conformal_reliable is True
        assert score.conformal_calibration_size == 6
        assert score.conformal_coverage == 0.9
        assert 0.0 <= score.conformal_risk_lower <= point_risk
        assert point_risk <= score.conformal_risk_upper <= 1.0

    def test_approval_is_not_altered_by_the_interval(self):
        plain = CoherenceScorer(threshold=0.3, use_nli=False)
        conformal = _calibrated_scorer()
        assert (
            plain.review(_PROMPT, _ACTION)[0] == conformal.review(_PROMPT, _ACTION)[0]
        )


class TestPathParity:
    def test_cache_hit_path_carries_the_same_fields(self):
        scorer = _calibrated_scorer()
        first = scorer.review(_PROMPT, _ACTION)[1]
        cached = scorer.review(_PROMPT, _ACTION)[1]
        assert cached.conformal_risk_lower == first.conformal_risk_lower
        assert cached.conformal_risk_upper == first.conformal_risk_upper
        assert cached.conformal_reliable is True

    def test_batch_path_carries_the_same_fields_as_direct_review(self):
        scorer = _calibrated_scorer()
        direct = scorer.review(_PROMPT, _ACTION)[1]
        batch_scorer = _calibrated_scorer()
        results = batch_scorer.review_batch(
            [(_PROMPT, _ACTION), ("Name a primary colour.", "Red is one.")]
        )
        assert len(results) == 2
        for _approved, score in results:
            assert score.conformal_coverage == 0.9
            assert score.conformal_reliable is True
        assert results[0][1].conformal_risk_upper == direct.conformal_risk_upper
