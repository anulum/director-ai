# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — selective prediction tests

from __future__ import annotations

import pytest

from director_ai.core.scoring.selective_prediction import (
    selective_prediction_metrics,
)
from director_ai.core.types import CoherenceScore


def _score(approved: bool, abstained: bool = False) -> CoherenceScore:
    return CoherenceScore(
        score=0.5,
        approved=approved,
        h_logical=0.0,
        h_factual=0.5,
        abstained=abstained,
    )


class TestSelectivePredictionMetrics:
    def test_empty_results_rejected(self):
        with pytest.raises(ValueError, match="at least one result"):
            selective_prediction_metrics([])

    def test_coverage_and_selective_accuracy(self):
        results = [
            (_score(approved=True), False),  # approved grounded -> correct
            (_score(approved=False), True),  # rejected halluc -> correct
            (_score(approved=True), True),  # approved halluc -> wrong
            (_score(approved=False, abstained=True), True),  # abstained
        ]
        report = selective_prediction_metrics(results)
        assert report.total == 4
        assert report.covered == 3
        assert report.abstained == 1
        assert report.coverage == pytest.approx(0.75)
        assert report.selective_accuracy == pytest.approx(2 / 3)
        assert report.selective_error == pytest.approx(1 / 3)
        # Overall counts abstentions as misses: 2 correct / 4 total.
        assert report.overall_accuracy == pytest.approx(0.5)

    def test_abstention_cannot_inflate_selective_accuracy(self):
        # A guard that abstains on the case it would get wrong keeps a perfect
        # selective accuracy but pays in coverage and overall accuracy.
        results = [
            (_score(approved=True), False),  # correct, covered
            (_score(approved=True, abstained=True), True),  # would be wrong; abstained
        ]
        report = selective_prediction_metrics(results)
        assert report.selective_accuracy == pytest.approx(1.0)
        assert report.coverage == pytest.approx(0.5)
        assert report.overall_accuracy == pytest.approx(0.5)

    def test_all_abstained_has_no_selective_accuracy(self):
        results = [
            (_score(approved=True, abstained=True), False),
            (_score(approved=False, abstained=True), True),
        ]
        report = selective_prediction_metrics(results)
        assert report.covered == 0
        assert report.coverage == pytest.approx(0.0)
        assert report.selective_accuracy is None
        assert report.selective_error is None
        assert report.overall_accuracy == pytest.approx(0.0)

    def test_to_dict_shape(self):
        report = selective_prediction_metrics([(_score(approved=True), False)])
        d = report.to_dict()
        assert d["total"] == 1
        assert d["coverage"] == 1.0
        assert d["selective_accuracy"] == 1.0
        assert d["overall_accuracy"] == 1.0

    def test_to_dict_none_selective_accuracy_serialises(self):
        report = selective_prediction_metrics(
            [(_score(approved=True, abstained=True), False)]
        )
        d = report.to_dict()
        assert d["selective_accuracy"] is None
        assert d["selective_error"] is None
