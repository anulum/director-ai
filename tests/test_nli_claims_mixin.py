# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ClaimCoverageMixin composition and floor contracts

"""Contract tests for the claim-coverage mixin behind NLIScorer.

Claim decomposition, coverage, and attribution live in
``director_ai.core.scoring._nli_claims`` and compose into
:class:`NLIScorer` on top of the chunking mixin. These tests pin the
mixin chain and the pure-Python floor semantics of the coverage and
attribution reducers; the behaviour matrix over real backends stays in
``tests/test_cost_and_attribution.py`` and ``tests/test_nli_scorer.py``.
"""

from __future__ import annotations

import pytest

import director_ai.core.scoring._nli_accel as nli_accel
from director_ai.core.scoring._nli_chunking import ChunkingMixin
from director_ai.core.scoring._nli_claims import ClaimCoverageMixin
from director_ai.core.scoring.nli import NLIScorer


class TestClaimsComposition:
    def test_nli_scorer_composes_claims_over_chunking(self):
        assert NLIScorer.__mro__.index(ClaimCoverageMixin) < NLIScorer.__mro__.index(
            ChunkingMixin
        )
        for name in (
            "decompose_claims",
            "score_decomposed",
            "score_claim_coverage",
            "score_claim_coverage_with_attribution",
        ):
            assert getattr(NLIScorer, name) is getattr(ClaimCoverageMixin, name)


class TestPythonFloorReduction:
    @pytest.fixture(autouse=True)
    def _force_python_floor(self, monkeypatch):
        monkeypatch.setattr(nli_accel, "_RUST_NLI", False)

    def test_coverage_counts_supported_claims_below_threshold(self):
        scorer = NLIScorer(use_model=False)
        source = "The reactor uses heavy water. It was built in 1962."
        summary = "The reactor uses heavy water. It was built in 1962."
        coverage, divs, claims = scorer.score_claim_coverage(
            source, summary, support_threshold=0.6
        )
        assert len(claims) == 2
        assert len(divs) == 2
        assert coverage == pytest.approx(1.0)

    def test_attribution_maps_each_claim_to_its_best_source_sentence(self):
        scorer = NLIScorer(use_model=False)
        source = "Alpha is red. Beta is blue."
        summary = "Alpha is red. Beta is blue."
        coverage, divs, claims, attrs = scorer.score_claim_coverage_with_attribution(
            source, summary, support_threshold=0.6
        )
        assert [a.claim_index for a in attrs] == [0, 1]
        assert attrs[0].source_sentence == "Alpha is red."
        assert attrs[1].source_sentence == "Beta is blue."
        assert all(a.supported for a in attrs)
        assert coverage == pytest.approx(1.0)

    def test_attribution_pair_limit_guards_quadratic_blowup(self):
        scorer = NLIScorer(use_model=False)
        source = " ".join(f"Fact number {i} stands." for i in range(101))
        summary = " ".join(f"Claim number {i} holds." for i in range(100))
        with pytest.raises(ValueError, match="exceeding limit"):
            scorer.score_claim_coverage_with_attribution(source, summary)

    def test_empty_summary_scores_whole_text_as_single_claim(self):
        scorer = NLIScorer(use_model=False)
        coverage, divs, claims = scorer.score_claim_coverage("source text", "")
        assert claims == [""]
        assert len(divs) == 1
        assert coverage in (0.0, 1.0)
