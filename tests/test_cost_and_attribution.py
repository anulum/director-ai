# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Â© Concepts 1996â€“2026 Miroslav Ĺ otek. All rights reserved.
# Â© Code 2020â€“2026 Miroslav Ĺ otek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI â€” Cost Transparency & Sentence Attribution Tests

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# â”€â”€ ClaimAttribution dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestClaimAttribution:
    def test_fields(self):
        from director_ai.core.types import ClaimAttribution

        a = ClaimAttribution(
            claim="The sky is blue.",
            claim_index=0,
            source_sentence="Due to Rayleigh scattering, the sky appears blue.",
            source_index=2,
            divergence=0.15,
            supported=True,
        )
        assert a.claim_index == 0
        assert a.supported is True
        assert a.divergence == 0.15

    def test_unsupported(self):
        from director_ai.core.types import ClaimAttribution

        a = ClaimAttribution(
            claim="Water boils at 50C.",
            claim_index=1,
            source_sentence="Water boils at 100 degrees Celsius.",
            source_index=0,
            divergence=0.85,
            supported=False,
        )
        assert a.supported is False


# â”€â”€ ScoringEvidence new fields â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestScoringEvidenceNewFields:
    def test_defaults_none(self):
        from director_ai.core.types import EvidenceChunk, ScoringEvidence

        ev = ScoringEvidence(
            chunks=[EvidenceChunk(text="x", distance=0.0)],
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.3,
        )
        assert ev.token_count is None
        assert ev.estimated_cost_usd is None
        assert ev.attributions is None

    def test_with_values(self):
        from director_ai.core.types import (
            ClaimAttribution,
            EvidenceChunk,
            ScoringEvidence,
        )

        ev = ScoringEvidence(
            chunks=[EvidenceChunk(text="x", distance=0.0)],
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.3,
            token_count=1024,
            estimated_cost_usd=0.01024,
            attributions=[
                ClaimAttribution(
                    claim="c",
                    claim_index=0,
                    source_sentence="s",
                    source_index=0,
                    divergence=0.2,
                    supported=True,
                ),
            ],
        )
        assert ev.token_count == 1024
        assert ev.estimated_cost_usd == pytest.approx(0.01024)
        assert len(ev.attributions) == 1


# â”€â”€ NLIScorer token counting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestNLITokenCounting:
    def test_reset_and_properties(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        assert scorer.last_token_count == 0
        assert scorer.last_estimated_cost == 0.0

        scorer._last_token_count = 500
        assert scorer.last_token_count == 500
        assert scorer.last_estimated_cost == pytest.approx(500 * 1e-5)

        scorer.reset_token_counter()
        assert scorer.last_token_count == 0

    def test_custom_cost_per_token(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False, cost_per_token=0.001)
        scorer._last_token_count = 100
        assert scorer.last_estimated_cost == pytest.approx(0.1)


# â”€â”€ score_claim_coverage_with_attribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestClaimCoverageAttribution:
    def _mock_scorer(self, batch_scores):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        scorer.score_batch = MagicMock(return_value=batch_scores)
        return scorer

    def test_single_claim_single_source(self):
        scorer = self._mock_scorer([0.2])

        coverage, divs, claims, attrs = scorer.score_claim_coverage_with_attribution(
            source="The sky is blue.",
            summary="The sky is blue.",
            support_threshold=0.6,
        )
        assert coverage == 1.0
        assert len(attrs) == 1
        assert attrs[0].supported is True
        assert attrs[0].claim_index == 0
        assert attrs[0].source_index == 0
        assert attrs[0].divergence == 0.2

    def test_multi_claim_picks_best_source(self):
        # 2 claims Ă— 3 source sentences = 6 scores
        # claim0: [0.8, 0.3, 0.5] â†’ best at idx 1 (0.3, supported)
        # claim1: [0.7, 0.9, 0.1] â†’ best at idx 2 (0.1, supported)
        scorer = self._mock_scorer([0.8, 0.3, 0.5, 0.7, 0.9, 0.1])

        coverage, divs, claims, attrs = scorer.score_claim_coverage_with_attribution(
            source="Sentence A. Sentence B. Sentence C.",
            summary="Claim one. Claim two.",
            support_threshold=0.6,
        )
        assert len(attrs) == 2
        assert attrs[0].source_index == 1
        assert attrs[0].divergence == pytest.approx(0.3)
        assert attrs[0].supported is True
        assert attrs[1].source_index == 2
        assert attrs[1].divergence == pytest.approx(0.1)
        assert attrs[1].supported is True
        assert coverage == 1.0

    def test_unsupported_claim(self):
        # 2 claims Ă— 2 source sentences = 4 scores
        # claim0: [0.2, 0.3] â†’ best=0.2 (supported)
        # claim1: [0.8, 0.7] â†’ best=0.7 (unsupported, threshold=0.6)
        scorer = self._mock_scorer([0.2, 0.3, 0.8, 0.7])

        coverage, divs, claims, attrs = scorer.score_claim_coverage_with_attribution(
            source="Source one. Source two.",
            summary="Good claim. Bad claim.",
            support_threshold=0.6,
        )
        assert coverage == 0.5
        assert attrs[0].supported is True
        assert attrs[1].supported is False
        assert attrs[1].divergence == pytest.approx(0.7)

    def test_empty_claims(self):
        from director_ai.core.nli import NLIScorer

        scorer = NLIScorer(use_model=False)
        coverage, divs, claims, attrs = scorer.score_claim_coverage_with_attribution(
            source="Some source text here.",
            summary="",
        )
        assert len(attrs) == 1
        assert attrs[0].claim == ""


# â”€â”€ server _evidence_to_dict â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestEvidenceToDictExtensions:
    def test_includes_attributions(self):
        from director_ai.core.types import (
            ClaimAttribution,
            EvidenceChunk,
            ScoringEvidence,
        )
        from director_ai.server import _evidence_to_dict

        ev = ScoringEvidence(
            chunks=[EvidenceChunk(text="x", distance=0.0)],
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.3,
            attributions=[
                ClaimAttribution(
                    claim="c",
                    claim_index=0,
                    source_sentence="s",
                    source_index=1,
                    divergence=0.2,
                    supported=True,
                ),
            ],
        )
        d = _evidence_to_dict(ev)
        assert "attributions" in d
        assert d["attributions"][0]["claim"] == "c"
        assert d["attributions"][0]["source_index"] == 1

    def test_includes_cost(self):
        from director_ai.core.types import EvidenceChunk, ScoringEvidence
        from director_ai.server import _evidence_to_dict

        ev = ScoringEvidence(
            chunks=[EvidenceChunk(text="x", distance=0.0)],
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.3,
            token_count=512,
            estimated_cost_usd=0.00512,
        )
        d = _evidence_to_dict(ev)
        assert d["token_count"] == 512
        assert d["estimated_cost_usd"] == pytest.approx(0.00512)

    def test_omits_when_none(self):
        from director_ai.core.types import EvidenceChunk, ScoringEvidence
        from director_ai.server import _evidence_to_dict

        ev = ScoringEvidence(
            chunks=[EvidenceChunk(text="x", distance=0.0)],
            nli_premise="p",
            nli_hypothesis="h",
            nli_score=0.3,
        )
        d = _evidence_to_dict(ev)
        assert "token_count" not in d
        assert "attributions" not in d


# â”€â”€ Public API exports â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


class TestExports:
    def test_claim_attribution_in_core_all(self):
        from director_ai.core import __all__ as core_all

        assert "ClaimAttribution" in core_all

    def test_claim_attribution_in_package_all(self):
        from director_ai import __all__ as pkg_all

        assert "ClaimAttribution" in pkg_all

    def test_import_from_package(self):
        from director_ai import ClaimAttribution

        assert ClaimAttribution is not None

    def test_version_bumped(self):
        from director_ai import __version__

        assert __version__ == "3.9.2"
