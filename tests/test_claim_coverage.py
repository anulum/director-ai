# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Layer C: Claim Coverage Tests
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Tests for claim decomposition + coverage scoring (Layer C).

Covers NLIScorer.score_claim_coverage(), config wiring,
scorer integration, and evidence propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.nli import NLIScorer
from director_ai.core.scorer import CoherenceScorer
from director_ai.core.types import ScoringEvidence


# ── NLIScorer.score_claim_coverage ──────────────────────────────────


class TestScoreClaimCoverage:
    """Unit tests for NLIScorer.score_claim_coverage()."""

    def _make_scorer(self, scores_map: dict[str, float] | None = None):
        """Build an NLIScorer with mocked score_chunked."""
        s = NLIScorer(use_model=False)
        default_score = 0.3
        if scores_map is None:
            scores_map = {}

        def fake_score_chunked(source, claim, **_kwargs):
            return scores_map.get(claim, default_score), []

        s.score_chunked = fake_score_chunked  # type: ignore[assignment]
        s.score = lambda p, h: scores_map.get(h, default_score)  # type: ignore[assignment]
        return s

    def test_single_claim_supported(self):
        scorer = self._make_scorer({"The sky is blue.": 0.1})
        cov, divs, claims = scorer.score_claim_coverage(
            "Source text about sky.",
            "The sky is blue.",
        )
        assert cov == 1.0
        assert len(claims) == 1
        assert divs[0] == 0.1

    def test_single_claim_unsupported(self):
        scorer = self._make_scorer({"The sky is green.": 0.8})
        cov, divs, claims = scorer.score_claim_coverage(
            "Source text.",
            "The sky is green.",
        )
        assert cov == 0.0
        assert divs[0] == 0.8

    def test_mixed_claims(self):
        scorer = self._make_scorer({
            "The sky is blue.": 0.1,   # supported
            "Water is wet.": 0.2,       # supported
            "Mars is flat.": 0.9,       # unsupported
        })
        cov, divs, claims = scorer.score_claim_coverage(
            "Source about sky and water.",
            "The sky is blue. Water is wet. Mars is flat.",
        )
        assert len(claims) == 3
        assert cov == pytest.approx(2.0 / 3.0)

    def test_all_supported(self):
        scorer = self._make_scorer({
            "Claim A.": 0.1,
            "Claim B.": 0.2,
        })
        cov, divs, claims = scorer.score_claim_coverage(
            "Source.", "Claim A. Claim B.",
        )
        assert cov == 1.0

    def test_none_supported(self):
        scorer = self._make_scorer({
            "Bad claim A.": 0.8,
            "Bad claim B.": 0.9,
        })
        cov, divs, claims = scorer.score_claim_coverage(
            "Source.", "Bad claim A. Bad claim B.",
        )
        assert cov == 0.0

    def test_custom_threshold(self):
        scorer = self._make_scorer({
            "Claim X.": 0.35,  # supported at 0.5, not at 0.3
        })
        cov_loose, _, _ = scorer.score_claim_coverage(
            "Source.", "Claim X.", support_threshold=0.5,
        )
        cov_strict, _, _ = scorer.score_claim_coverage(
            "Source.", "Claim X.", support_threshold=0.3,
        )
        assert cov_loose == 1.0
        assert cov_strict == 0.0

    def test_empty_summary_fallback(self):
        scorer = self._make_scorer()
        cov, divs, claims = scorer.score_claim_coverage("Source.", "")
        assert len(claims) == 1
        assert len(divs) == 1


# ── Config wiring ──────────────────────────────────────────────────


class TestClaimCoverageConfig:
    """Config fields for Layer C."""

    def test_defaults(self):
        cfg = DirectorConfig()
        assert cfg.nli_claim_coverage_enabled is True
        assert cfg.nli_claim_support_threshold == 0.6
        assert cfg.nli_claim_coverage_alpha == 0.4

    def test_summarization_profile(self):
        cfg = DirectorConfig.from_profile("summarization")
        assert cfg.nli_claim_coverage_enabled is True
        assert cfg.nli_claim_support_threshold == 0.6
        assert cfg.nli_claim_coverage_alpha == 0.4

    def test_config_wires_to_scorer(self):
        cfg = DirectorConfig(
            nli_claim_coverage_enabled=False,
            nli_claim_support_threshold=0.6,
            nli_claim_coverage_alpha=0.3,
        )
        scorer = cfg.build_scorer()
        assert scorer._claim_coverage_enabled is False
        assert scorer._claim_support_threshold == 0.6
        assert scorer._claim_coverage_alpha == 0.3

    def test_env_override(self):
        with patch.dict("os.environ", {
            "DIRECTOR_NLI_CLAIM_COVERAGE_ENABLED": "false",
            "DIRECTOR_NLI_CLAIM_COVERAGE_ALPHA": "0.6",
        }):
            cfg = DirectorConfig.from_env()
            assert cfg.nli_claim_coverage_enabled is False
            assert cfg.nli_claim_coverage_alpha == 0.6


# ── Scorer integration ─────────────────────────────────────────────


class TestScorerClaimCoverageIntegration:
    """Layer C integration into _summarization_factual_divergence."""

    def _make_scorer_with_mocked_nli(self, coverage=0.8, layer_a_div=0.3):
        """Build CoherenceScorer with mocked NLI for summarization mode."""
        scorer = CoherenceScorer(
            threshold=0.15,
            use_nli=True,
            w_logic=0.0,
            w_fact=1.0,
        )
        scorer._use_prompt_as_premise = True
        scorer._summarization_nli_baseline = 0.20
        scorer._claim_coverage_enabled = True
        scorer._claim_coverage_alpha = 0.4
        scorer._claim_support_threshold = 0.6

        mock_nli = MagicMock()
        mock_nli.model_available = True

        # Layer A: forward + reverse scoring
        mock_nli._score_chunked_with_counts.return_value = (
            layer_a_div, [layer_a_div], 1, 1,
        )
        mock_nli.score_chunked.return_value = (layer_a_div, [])

        # Layer C: claim coverage
        num_claims = 5
        supported = int(coverage * num_claims)
        divs = [0.1] * supported + [0.8] * (num_claims - supported)
        claims = [f"Claim {i}." for i in range(num_claims)]
        mock_nli.score_claim_coverage.return_value = (coverage, divs, claims)

        scorer._nli = mock_nli
        return scorer

    def test_layer_c_blending(self):
        """alpha * (1 - coverage) + (1 - alpha) * layer_a."""
        scorer = self._make_scorer_with_mocked_nli(coverage=0.8, layer_a_div=0.3)
        # Layer A after baseline: max(0, (0.3-0.20)/(1-0.20)) = 0.125
        # Layer C: 1 - 0.8 = 0.2
        # Final: 0.4 * 0.2 + 0.6 * 0.125 = 0.08 + 0.075 = 0.155
        div, ev = scorer._summarization_factual_divergence("doc", "summary")
        expected_layer_a = max(0, (0.3 - 0.20) / (1.0 - 0.20))
        expected = 0.4 * (1.0 - 0.8) + 0.6 * expected_layer_a
        assert div == pytest.approx(expected, abs=0.01)

    def test_layer_c_disabled(self):
        scorer = self._make_scorer_with_mocked_nli(coverage=0.5, layer_a_div=0.4)
        scorer._claim_coverage_enabled = False
        div, ev = scorer._summarization_factual_divergence("doc", "summary")
        expected_layer_a = max(0, (0.4 - 0.20) / (1.0 - 0.20))
        assert div == pytest.approx(expected_layer_a, abs=0.01)
        assert ev.claim_coverage is None

    def test_evidence_populated(self):
        scorer = self._make_scorer_with_mocked_nli(coverage=0.6, layer_a_div=0.25)
        _, ev = scorer._summarization_factual_divergence("doc", "summary")
        assert ev is not None
        assert ev.claim_coverage == pytest.approx(0.6)
        assert ev.per_claim_divergences is not None
        assert ev.claims is not None
        assert len(ev.claims) == 5

    def test_full_coverage_reduces_divergence(self):
        scorer = self._make_scorer_with_mocked_nli(coverage=1.0, layer_a_div=0.4)
        div, _ = scorer._summarization_factual_divergence("doc", "summary")
        layer_a = max(0, (0.4 - 0.20) / 0.80)
        # Layer C: alpha * 0 + (1-alpha) * layer_a = 0.6 * layer_a
        assert div < layer_a

    def test_zero_coverage_increases_divergence(self):
        scorer = self._make_scorer_with_mocked_nli(coverage=0.0, layer_a_div=0.15)
        div, _ = scorer._summarization_factual_divergence("doc", "summary")
        layer_a = max(0, (0.15 - 0.20) / 0.80)  # 0.0 (below baseline)
        # Layer C: alpha * 1.0 + (1-alpha) * 0.0 = 0.4
        assert div == pytest.approx(0.4, abs=0.01)


# ── ScoringEvidence fields ──────────────────────────────────────────


class TestScoringEvidenceClaimFields:
    """Claim coverage fields on ScoringEvidence dataclass."""

    def test_defaults_none(self):
        ev = ScoringEvidence(
            chunks=[], nli_premise="p", nli_hypothesis="h", nli_score=0.5,
        )
        assert ev.claim_coverage is None
        assert ev.per_claim_divergences is None
        assert ev.claims is None

    def test_set_fields(self):
        ev = ScoringEvidence(
            chunks=[], nli_premise="p", nli_hypothesis="h", nli_score=0.5,
            claim_coverage=0.75,
            per_claim_divergences=[0.1, 0.2, 0.8],
            claims=["A.", "B.", "C."],
        )
        assert ev.claim_coverage == 0.75
        assert len(ev.per_claim_divergences) == 3
        assert len(ev.claims) == 3


# ── Server evidence serialization ──────────────────────────────────


class TestServerEvidenceSerialization:
    """_evidence_to_dict includes claim coverage fields."""

    def test_without_claim_coverage(self):
        from director_ai.server import _evidence_to_dict

        ev = ScoringEvidence(
            chunks=[], nli_premise="p", nli_hypothesis="h", nli_score=0.5,
        )
        d = _evidence_to_dict(ev)
        assert "claim_coverage" not in d

    def test_with_claim_coverage(self):
        from director_ai.server import _evidence_to_dict

        ev = ScoringEvidence(
            chunks=[], nli_premise="p", nli_hypothesis="h", nli_score=0.5,
            claim_coverage=0.75,
            per_claim_divergences=[0.1, 0.8],
            claims=["Good.", "Bad."],
        )
        d = _evidence_to_dict(ev)
        assert d["claim_coverage"] == 0.75
        assert d["per_claim_divergences"] == [0.1, 0.8]
        assert d["claims"] == ["Good.", "Bad."]


# ── CoherenceScorer defaults ────────────────────────────────────────


class TestScorerClaimDefaults:
    """Default claim coverage attributes on CoherenceScorer."""

    def test_defaults(self):
        scorer = CoherenceScorer()
        assert scorer._claim_coverage_enabled is True
        assert scorer._claim_support_threshold == 0.6
        assert scorer._claim_coverage_alpha == 0.4
