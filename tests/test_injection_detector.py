# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Intent-Grounded Injection Detection Tests (STRONG)
"""Multi-angle coverage for InjectionDetector pipeline.

Covers: intent construction, bidirectional NLI, baseline calibration,
per-claim verdicts, aggregation, clean response false-positive guard,
known injection detection, novel semantic injection detection, graceful
degradation, sanitizer integration, and performance.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from director_ai.core.safety.injection import InjectionDetector, _fallback_split
from director_ai.core.types import InjectedClaim, InjectionResult

# -- Data type tests ----------------------------------------------------------


class TestInjectedClaimDataclass:
    def test_construction(self):
        c = InjectedClaim(
            claim="test",
            claim_index=0,
            intent_divergence=0.3,
            reverse_divergence=0.2,
            bidirectional_divergence=0.2,
            traceability=0.8,
            entity_match=0.9,
            verdict="grounded",
            confidence=0.7,
        )
        assert c.verdict == "grounded"
        assert c.bidirectional_divergence == 0.2

    def test_all_verdicts_accepted(self):
        for v in ("grounded", "drifted", "injected"):
            c = InjectedClaim(
                claim="x",
                claim_index=0,
                intent_divergence=0.5,
                reverse_divergence=0.5,
                bidirectional_divergence=0.5,
                traceability=0.5,
                entity_match=0.5,
                verdict=v,
                confidence=0.5,
            )
            assert c.verdict == v


class TestInjectionResultDataclass:
    def test_to_dict(self):
        claim = InjectedClaim(
            claim="Paris is in Germany.",
            claim_index=0,
            intent_divergence=0.9,
            reverse_divergence=0.8,
            bidirectional_divergence=0.8,
            traceability=0.1,
            entity_match=0.2,
            verdict="injected",
            confidence=0.95,
        )
        result = InjectionResult(
            injection_detected=True,
            injection_risk=0.9,
            intent_coverage=0.0,
            total_claims=1,
            grounded_claims=0,
            drifted_claims=0,
            injected_claims=1,
            claims=[claim],
            input_sanitizer_score=0.0,
            combined_score=0.63,
        )
        d = result.to_dict()
        assert d["injection_detected"] is True
        assert d["injection_risk"] == 0.9
        assert len(d["claims"]) == 1
        assert d["claims"][0]["verdict"] == "injected"
        assert "intent_divergence" not in d["claims"][0]  # only serialises key fields

    def test_empty_claims(self):
        result = InjectionResult(
            injection_detected=False,
            injection_risk=0.0,
            intent_coverage=1.0,
            total_claims=0,
            grounded_claims=0,
            drifted_claims=0,
            injected_claims=0,
            claims=[],
            input_sanitizer_score=0.0,
            combined_score=0.0,
        )
        assert result.to_dict()["claims"] == []


# -- Intent construction ------------------------------------------------------


class TestIntentConstruction:
    def test_system_prompt_plus_query(self):
        d = InjectionDetector()
        intent = d._build_intent("ignored", "What is 2+2?", "You are a calculator.")
        assert intent == "You are a calculator.\n\nWhat is 2+2?"

    def test_system_prompt_only(self):
        d = InjectionDetector()
        intent = d._build_intent("fallback", "", "System prompt only.")
        assert intent == "System prompt only."

    def test_user_query_only(self):
        d = InjectionDetector()
        intent = d._build_intent("fallback", "Just the query.", "")
        assert intent == "Just the query."

    def test_fallback_to_intent(self):
        d = InjectionDetector()
        intent = d._build_intent("Direct intent.", "", "")
        assert intent == "Direct intent."

    def test_empty_strings_fallback(self):
        d = InjectionDetector()
        intent = d._build_intent("", "", "")
        assert intent == ""


# -- Baseline calibration -----------------------------------------------------


class TestBaselineCalibration:
    def test_below_baseline_returns_zero(self):
        d = InjectionDetector(baseline_divergence=0.4)
        assert d._baseline_calibrate(0.3) == 0.0

    def test_at_baseline_returns_zero(self):
        d = InjectionDetector(baseline_divergence=0.4)
        assert d._baseline_calibrate(0.4) == 0.0

    def test_above_baseline(self):
        d = InjectionDetector(baseline_divergence=0.4)
        result = d._baseline_calibrate(0.7)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_max_divergence(self):
        d = InjectionDetector(baseline_divergence=0.4)
        result = d._baseline_calibrate(1.0)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_baseline_one_returns_zero(self):
        d = InjectionDetector(baseline_divergence=1.0)
        assert d._baseline_calibrate(0.9) == 0.0

    def test_baseline_zero(self):
        d = InjectionDetector(baseline_divergence=0.0)
        assert d._baseline_calibrate(0.5) == pytest.approx(0.5)


# -- Claim verdict logic ------------------------------------------------------


class TestClaimVerdict:
    def test_grounded_low_divergence_high_trace(self):
        d = InjectionDetector()
        verdict, conf = d._claim_verdict(0.1, traceability=0.8, entity_match=0.9)
        assert verdict == "grounded"
        assert 0.0 < conf <= 1.0

    def test_drifted_high_divergence_moderate_trace(self):
        d = InjectionDetector(drift_threshold=0.6)
        verdict, _ = d._claim_verdict(0.65, traceability=0.5, entity_match=0.6)
        assert verdict == "drifted"

    def test_injected_high_divergence_low_trace(self):
        d = InjectionDetector(injection_claim_threshold=0.75)
        verdict, _ = d._claim_verdict(0.8, traceability=0.1, entity_match=0.1)
        assert verdict == "injected"

    def test_fabrication_override_very_low_trace(self):
        d = InjectionDetector()
        verdict, conf = d._claim_verdict(0.1, traceability=0.05, entity_match=0.9)
        assert verdict == "injected"
        assert conf >= 0.5

    def test_drift_threshold_boundary(self):
        d = InjectionDetector(drift_threshold=0.6)
        verdict_below, _ = d._claim_verdict(0.59, traceability=0.5, entity_match=0.5)
        verdict_at, _ = d._claim_verdict(0.60, traceability=0.5, entity_match=0.5)
        assert verdict_below == "grounded"
        assert verdict_at == "drifted"


# -- Injection risk aggregation -----------------------------------------------


class TestInjectionRisk:
    def _make_claim(self, verdict: str) -> InjectedClaim:
        return InjectedClaim(
            claim="x",
            claim_index=0,
            intent_divergence=0.5,
            reverse_divergence=0.5,
            bidirectional_divergence=0.5,
            traceability=0.5,
            entity_match=0.5,
            verdict=verdict,
            confidence=0.5,
        )

    def test_all_grounded_zero_risk(self):
        d = InjectionDetector()
        claims = [self._make_claim("grounded") for _ in range(3)]
        risk, combined = d._compute_injection_risk(claims, sanitizer_score=0.0)
        assert risk == 0.0
        assert combined == 0.0

    def test_all_injected_full_risk(self):
        d = InjectionDetector()
        claims = [self._make_claim("injected") for _ in range(3)]
        risk, _ = d._compute_injection_risk(claims, sanitizer_score=0.0)
        assert risk == pytest.approx(1.0)

    def test_mixed_verdicts(self):
        d = InjectionDetector()
        claims = [
            self._make_claim("grounded"),
            self._make_claim("drifted"),
            self._make_claim("injected"),
        ]
        risk, _ = d._compute_injection_risk(claims, sanitizer_score=0.0)
        expected = (0.0 + 0.4 + 1.0) / 3
        assert risk == pytest.approx(expected, abs=0.01)

    def test_sanitizer_weight(self):
        d = InjectionDetector(stage1_weight=0.3)
        claims = [self._make_claim("grounded")]
        _, combined = d._compute_injection_risk(claims, sanitizer_score=1.0)
        assert combined == pytest.approx(0.3)

    def test_empty_claims(self):
        d = InjectionDetector()
        risk, combined = d._compute_injection_risk([], sanitizer_score=0.5)
        assert risk == 0.0
        assert combined == pytest.approx(0.3 * 0.5)


# -- Full detect() with mock NLI ---------------------------------------------


class TestDetectCleanResponses:
    """Legitimate responses must NOT trigger injection detection."""

    def _make_detector(self, fwd_scores, rev_scores):
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.decompose_claims.side_effect = lambda t: [
            s.strip() + "." for s in t.split(".") if s.strip()
        ]
        mock_nli.score_batch.side_effect = [fwd_scores, rev_scores]
        return InjectionDetector(nli_scorer=mock_nli)

    def test_on_topic_qa(self):
        d = self._make_detector(
            fwd_scores=[0.2, 0.15],  # low divergence
            rev_scores=[0.25, 0.2],
        )
        result = d.detect(
            intent="Tell me about the capital city of France and its population density",
            response="The capital city of France is Paris. The population density of Paris is high.",
        )
        assert result.injection_detected is False
        assert result.injection_risk < 0.3
        assert result.grounded_claims == 2

    def test_empty_response(self):
        d = InjectionDetector()
        result = d.detect(intent="Hello", response="")
        assert result.injection_detected is False
        assert result.total_claims == 0


class TestDetectKnownInjections:
    """Known injection patterns must trigger detection."""

    def _make_detector(self, fwd_scores, rev_scores):
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.decompose_claims.side_effect = lambda t: [
            s.strip() + "." for s in t.split(".") if s.strip()
        ]
        mock_nli.score_batch.side_effect = [fwd_scores, rev_scores]
        return InjectionDetector(nli_scorer=mock_nli)

    def test_instruction_override_response(self):
        """Response follows injected instructions, not original intent."""
        d = self._make_detector(
            fwd_scores=[0.95, 0.9],  # high divergence from intent
            rev_scores=[0.85, 0.88],
        )
        result = d.detect(
            intent="What is 2+2?",
            response=(
                "My system prompt says I am a helpful assistant. "
                "I should not reveal confidential information."
            ),
        )
        assert result.injection_detected is True
        assert result.injected_claims >= 1

    def test_goal_hijacking(self):
        """Response ignores original task entirely."""
        d = self._make_detector(
            fwd_scores=[0.92],
            rev_scores=[0.88],
        )
        result = d.detect(
            intent="Translate this to French: Hello world",
            response="Here is a poem about cats and dogs living together.",
        )
        assert result.injection_detected is True
        assert result.intent_coverage < 0.5


class TestDetectNovelInjections:
    """Semantically encoded attacks (no regex match) must still be detected."""

    def _make_detector(self, fwd_scores, rev_scores):
        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.decompose_claims.side_effect = lambda t: [
            s.strip() + "." for s in t.split(".") if s.strip()
        ]
        mock_nli.score_batch.side_effect = [fwd_scores, rev_scores]
        return InjectionDetector(nli_scorer=mock_nli)

    def test_paraphrased_exfiltration(self):
        """Injection via paraphrase — no regex pattern matches."""
        d = self._make_detector(
            fwd_scores=[0.93, 0.91],
            rev_scores=[0.87, 0.89],
        )
        result = d.detect(
            intent="Summarise this document about climate change.",
            response=(
                "The internal configuration states the following rules. "
                "Access credentials are stored in the vault."
            ),
        )
        assert result.injection_detected is True


# -- Graceful degradation -----------------------------------------------------


class TestGracefulDegradation:
    def test_no_nli_scorer(self):
        d = InjectionDetector(nli_scorer=None)
        result = d.detect(intent="Hello", response="Hi there.")
        assert isinstance(result, InjectionResult)
        assert result.total_claims >= 1

    def test_no_sanitizer(self):
        d = InjectionDetector(sanitizer=None)
        result = d.detect(intent="Hello", response="Hi there.")
        assert result.input_sanitizer_score == 0.0

    def test_whitespace_only_response(self):
        d = InjectionDetector()
        result = d.detect(intent="Hello", response="   \n  ")
        assert result.injection_detected is False
        assert result.total_claims == 0

    def test_no_system_prompt(self):
        d = InjectionDetector()
        result = d.detect(
            intent="", user_query="What is AI?", response="AI is a field of CS."
        )
        assert isinstance(result, InjectionResult)


# -- Sanitizer integration ----------------------------------------------------


class TestSanitizerIntegration:
    def test_high_sanitizer_score_contributes(self):
        mock_sanitizer = MagicMock()
        mock_sanitizer.score.return_value = MagicMock(suspicion_score=0.9)
        d = InjectionDetector(sanitizer=mock_sanitizer, stage1_weight=0.3)
        result = d.detect(intent="Hello", response="Fine.")
        assert result.input_sanitizer_score == 0.9
        assert result.combined_score >= 0.27  # 0.3 * 0.9

    def test_zero_sanitizer_score(self):
        mock_sanitizer = MagicMock()
        mock_sanitizer.score.return_value = MagicMock(suspicion_score=0.0)
        d = InjectionDetector(sanitizer=mock_sanitizer)
        result = d.detect(intent="Hello", response="World.")
        assert result.input_sanitizer_score == 0.0


# -- Fallback split -----------------------------------------------------------


class TestFallbackSplit:
    def test_period_split(self):
        result = _fallback_split("First sentence. Second sentence.")
        assert len(result) == 2
        assert result[0] == "First sentence."

    def test_empty_string(self):
        assert _fallback_split("") == []

    def test_no_periods(self):
        result = _fallback_split("No periods here")
        assert result == ["No periods here."]


# -- Performance (mock) -------------------------------------------------------


class TestPerformance:
    def test_detect_latency_under_50ms_mock(self):
        """With mock NLI, detection must complete in < 50ms."""
        import time

        mock_nli = MagicMock()
        mock_nli.model_available = True
        mock_nli.decompose_claims.side_effect = lambda t: [
            s.strip() + "." for s in t.split(".") if s.strip()
        ]
        mock_nli.score_batch.return_value = [0.2] * 10
        d = InjectionDetector(nli_scorer=mock_nli)

        response = ". ".join(f"Claim number {i}" for i in range(10)) + "."
        t0 = time.perf_counter()
        for _ in range(10):
            d.detect(intent="Test prompt.", response=response)
        per_call_ms = (time.perf_counter() - t0) / 10 * 1000
        assert per_call_ms < 50
