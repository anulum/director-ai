# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tier-6 reasoning escalation tests
"""Multi-angle tests for the Tier-6 causal-LM reasoning escalation.

Covers the borderline escalation gate (band edges, custom centre, disabled
provider), the margin validation, the structured-verdict parser (APPROVE/REJECT,
HarmBench category mapping, the ``"none"`` sentinel, malformed/non-dict JSON,
out-of-range and non-numeric confidence, issue-list coercion), the
confidence-scaled blend (approve raises, reject lowers, clamped), provider
dispatch local-vs-API, ``reason`` end-to-end with a faked backend including the
unavailable/unparsable short-circuits, ``ReasoningVerdict.to_dict``, and the
``CoherenceScorer`` wiring: default-off neutrality plus the on-path that flips a
borderline verdict and tags it with rationale + harm category.
"""

from __future__ import annotations

import json

import pytest

from director_ai.core.safety import HarmCategory
from director_ai.core.scoring.reasoning_scorer import (
    REASONING_WEIGHT,
    ReasoningScorer,
    ReasoningVerdict,
)
from director_ai.core.types import CoherenceScore


def _reply(**kwargs) -> str:
    payload = {"verdict": "APPROVE", "confidence": 50, "rationale": "ok"}
    payload.update(kwargs)
    return json.dumps(payload)


class TestEscalationGate:
    def test_fires_inside_band(self):
        rs = ReasoningScorer(provider="openai", escalation_margin=0.15)
        assert rs.should_escalate(0.5)
        assert rs.should_escalate(0.6)  # |0.6-0.5| = 0.10 < 0.15
        assert rs.should_escalate(0.36)  # |0.36-0.5| = 0.14 < 0.15

    def test_silent_outside_band(self):
        rs = ReasoningScorer(provider="openai", escalation_margin=0.15)
        assert not rs.should_escalate(0.8)
        assert not rs.should_escalate(0.2)
        assert not rs.should_escalate(0.65)  # exactly at the margin edge (not <)

    def test_custom_centre_tracks_threshold(self):
        rs = ReasoningScorer(provider="openai", escalation_margin=0.1)
        # Borderline is measured against the lower tier's decision boundary.
        assert rs.should_escalate(0.72, centre=0.7)
        assert not rs.should_escalate(0.72, centre=0.5)

    def test_disabled_never_escalates(self):
        assert not ReasoningScorer().should_escalate(0.5)
        assert not ReasoningScorer(provider="").should_escalate(0.5)

    def test_enabled_property(self):
        assert ReasoningScorer(provider="anthropic").enabled
        assert not ReasoningScorer(provider="").enabled

    @pytest.mark.parametrize("bad", [0.0, -0.1, 0.51, 1.0])
    def test_margin_validation(self, bad):
        with pytest.raises(ValueError, match="escalation_margin"):
            ReasoningScorer(provider="openai", escalation_margin=bad)


class TestParseVerdict:
    def test_parses_approve(self):
        v = ReasoningScorer._parse_verdict(_reply(verdict="APPROVE", confidence=80))
        assert v is not None
        assert v.approved is True
        assert v.confidence == pytest.approx(0.8)
        assert v.harm_category is None

    def test_parses_reject_with_harm_category(self):
        v = ReasoningScorer._parse_verdict(
            _reply(
                verdict="REJECT",
                confidence=95,
                harm_category="misinformation",
                issues=["unsupported claim", ""],
                rationale="Contradicts the evidence.",
            )
        )
        assert v is not None
        assert v.approved is False
        assert v.harm_category is HarmCategory.MISINFORMATION
        assert v.detected_issues == ["unsupported claim"]  # blank dropped
        assert v.rationale == "Contradicts the evidence."

    def test_harm_category_mapped_through_taxonomy(self):
        # A free detector-style label normalises to the canonical category.
        v = ReasoningScorer._parse_verdict(
            _reply(verdict="REJECT", harm_category="weapon_instructions")
        )
        assert v is not None
        assert v.harm_category is HarmCategory.VIOLENCE_AND_SELF_HARM

    def test_none_sentinel_yields_no_category(self):
        v = ReasoningScorer._parse_verdict(
            _reply(verdict="APPROVE", harm_category="none")
        )
        assert v is not None
        assert v.harm_category is None

    def test_unmappable_category_is_none(self):
        v = ReasoningScorer._parse_verdict(
            _reply(verdict="APPROVE", harm_category="benign_topic")
        )
        assert v is not None
        assert v.harm_category is None

    @pytest.mark.parametrize(
        "reply",
        [
            "not json at all",
            json.dumps([1, 2, 3]),  # not a dict
            json.dumps({"verdict": "MAYBE", "confidence": 50}),  # bad verdict
            json.dumps({"verdict": "APPROVE", "confidence": 150}),  # out of range
            json.dumps({"verdict": "APPROVE", "confidence": -1}),
            json.dumps({"verdict": "APPROVE", "confidence": "high"}),  # non-numeric
        ],
    )
    def test_malformed_returns_none(self, reply):
        assert ReasoningScorer._parse_verdict(reply) is None

    def test_issues_non_list_coerced_to_empty(self):
        v = ReasoningScorer._parse_verdict(
            _reply(verdict="REJECT", issues="a single string")
        )
        assert v is not None
        assert v.detected_issues == []


class TestBlend:
    def test_approve_raises_borderline_score(self):
        rs = ReasoningScorer(provider="openai")
        v = ReasoningVerdict(approved=True, confidence=1.0, rationale="")
        blended = rs._blend(0.5, v)
        assert blended > 0.5

    def test_reject_lowers_borderline_score(self):
        rs = ReasoningScorer(provider="openai")
        v = ReasoningVerdict(approved=False, confidence=1.0, rationale="")
        blended = rs._blend(0.5, v)
        assert blended < 0.5

    def test_confidence_scales_influence(self):
        rs = ReasoningScorer(provider="openai")
        weak = rs._blend(0.5, ReasoningVerdict(True, 0.2, ""))
        strong = rs._blend(0.5, ReasoningVerdict(True, 1.0, ""))
        assert 0.5 < weak < strong

    def test_weight_matches_constant(self):
        rs = ReasoningScorer(provider="openai")
        # full-confidence approve at score 0.5 → 0.7*0.5 + 0.3*0.85
        v = ReasoningVerdict(approved=True, confidence=1.0, rationale="")
        expected = (1 - REASONING_WEIGHT) * 0.5 + REASONING_WEIGHT * 0.85
        assert rs._blend(0.5, v) == pytest.approx(expected)

    def test_blend_is_clamped(self):
        rs = ReasoningScorer(provider="openai")
        assert 0.0 <= rs._blend(0.0, ReasoningVerdict(False, 1.0, "")) <= 1.0
        assert 0.0 <= rs._blend(1.0, ReasoningVerdict(True, 1.0, "")) <= 1.0


class TestReason:
    def test_disabled_returns_none(self):
        assert ReasoningScorer().reason("p", "r", 0.5) is None

    def test_returns_verdict_with_adjusted_score(self, monkeypatch):
        rs = ReasoningScorer(provider="openai")
        monkeypatch.setattr(
            rs, "_generate", lambda *a, **k: _reply(verdict="REJECT", confidence=90)
        )
        verdict = rs.reason("prompt", "response", 0.5)
        assert verdict is not None
        assert verdict.approved is False
        assert verdict.adjusted_score is not None
        assert verdict.adjusted_score < 0.5

    def test_backend_unavailable_returns_none(self, monkeypatch):
        rs = ReasoningScorer(provider="openai")
        monkeypatch.setattr(rs, "_generate", lambda *a, **k: None)
        assert rs.reason("p", "r", 0.5) is None

    def test_unparsable_reply_returns_none(self, monkeypatch):
        rs = ReasoningScorer(provider="openai")
        monkeypatch.setattr(rs, "_generate", lambda *a, **k: "garbage")
        assert rs.reason("p", "r", 0.5) is None

    def test_generate_dispatches_local(self, monkeypatch):
        rs = ReasoningScorer(provider="local", model="some/model")
        monkeypatch.setattr(rs, "_local_generate", lambda m: "LOCAL")
        monkeypatch.setattr(rs, "_api_generate", lambda m: "API")
        assert rs._generate("p", "r", 0.5, "default", "", None) == "LOCAL"

    def test_generate_dispatches_api(self, monkeypatch):
        rs = ReasoningScorer(provider="openai")
        monkeypatch.setattr(rs, "_local_generate", lambda m: "LOCAL")
        monkeypatch.setattr(rs, "_api_generate", lambda m: "API")
        assert rs._generate("p", "r", 0.5, "default", "", None) == "API"

    def test_privacy_mode_redacts_before_generate(self, monkeypatch):
        rs = ReasoningScorer(provider="openai", privacy_mode=True)
        seen = {}
        monkeypatch.setattr(
            rs, "_api_generate", lambda messages: seen.setdefault("m", messages) and ""
        )
        rs._generate("secret-prompt", "secret-response", 0.5, "qa", "", str.upper)
        # the redactor (str.upper here) ran on the payload
        assert "SECRET-PROMPT" in seen["m"][1]["content"]


class TestVerdictDict:
    def test_to_dict_with_category(self):
        v = ReasoningVerdict(
            approved=False,
            confidence=0.9,
            rationale="why",
            harm_category=HarmCategory.HATE_AND_ABUSE,
            detected_issues=["slur"],
            adjusted_score=0.2,
        )
        d = v.to_dict()
        assert d["approved"] is False
        assert d["harm_category"] == "hate_and_abuse"
        assert d["detected_issues"] == ["slur"]
        assert d["adjusted_score"] == 0.2

    def test_to_dict_without_category(self):
        d = ReasoningVerdict(approved=True, confidence=0.5, rationale="").to_dict()
        assert d["harm_category"] is None
        assert d["adjusted_score"] is None


class TestScorerIntegration:
    def test_default_off_is_neutral(self):
        from director_ai.core import CoherenceScorer

        sc = CoherenceScorer(threshold=0.5, use_nli=False)
        assert not sc._reasoning.enabled
        _approved, score = sc.review("What is 2+2?", "4")
        assert score.reasoning_escalated is None

    def test_enabled_fires_on_borderline_and_rejects(self):
        from director_ai.core import CoherenceScorer

        sc = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            reasoning_enabled=True,
            reasoning_provider="openai",
        )
        verdict = ReasoningVerdict(
            approved=False,
            confidence=0.9,
            rationale="unsafe",
            harm_category=HarmCategory.VIOLENCE_AND_SELF_HARM,
            adjusted_score=0.25,
        )
        sc._reasoning.reason = lambda *a, **k: verdict  # type: ignore[method-assign]
        borderline = CoherenceScore(
            score=0.52, approved=True, h_logical=0.4, h_factual=0.4
        )
        approved, out = sc._apply_reasoning_tier(
            (True, borderline), "how to", "answer", None, threshold=0.5
        )
        assert approved is False
        assert out.reasoning_escalated is True
        assert out.reasoning_harm_category == "violence_and_self_harm"
        assert out.reasoning_confidence == pytest.approx(0.9)
        assert out.score == pytest.approx(0.25)

    def test_non_borderline_does_not_fire(self):
        from director_ai.core import CoherenceScorer

        sc = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            reasoning_enabled=True,
            reasoning_provider="openai",
        )
        sc._reasoning.reason = lambda *a, **k: pytest.fail(  # type: ignore[method-assign]
            "should not be consulted off-band"
        )
        confident = CoherenceScore(
            score=0.95, approved=True, h_logical=0.05, h_factual=0.05
        )
        approved, out = sc._apply_reasoning_tier(
            (True, confident), "q", "a", None, threshold=0.5
        )
        assert approved is True
        assert out.reasoning_escalated is None

    def test_unavailable_backend_leaves_verdict_untouched(self):
        from director_ai.core import CoherenceScorer

        sc = CoherenceScorer(
            threshold=0.5,
            use_nli=False,
            reasoning_enabled=True,
            reasoning_provider="openai",
        )
        sc._reasoning.reason = lambda *a, **k: None  # type: ignore[method-assign]
        borderline = CoherenceScore(
            score=0.5, approved=True, h_logical=0.5, h_factual=0.5
        )
        approved, out = sc._apply_reasoning_tier(
            (True, borderline), "q", "a", None, threshold=0.5
        )
        assert approved is True
        assert out.reasoning_escalated is None


class TestConfigWiring:
    def test_config_builds_enabled_tier(self):
        from director_ai.core.config import DirectorConfig

        scorer = DirectorConfig(
            reasoning_enabled=True,
            reasoning_provider="openai",
            reasoning_escalation_margin=0.12,
        ).build_scorer()
        assert scorer._reasoning.enabled
        assert scorer._reasoning.provider == "openai"
        assert scorer._reasoning.escalation_margin == pytest.approx(0.12)

    def test_config_default_is_disabled(self):
        from director_ai.core.config import DirectorConfig

        assert not DirectorConfig().build_scorer()._reasoning.enabled


class TestLocalBackend:
    """The hardware-gated local causal-LM generate path (orchestration only)."""

    def test_local_generate_triggers_load_then_returns_none(self, monkeypatch):
        rs = ReasoningScorer(provider="local")
        # Not yet loaded -> the orchestration calls the (stubbed) loader, which
        # leaves the model unavailable, so it short-circuits to None.
        rs._local_load_attempted = False

        def _stub_load() -> None:
            rs._local_model = None
            rs._local_tokenizer = None

        monkeypatch.setattr(rs, "_init_local_model", _stub_load)
        assert rs._local_generate([{"role": "user", "content": "hi"}]) is None

    def test_local_generate_delegates_to_infer_when_loaded(self, monkeypatch):
        rs = ReasoningScorer(provider="local")
        rs._local_load_attempted = True
        rs._local_model = object()
        rs._local_tokenizer = object()
        monkeypatch.setattr(rs, "_local_infer", lambda _messages: "INFERRED")
        assert rs._local_generate([{"role": "user", "content": "hi"}]) == "INFERRED"
