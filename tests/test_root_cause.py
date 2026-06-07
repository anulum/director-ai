# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Hallucination Root-Cause Analyzer Tests
"""Tests for prescriptive mechanistic root-cause analysis."""

from __future__ import annotations

from director_ai.core.interpretability import (
    HallucinationRootCauseAnalyzer,
    Recommendation,
    RootCauseDiagnosis,
)
from director_ai.core.interpretability.redeep import (
    HeadContribution,
    LayerContribution,
    MechanisticAttributionReport,
)
from director_ai.core.interpretability.root_cause import (
    ATTENTION_IGNORES_EVIDENCE,
    NO_HALLUCINATION,
    PARAMETRIC_OVERRIDE,
    UNDERACTIVE_COPYING,
)


def _report(
    *,
    is_hallucination: bool = True,
    risk: float = 0.8,
    layers: tuple[LayerContribution, ...] = (),
    heads: tuple[HeadContribution, ...] = (),
) -> MechanisticAttributionReport:
    return MechanisticAttributionReport(
        hallucination_risk=risk,
        is_hallucination=is_hallucination,
        knowledge_ffn_layers=layers,
        copying_heads=heads,
        reason="fixture",
    )


class TestNoHallucination:
    def test_clean_report_has_no_cause(self):
        d = HallucinationRootCauseAnalyzer().diagnose(_report(is_hallucination=False))
        assert d.is_hallucination is False
        assert d.dominant_cause == NO_HALLUCINATION
        assert d.recommendations == ()
        assert d.causes == ()


class TestCauses:
    def test_parametric_override(self):
        rep = _report(
            layers=(
                LayerContribution(
                    layer_index=18,
                    ffn_knowledge=0.9,
                    external_attention=0.1,
                    decoupled_risk=0.8,
                ),
            ),
        )
        d = HallucinationRootCauseAnalyzer().diagnose(rep)
        assert d.dominant_cause == PARAMETRIC_OVERRIDE
        assert d.implicated_layers == (18,)
        rec = next(r for r in d.recommendations if r.cause == PARAMETRIC_OVERRIDE)
        assert rec.targets == ("ffn_layer:18",)
        assert "down-weight parametric" in rec.action

    def test_attention_ignores_evidence(self):
        # Low external attention but FFN knowledge below the parametric threshold.
        rep = _report(
            layers=(
                LayerContribution(
                    layer_index=7,
                    ffn_knowledge=0.2,
                    external_attention=0.1,
                    decoupled_risk=0.6,
                ),
            ),
        )
        d = HallucinationRootCauseAnalyzer().diagnose(rep)
        assert d.dominant_cause == ATTENTION_IGNORES_EVIDENCE
        assert d.implicated_layers == (7,)

    def test_underactive_copying_heads(self):
        rep = _report(
            heads=(
                HeadContribution(layer_index=5, head_index=2, copying_score=0.1),
                HeadContribution(layer_index=6, head_index=0, copying_score=0.05),
            ),
        )
        d = HallucinationRootCauseAnalyzer().diagnose(rep)
        assert d.dominant_cause == UNDERACTIVE_COPYING
        assert d.implicated_heads == ((5, 2), (6, 0))
        rec = next(r for r in d.recommendations if r.cause == UNDERACTIVE_COPYING)
        assert rec.targets == ("head:5.2", "head:6.0")

    def test_multiple_causes_listed(self):
        rep = _report(
            layers=(
                LayerContribution(
                    layer_index=18,
                    ffn_knowledge=0.9,
                    external_attention=0.1,
                    decoupled_risk=0.9,
                ),
            ),
            heads=(HeadContribution(layer_index=5, head_index=2, copying_score=0.1),),
        )
        d = HallucinationRootCauseAnalyzer().diagnose(rep)
        assert set(d.causes) == {PARAMETRIC_OVERRIDE, UNDERACTIVE_COPYING}
        assert len(d.recommendations) == 2

    def test_healthy_components_yield_no_specific_cause(self):
        # Hallucination flagged, but FFN/heads look healthy → falls back to reason.
        rep = _report(
            layers=(
                LayerContribution(
                    layer_index=1,
                    ffn_knowledge=0.2,
                    external_attention=0.9,
                    decoupled_risk=0.1,
                ),
            ),
            heads=(HeadContribution(layer_index=2, head_index=1, copying_score=0.9),),
        )
        d = HallucinationRootCauseAnalyzer().diagnose(rep)
        assert d.causes == ()
        assert d.dominant_cause == "fixture"  # report.reason fallback
        assert d.recommendations == ()


class TestThresholds:
    def test_custom_thresholds(self):
        analyzer = HallucinationRootCauseAnalyzer(
            parametric_knowledge_threshold=0.95,
            low_attention_threshold=0.05,
            low_copying_threshold=0.5,
        )
        # ffn 0.9 < 0.95 and attention 0.1 > 0.05 → not parametric override.
        rep = _report(
            layers=(
                LayerContribution(
                    layer_index=3,
                    ffn_knowledge=0.9,
                    external_attention=0.1,
                    decoupled_risk=0.5,
                ),
            ),
            heads=(HeadContribution(layer_index=1, head_index=1, copying_score=0.4),),
        )
        d = analyzer.diagnose(rep)
        # Only the head (0.4 <= 0.5) triggers.
        assert d.causes == (UNDERACTIVE_COPYING,)


class TestSerialisation:
    def test_diagnosis_to_dict_tenant_safe(self):
        rep = _report(
            layers=(
                LayerContribution(
                    layer_index=18,
                    ffn_knowledge=0.9,
                    external_attention=0.1,
                    decoupled_risk=0.8,
                ),
            ),
        )
        d = HallucinationRootCauseAnalyzer().diagnose(rep).to_dict()
        assert set(d) == {
            "is_hallucination",
            "hallucination_risk",
            "dominant_cause",
            "causes",
            "recommendations",
            "implicated_layers",
            "implicated_heads",
        }
        assert d["recommendations"][0]["targets"] == ["ffn_layer:18"]

    def test_recommendation_to_dict(self):
        r = Recommendation(cause="c", action="do x", targets=("t1",))
        assert r.to_dict() == {"cause": "c", "action": "do x", "targets": ["t1"]}


class TestGuardWiring:
    def test_guard_root_cause_analyzer(self):
        from director_ai.core.config import DirectorConfig
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
        analyzer = guard.root_cause_analyzer
        assert guard.root_cause_analyzer is analyzer  # persists
        rep = _report(
            heads=(HeadContribution(layer_index=5, head_index=2, copying_score=0.1),),
        )
        diag = analyzer.diagnose(rep)
        assert isinstance(diag, RootCauseDiagnosis)
        assert diag.dominant_cause == UNDERACTIVE_COPYING
