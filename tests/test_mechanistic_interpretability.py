# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ReDeEP mechanistic attribution tests

"""Tests for ReDeEP mechanistic hallucination attribution.

Covers the decoupled per-layer risk (high parametric reliance + low external
context → hallucination), grounded responses, weight normalisation, top-k
ranking of Knowledge-FFN layers and Copying-Heads, the provider and array
helpers, threshold behaviour, and signal/parameter validation."""

from __future__ import annotations

import pytest

from director_ai.core.interpretability import (
    HeadSignal,
    LayerSignal,
    MechanisticAttributor,
)


def _hallucinating_layers() -> list[LayerSignal]:
    # High FFN parametric injection, low external-context attention.
    return [
        LayerSignal(layer_index=i, ffn_knowledge=0.9, external_attention=0.1)
        for i in range(4)
    ]


def _grounded_layers() -> list[LayerSignal]:
    return [
        LayerSignal(layer_index=i, ffn_knowledge=0.1, external_attention=0.9)
        for i in range(4)
    ]


class TestAttribution:
    def test_high_parametric_low_external_flags_hallucination(self):
        report = MechanisticAttributor().attribute(_hallucinating_layers())
        assert report.is_hallucination
        assert report.hallucination_risk == pytest.approx(0.9)
        assert "hallucination" in report.reason

    def test_grounded_response_low_risk(self):
        report = MechanisticAttributor().attribute(_grounded_layers())
        assert not report.is_hallucination
        assert report.hallucination_risk == pytest.approx(0.1)
        assert "grounded" in report.reason

    def test_ffn_only_weight_uses_ffn_knowledge(self):
        attr = MechanisticAttributor(ffn_weight=1.0, attention_weight=0.0)
        report = attr.attribute(
            [LayerSignal(layer_index=0, ffn_knowledge=0.8, external_attention=0.5)]
        )
        assert report.hallucination_risk == pytest.approx(0.8)

    def test_top_knowledge_layers_ranked(self):
        layers = [
            LayerSignal(layer_index=0, ffn_knowledge=0.2, external_attention=0.8),
            LayerSignal(layer_index=1, ffn_knowledge=0.95, external_attention=0.05),
            LayerSignal(layer_index=2, ffn_knowledge=0.6, external_attention=0.4),
        ]
        report = MechanisticAttributor(top_k=2).attribute(layers)
        assert len(report.knowledge_ffn_layers) == 2
        assert report.knowledge_ffn_layers[0].layer_index == 1
        assert (
            report.knowledge_ffn_layers[0].decoupled_risk
            >= report.knowledge_ffn_layers[1].decoupled_risk
        )

    def test_copying_heads_ranked_by_copying_score(self):
        heads = [
            HeadSignal(layer_index=0, head_index=0, copying_score=0.2),
            HeadSignal(layer_index=1, head_index=3, copying_score=0.9),
            HeadSignal(layer_index=2, head_index=1, copying_score=0.5),
        ]
        report = MechanisticAttributor(top_k=2).attribute(
            _hallucinating_layers(), heads
        )
        assert [(h.layer_index, h.head_index) for h in report.copying_heads] == [
            (1, 3),
            (2, 1),
        ]

    def test_threshold_boundary(self):
        layers = [LayerSignal(layer_index=0, ffn_knowledge=0.5, external_attention=0.5)]
        # risk = 0.5*0.5 + 0.5*0.5 = 0.5 → at threshold → hallucination.
        assert (
            MechanisticAttributor(risk_threshold=0.5).attribute(layers).is_hallucination
        )
        assert (
            not MechanisticAttributor(risk_threshold=0.51)
            .attribute(layers)
            .is_hallucination
        )


class TestProviderAndHelpers:
    def test_attribute_from_provider(self):
        class FakeProvider:
            def layer_signals(self):
                return _hallucinating_layers()

            def head_signals(self):
                return [HeadSignal(layer_index=0, head_index=0, copying_score=0.3)]

        report = MechanisticAttributor().attribute_from(FakeProvider())
        assert report.is_hallucination
        assert len(report.copying_heads) == 1

    def test_layer_signals_from_arrays(self):
        signals = MechanisticAttributor.layer_signals_from_arrays(
            [0.9, 0.1], [0.1, 0.9]
        )
        assert [s.layer_index for s in signals] == [0, 1]
        assert signals[0].ffn_knowledge == pytest.approx(0.9)

    def test_layer_arrays_length_mismatch(self):
        with pytest.raises(ValueError, match="length mismatch"):
            MechanisticAttributor.layer_signals_from_arrays([0.1], [0.1, 0.2])

    def test_head_signals_from_matrix(self):
        signals = MechanisticAttributor.head_signals_from_matrix([[0.1, 0.2], [0.3]])
        assert len(signals) == 3
        assert (signals[2].layer_index, signals[2].head_index) == (1, 0)


class TestValidation:
    def test_empty_layers_rejected(self):
        with pytest.raises(ValueError, match="at least one layer signal"):
            MechanisticAttributor().attribute([])

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            MechanisticAttributor(ffn_weight=-0.1)

    def test_zero_total_weight_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            MechanisticAttributor(ffn_weight=0.0, attention_weight=0.0)

    def test_bad_top_k(self):
        with pytest.raises(ValueError, match="top_k must be positive"):
            MechanisticAttributor(top_k=0)

    def test_bad_risk_threshold(self):
        with pytest.raises(ValueError, match="risk_threshold"):
            MechanisticAttributor(risk_threshold=1.5)

    def test_layer_signal_out_of_range(self):
        with pytest.raises(ValueError, match="ffn_knowledge"):
            LayerSignal(layer_index=0, ffn_knowledge=1.5, external_attention=0.5)

    def test_layer_signal_negative_index(self):
        with pytest.raises(ValueError, match="layer_index"):
            LayerSignal(layer_index=-1, ffn_knowledge=0.5, external_attention=0.5)

    def test_head_signal_validation(self):
        with pytest.raises(ValueError, match="copying_score"):
            HeadSignal(layer_index=0, head_index=0, copying_score=2.0)

    def test_head_signal_negative_layer(self):
        with pytest.raises(ValueError, match="layer_index"):
            HeadSignal(layer_index=-1, head_index=0, copying_score=0.5)

    def test_head_signal_negative_head(self):
        with pytest.raises(ValueError, match="head_index"):
            HeadSignal(layer_index=0, head_index=-1, copying_score=0.5)
