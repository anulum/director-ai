# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for the active-labelling cockpit.

Covers item validation, uncertainty ranking, false-halt vs missed-hallucination
breakdown, the threshold trade-off curve and recommendation (with weights and
per-domain filtering), deterministic packet export, and the ProductionGuard
integration.
"""

from __future__ import annotations

import pytest

from director_ai.core.labelling_cockpit import (
    ActiveLabellingCockpit,
    LabelItem,
)

# ── LabelItem ───────────────────────────────────────────────────────────


class TestLabelItem:
    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match="item_id is required"):
            LabelItem(" ", 0.5, True)

    def test_score_range(self):
        with pytest.raises(ValueError, match="score must be"):
            LabelItem("a", 1.5, True)

    def test_bad_label_rejected(self):
        with pytest.raises(ValueError, match="label must be one of"):
            LabelItem("a", 0.5, True, label="maybe")

    def test_labelled_property(self):
        assert not LabelItem("a", 0.5, True).labelled
        assert LabelItem("a", 0.5, True, label="grounded").labelled

    def test_is_hallucination(self):
        assert LabelItem("a", 0.5, True, label="hallucination").is_hallucination
        assert not LabelItem("a", 0.5, True, label="grounded").is_hallucination

    def test_to_packet_row(self):
        row = LabelItem(
            "a", 0.5, True, domain="finance", label="grounded"
        ).to_packet_row()
        assert row["item_id"] == "a"
        assert row["domain"] == "finance"
        assert row["label"] == "grounded"


# ── ranking ─────────────────────────────────────────────────────────────


class TestRanking:
    def test_threshold_validation(self):
        with pytest.raises(ValueError, match="threshold must be"):
            ActiveLabellingCockpit(threshold=2.0)

    def test_ranks_unlabelled_by_proximity(self):
        cockpit = ActiveLabellingCockpit(threshold=0.6)
        items = [
            LabelItem("far", 0.1, False),
            LabelItem("near", 0.58, False),
            LabelItem("mid", 0.4, False),
        ]
        ranked = cockpit.rank_for_labelling(items)
        assert [i.item_id for i in ranked] == ["near", "mid", "far"]

    def test_excludes_labelled(self):
        cockpit = ActiveLabellingCockpit(threshold=0.6)
        items = [
            LabelItem("done", 0.6, True, label="grounded"),
            LabelItem("todo", 0.59, False),
        ]
        assert [i.item_id for i in cockpit.rank_for_labelling(items)] == ["todo"]

    def test_top_n_limits(self):
        cockpit = ActiveLabellingCockpit(threshold=0.6)
        items = [LabelItem(f"i{n}", 0.5 + n / 100, False) for n in range(10)]
        assert len(cockpit.rank_for_labelling(items, top_n=3)) == 3

    def test_top_n_zero(self):
        cockpit = ActiveLabellingCockpit()
        assert cockpit.rank_for_labelling([LabelItem("a", 0.5, False)], top_n=0) == []

    def test_negative_top_n_rejected(self):
        with pytest.raises(ValueError, match="top_n must be non-negative"):
            ActiveLabellingCockpit().rank_for_labelling([], top_n=-1)

    def test_tie_break_by_id(self):
        cockpit = ActiveLabellingCockpit(threshold=0.6)
        # Both 0.1 away from threshold; id order decides.
        items = [LabelItem("b", 0.7, False), LabelItem("a", 0.5, False)]
        assert [i.item_id for i in cockpit.rank_for_labelling(items)] == ["a", "b"]


# ── error breakdown ─────────────────────────────────────────────────────


class TestErrorBreakdown:
    def test_breakdown(self):
        cockpit = ActiveLabellingCockpit()
        items = [
            LabelItem("miss", 0.7, True, label="hallucination"),
            LabelItem("false_halt", 0.4, False, label="grounded"),
            LabelItem("ok_pass", 0.9, True, label="grounded"),
            LabelItem("ok_halt", 0.1, False, label="hallucination"),
            LabelItem("unlabelled", 0.5, True),
        ]
        breakdown = cockpit.error_breakdown(items)
        assert breakdown.false_halts == 1
        assert breakdown.missed_hallucinations == 1
        assert breakdown.correct == 2
        assert breakdown.labelled_total == 4

    def test_to_dict(self):
        cockpit = ActiveLabellingCockpit()
        payload = cockpit.error_breakdown(
            [LabelItem("a", 0.9, True, label="grounded")]
        ).to_dict()
        assert payload["correct"] == 1


# ── trade-off curve + recommendation ────────────────────────────────────


class TestTradeoff:
    def _items(self):
        # Grounded answers score high (should pass); hallucinations score low
        # (should halt) — the orientation a healthy scorer produces.
        return [
            LabelItem("g1", 0.8, True, label="grounded"),
            LabelItem("g2", 0.7, True, label="grounded"),
            LabelItem("h1", 0.4, False, label="hallucination"),
            LabelItem("h2", 0.3, False, label="hallucination"),
        ]

    def test_curve_has_endpoints(self):
        curve = ActiveLabellingCockpit().tradeoff_curve(self._items())
        thresholds = [p.threshold for p in curve]
        assert thresholds[0] == 0.0
        assert thresholds[-1] == 1.0

    def test_curve_counts_at_zero_and_one(self):
        curve = ActiveLabellingCockpit().tradeoff_curve(self._items())
        at_zero = next(p for p in curve if p.threshold == 0.0)
        at_one = next(p for p in curve if p.threshold == 1.0)
        # t=0: everything approved -> both hallucinations missed, no false halt.
        assert at_zero.missed_hallucinations == 2
        assert at_zero.false_halts == 0
        # t=1: everything halted -> both grounded false-halted, no miss.
        assert at_one.false_halts == 2
        assert at_one.missed_hallucinations == 0

    def test_recommend_separable_threshold(self):
        # Grounded scores <0.5, hallucination >0.5 — a threshold in (0.4, 0.7]
        # separates perfectly with zero error.
        rec = ActiveLabellingCockpit().recommend_threshold(self._items())
        assert rec.point.total_errors == 0
        assert 0.4 < rec.threshold <= 0.7

    def test_recommend_miss_weight_drives_misses_to_zero(self):
        # A heavy miss penalty pushes the recommendation to a threshold with no
        # missed hallucinations.
        items = self._items()
        rec = ActiveLabellingCockpit().recommend_threshold(items, miss_weight=100.0)
        assert rec.point.missed_hallucinations == 0

    def test_recommend_empty_raises(self):
        with pytest.raises(ValueError, match="no labelled items"):
            ActiveLabellingCockpit().recommend_threshold([LabelItem("a", 0.5, True)])

    def test_recommendation_to_dict(self):
        rec = ActiveLabellingCockpit().recommend_threshold(self._items())
        payload = rec.to_dict()
        assert payload["threshold"] == rec.threshold
        assert payload["point"]["total_errors"] == 0
        assert len(payload["curve"]) == len(rec.curve)
        assert "false_halts" in payload["curve"][0]

    def test_domain_filter(self):
        items = [
            LabelItem("f", 0.4, False, domain="finance", label="grounded"),
            LabelItem("m", 0.4, True, domain="medical", label="hallucination"),
        ]
        curve = ActiveLabellingCockpit().tradeoff_curve(items, domain="finance")
        at_one = next(p for p in curve if p.threshold == 1.0)
        # Only the finance grounded item is considered.
        assert at_one.false_halts == 1
        assert at_one.missed_hallucinations == 0


# ── packet export ───────────────────────────────────────────────────────


class TestExport:
    def _labelled(self, n):
        return [
            LabelItem(f"i{idx:02d}", 0.5, True, label="grounded") for idx in range(n)
        ]

    def test_eval_fraction_validation(self):
        with pytest.raises(ValueError, match="eval_fraction"):
            ActiveLabellingCockpit().export_packet([], eval_fraction=1.0)

    def test_split_counts(self):
        packet = ActiveLabellingCockpit().export_packet(
            self._labelled(10), eval_fraction=0.2
        )
        assert packet["counts"]["labelled"] == 10
        assert packet["counts"]["eval"] == 2
        assert packet["counts"]["train"] == 8

    def test_split_deterministic(self):
        items = self._labelled(10)
        cockpit = ActiveLabellingCockpit()
        first = cockpit.export_packet(items)
        second = cockpit.export_packet(items)
        assert first == second

    def test_eval_fraction_zero_all_train(self):
        packet = ActiveLabellingCockpit().export_packet(
            self._labelled(5), eval_fraction=0.0
        )
        assert packet["counts"]["eval"] == 0
        assert packet["counts"]["train"] == 5

    def test_only_labelled_exported(self):
        items = [
            LabelItem("a", 0.5, True, label="grounded"),
            LabelItem("b", 0.5, True),  # unlabelled
        ]
        packet = ActiveLabellingCockpit().export_packet(items)
        assert packet["counts"]["labelled"] == 1


# ── guard integration ───────────────────────────────────────────────────


class TestGuardIntegration:
    def test_cockpit_property(self):
        from director_ai.guard import ProductionGuard

        guard = ProductionGuard()
        cockpit = guard.labelling_cockpit
        assert isinstance(cockpit, ActiveLabellingCockpit)
        assert guard.labelling_cockpit is cockpit
        assert cockpit.threshold == guard.config.coherence_threshold
