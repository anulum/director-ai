# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multi-scale alignment tests

"""Multi-angle coverage: ValueVector + Action validation,
ValueLatticeScorer logistic semantics, HierarchicalAligner
composite + failing-scale detection, ScaleConflictDetector
conformal calibration + threshold-bound detection, protocol
runtime-check."""

from __future__ import annotations

from typing import Any, cast

import pytest

from director_ai.core.multi_scale_alignment import (
    Action,
    AlignmentReport,
    HierarchicalAligner,
    ScaleConflictDetector,
    ScaleScorer,
    ScaleScoreTable,
    ValueLatticeScorer,
    ValueVector,
)

# --- ValueVector ---------------------------------------------------


class TestValueVector:
    def test_valid(self):
        vv = ValueVector(weights={"safety": 0.8, "autonomy": 0.3})
        assert vv.weights["safety"] == 0.8

    def test_empty_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            ValueVector(weights={})

    def test_weight_out_of_range(self):
        with pytest.raises(ValueError, match="weight"):
            ValueVector(weights={"safety": 1.5})

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            ValueVector(weights={"": 0.5})


# --- Action --------------------------------------------------------


class TestAction:
    def test_valid(self):
        a = Action(label="deploy", impacts={"safety": -0.5, "speed": 0.8})
        assert a.label == "deploy"

    def test_empty_label_rejected(self):
        with pytest.raises(ValueError, match="label"):
            Action(label="")

    def test_impact_out_of_range(self):
        with pytest.raises(ValueError, match="impact"):
            Action(label="x", impacts={"safety": 2.0})

    def test_empty_impact_name(self):
        with pytest.raises(ValueError, match="value name"):
            Action(label="x", impacts={"": 0.5})


# --- ValueLatticeScorer -------------------------------------------


class TestValueLatticeScorer:
    def test_positive_affinity_scores_high(self):
        scorer = ValueLatticeScorer(
            scale="agent",
            values=ValueVector(weights={"safety": 1.0}),
        )
        high = scorer.score(Action(label="safe_action", impacts={"safety": 0.9}))
        low = scorer.score(
            Action(label="unsafe_action", impacts={"safety": -0.9})
        )
        assert high > 0.8
        assert low < 0.2

    def test_unknown_value_ignored(self):
        scorer = ValueLatticeScorer(
            scale="swarm",
            values=ValueVector(weights={"safety": 1.0}),
        )
        # Action advances a value the scale does not track.
        score = scorer.score(
            Action(label="x", impacts={"autonomy": 1.0})
        )
        # Affinity is zero → logistic(0) = 0.5.
        assert score == pytest.approx(0.5)

    def test_bad_scale(self):
        bad = cast(Any, "multiverse")
        with pytest.raises(ValueError, match="scale"):
            ValueLatticeScorer(scale=bad, values=ValueVector(weights={"x": 0.5}))

    def test_bad_steepness(self):
        with pytest.raises(ValueError, match="steepness"):
            ValueLatticeScorer(
                scale="agent",
                values=ValueVector(weights={"x": 0.5}),
                steepness=0.0,
            )

    def test_steepness_affects_sharpness(self):
        flat = ValueLatticeScorer(
            scale="agent",
            values=ValueVector(weights={"safety": 1.0}),
            steepness=1.0,
        )
        sharp = ValueLatticeScorer(
            scale="agent",
            values=ValueVector(weights={"safety": 1.0}),
            steepness=8.0,
        )
        act = Action(label="x", impacts={"safety": 0.5})
        assert sharp.score(act) > flat.score(act)

    def test_protocol_runtime_check(self):
        scorer = ValueLatticeScorer(
            scale="agent", values=ValueVector(weights={"x": 0.5})
        )
        assert isinstance(scorer, ScaleScorer)


# --- HierarchicalAligner ------------------------------------------


def _four_scorers() -> list[ScaleScorer]:
    return [
        ValueLatticeScorer(
            scale="agent",
            values=ValueVector(weights={"safety": 0.9, "autonomy": 0.2}),
        ),
        ValueLatticeScorer(
            scale="swarm",
            values=ValueVector(weights={"safety": 0.7, "transparency": 0.4}),
        ),
        ValueLatticeScorer(
            scale="org",
            values=ValueVector(weights={"compliance": 0.9, "reputation": 0.6}),
        ),
        ValueLatticeScorer(
            scale="planetary",
            values=ValueVector(
                weights={"sustainability": 0.8, "equity": 0.7}
            ),
        ),
    ]


class TestAligner:
    def test_aligned_action_all_scales_pass(self):
        aligner = HierarchicalAligner(scorers=_four_scorers())
        action = Action(
            label="good",
            impacts={
                "safety": 0.9,
                "autonomy": 0.4,
                "transparency": 0.8,
                "compliance": 0.9,
                "reputation": 0.7,
                "sustainability": 0.8,
                "equity": 0.6,
            },
        )
        report = aligner.evaluate(action)
        assert isinstance(report, AlignmentReport)
        assert report.aligned
        assert report.failing_scales == ()
        assert report.composite > 0.7

    def test_failing_scale_detected(self):
        aligner = HierarchicalAligner(
            scorers=_four_scorers(), allow_threshold=0.6
        )
        action = Action(
            label="bad_at_org",
            impacts={
                "safety": 0.9,
                "autonomy": 0.4,
                "transparency": 0.8,
                # Org values violated.
                "compliance": -0.9,
                "reputation": -0.8,
                "sustainability": 0.8,
                "equity": 0.6,
            },
        )
        report = aligner.evaluate(action)
        assert "org" in report.failing_scales
        assert not report.aligned

    def test_scores_in_unit_interval(self):
        aligner = HierarchicalAligner(scorers=_four_scorers())
        action = Action(label="x", impacts={})
        report = aligner.evaluate(action)
        for score in report.table.scores.values():
            assert 0.0 <= score <= 1.0

    def test_table_iteration_order(self):
        aligner = HierarchicalAligner(scorers=_four_scorers())
        action = Action(label="x", impacts={})
        ordered = aligner.evaluate(action).table.ordered()
        assert [s for s, _ in ordered] == ["agent", "swarm", "org", "planetary"]

    def test_custom_weights(self):
        scorers = _four_scorers()
        aligner = HierarchicalAligner(
            scorers=scorers,
            weights={"agent": 1.0, "swarm": 0.0, "org": 0.0, "planetary": 0.0},
        )
        # With all weight on agent, composite == agent score.
        action = Action(label="x", impacts={"safety": 1.0})
        report = aligner.evaluate(action)
        assert report.composite == pytest.approx(report.table["agent"])

    def test_duplicate_scorer_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            HierarchicalAligner(
                scorers=[
                    ValueLatticeScorer(
                        scale="agent",
                        values=ValueVector(weights={"x": 0.5}),
                    ),
                    ValueLatticeScorer(
                        scale="agent",
                        values=ValueVector(weights={"y": 0.5}),
                    ),
                ]
            )

    def test_empty_scorers_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            HierarchicalAligner(scorers=[])

    def test_bad_allow_threshold(self):
        with pytest.raises(ValueError, match="allow_threshold"):
            HierarchicalAligner(scorers=_four_scorers(), allow_threshold=1.5)

    def test_unknown_scale_in_weights(self):
        bogus_weights = cast(Any, {"cosmic": 1.0})
        with pytest.raises(ValueError, match="unknown scale"):
            HierarchicalAligner(
                scorers=_four_scorers(),
                weights=bogus_weights,
            )

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            HierarchicalAligner(
                scorers=_four_scorers(), weights={"agent": -0.1}
            )

    def test_zero_sum_weights_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            HierarchicalAligner(
                scorers=_four_scorers(),
                weights={
                    "agent": 0.0,
                    "swarm": 0.0,
                    "org": 0.0,
                    "planetary": 0.0,
                },
            )

    def test_scales_property_sorted(self):
        aligner = HierarchicalAligner(scorers=_four_scorers())
        assert aligner.scales == ("agent", "swarm", "org", "planetary")


# --- ScaleConflictDetector ----------------------------------------


class TestConflictDetector:
    def test_calibrate_then_detect(self):
        # Calibration set: four aligned actions with small deltas.
        calibration = [
            ScaleScoreTable(scores={"agent": 0.80, "swarm": 0.82, "org": 0.85, "planetary": 0.83}),
            ScaleScoreTable(scores={"agent": 0.70, "swarm": 0.72, "org": 0.75, "planetary": 0.73}),
            ScaleScoreTable(scores={"agent": 0.60, "swarm": 0.63, "org": 0.65, "planetary": 0.64}),
        ]
        detector = ScaleConflictDetector(target_coverage=0.9)
        threshold = detector.calibrate(calibration)
        assert 0.0 < threshold < 1.0
        # Observed: agent+swarm low, org+planetary high → conflict.
        observed = ScaleScoreTable(
            scores={"agent": 0.10, "swarm": 0.15, "org": 0.90, "planetary": 0.88}
        )
        conflicts = detector.detect(observed)
        # Org vs agent: 0.80 delta; way above threshold.
        pair_set = {c.scales for c in conflicts}
        assert ("org", "agent") in pair_set

    def test_detect_requires_calibration(self):
        detector = ScaleConflictDetector()
        table = ScaleScoreTable(scores={"agent": 0.5, "swarm": 0.6})
        with pytest.raises(ValueError, match="calibrate"):
            detector.detect(table)

    def test_bad_coverage(self):
        with pytest.raises(ValueError, match="target_coverage"):
            ScaleConflictDetector(target_coverage=1.2)

    def test_calibrate_rejects_insufficient_data(self):
        detector = ScaleConflictDetector()
        only_one = [ScaleScoreTable(scores={"agent": 0.5})]
        with pytest.raises(ValueError, match="pairwise"):
            detector.calibrate(only_one)

    def test_no_conflict_when_within_threshold(self):
        calibration = [
            ScaleScoreTable(scores={"agent": 0.80, "swarm": 0.82, "org": 0.85, "planetary": 0.83}),
            ScaleScoreTable(scores={"agent": 0.70, "swarm": 0.72, "org": 0.75, "planetary": 0.73}),
            ScaleScoreTable(scores={"agent": 0.60, "swarm": 0.63, "org": 0.65, "planetary": 0.64}),
        ]
        detector = ScaleConflictDetector(target_coverage=0.9)
        detector.calibrate(calibration)
        quiet = ScaleScoreTable(
            scores={"agent": 0.80, "swarm": 0.82, "org": 0.85, "planetary": 0.84}
        )
        assert detector.detect(quiet) == ()

    def test_severe_flag(self):
        calibration = [
            ScaleScoreTable(scores={"agent": 0.5, "swarm": 0.52, "org": 0.5, "planetary": 0.51}),
            ScaleScoreTable(scores={"agent": 0.5, "swarm": 0.50, "org": 0.51, "planetary": 0.52}),
        ]
        detector = ScaleConflictDetector(target_coverage=0.9)
        detector.calibrate(calibration)
        conflicting = ScaleScoreTable(
            scores={"agent": 0.05, "swarm": 0.10, "org": 0.95, "planetary": 0.95}
        )
        conflicts = detector.detect(conflicting)
        assert any(c.is_severe for c in conflicts)

    def test_wider_first_order(self):
        calibration = [
            ScaleScoreTable(scores={"agent": 0.5, "org": 0.52}),
            ScaleScoreTable(scores={"agent": 0.5, "org": 0.51}),
        ]
        detector = ScaleConflictDetector(target_coverage=0.5)
        detector.calibrate(calibration)
        observed = ScaleScoreTable(scores={"agent": 0.1, "org": 0.9})
        conflicts = detector.detect(observed)
        # Wider scope first.
        assert conflicts[0].scales == ("org", "agent")
