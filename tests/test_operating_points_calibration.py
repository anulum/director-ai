# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Matched-FPR operating-point calibration contracts (WCS-2a).

Pins the order-statistic threshold rule to the WCS-1 sweep semantics
(``benchmarks/run_longcontext_bench.py``), the sample bucketing through the
scorer calibration surface, and the config/env overlay rendering.
"""

from __future__ import annotations

import pytest

from director_ai.core.calibration.operating_points import (
    DEFAULT_TARGET_FPR,
    OperatingPoint,
    calibrate_from_samples,
    calibrate_operating_point,
    config_overlay,
    format_env_overlay,
    matched_fpr_support_threshold,
)


class TestMatchedFprThreshold:
    def test_threshold_is_the_allowed_order_statistic(self):
        good = [0.5, 0.9, 0.02, 0.7, 0.008, 0.85, 0.6, 0.95, 0.3, 0.4]
        # 10 goods at target 0.1 allow one false positive: the threshold
        # is the second-smallest good support, flagging only 0.008.
        assert matched_fpr_support_threshold(good, 0.1) == pytest.approx(0.02)

    def test_zero_target_flags_no_good_sample(self):
        good = [0.5, 0.03, 0.7]
        threshold = matched_fpr_support_threshold(good, 0.0)
        assert threshold == pytest.approx(0.03)
        assert sum(1 for s in good if s < threshold) == 0

    def test_matches_the_wcs1_sweep_rule_at_the_dialogue_target(self):
        # The sweep's rule: sorted(good)[int(target * n)]. 200 goods at
        # the dialogue target 0.045 allow ⌊9⌋ false positives.
        good = [i / 200.0 for i in range(200)]
        assert matched_fpr_support_threshold(good, 0.045) == pytest.approx(
            sorted(good)[9]
        )

    def test_empty_goods_raise(self):
        with pytest.raises(ValueError, match="needs good supports"):
            matched_fpr_support_threshold([], 0.05)

    @pytest.mark.parametrize("target", [-0.1, 1.0, 1.5])
    def test_out_of_range_target_raises(self, target):
        with pytest.raises(ValueError, match="target_fpr"):
            matched_fpr_support_threshold([0.5], target)


class TestCalibrateOperatingPoint:
    def test_point_reports_actual_fpr_and_catch(self):
        good = [0.5, 0.9, 0.02, 0.7, 0.008, 0.85, 0.6, 0.95, 0.3, 0.4]
        point = calibrate_operating_point(
            "dialogue", good, [0.001, 0.5, 0.005], target_fpr=0.1
        )
        assert point.task == "dialogue"
        assert point.support_threshold == pytest.approx(0.02)
        assert point.actual_fpr == pytest.approx(0.1)
        assert point.actual_fpr <= point.target_fpr
        assert point.catch_rate == pytest.approx(2 / 3)
        assert point.n_good == 10
        assert point.n_bad == 3

    def test_point_without_bad_samples_has_no_catch_rate(self):
        point = calibrate_operating_point("dialogue", [0.5, 0.7], target_fpr=0.0)
        assert point.catch_rate is None
        assert point.n_bad == 0


class _CalibrationScorer:
    """Scorer double: task from the prompt, support from the response."""

    def raw_task_support(self, prompt, response):
        task = prompt.split(":", 1)[0]
        return task, float(response)


class TestCalibrateFromSamples:
    def test_buckets_by_detected_task_and_ignores_composite_routes(self):
        samples = [
            ("dialogue: a", "0.9", False),
            ("dialogue: b", "0.5", False),
            ("dialogue: c", "0.01", True),
            ("summarization: d", "0.8", False),
            ("summarization: e", "0.02", True),
            ("qa: f", "0.9", False),
        ]
        points = calibrate_from_samples(_CalibrationScorer(), samples)
        assert [p.task for p in points] == ["dialogue", "summarization"]
        dialogue = points[0]
        assert dialogue.n_good == 2
        assert dialogue.n_bad == 1
        assert dialogue.target_fpr == DEFAULT_TARGET_FPR["dialogue"]
        assert points[1].target_fpr == DEFAULT_TARGET_FPR["summarization"]

    def test_task_without_good_samples_is_skipped(self):
        samples = [("dialogue: only-bad", "0.1", True)]
        assert calibrate_from_samples(_CalibrationScorer(), samples) == []

    def test_target_fpr_overrides_apply_per_task(self):
        samples = [
            ("dialogue: a", "0.9", False),
            ("dialogue: b", "0.5", False),
        ]
        points = calibrate_from_samples(
            _CalibrationScorer(),
            samples,
            target_fpr_by_task={"dialogue": 0.5},
        )
        assert points[0].target_fpr == 0.5
        assert points[0].support_threshold == pytest.approx(0.9)

    def test_default_targets_pin_the_tracked_baseline_rates(self):
        # judge_bench_nli_only_200.json: dialogue FPR 4.5 %, summ 2.5 %.
        assert DEFAULT_TARGET_FPR == {"dialogue": 0.045, "summarization": 0.025}


class TestOverlayRendering:
    def _points(self):
        return [
            OperatingPoint(
                task="dialogue",
                support_threshold=0.0090618394,
                target_fpr=0.045,
                actual_fpr=0.045,
                catch_rate=0.27,
                n_good=200,
                n_bad=200,
            ),
            OperatingPoint(
                task="summarization",
                support_threshold=0.0402165,
                target_fpr=0.025,
                actual_fpr=0.025,
                catch_rate=0.125,
                n_good=200,
                n_bad=200,
            ),
        ]

    def test_config_overlay_enables_the_calibrated_routes(self):
        overlay = config_overlay(self._points())
        assert overlay == {
            "nli_dialogue_support_threshold": 0.009062,
            "nli_dialogue_scoring": "raw_support",
            "nli_summarization_support_threshold": 0.040217,
            "nli_summarization_aggregation": "weakest_link",
        }

    def test_env_overlay_renders_director_variables(self):
        lines = format_env_overlay(self._points()).splitlines()
        assert "DIRECTOR_NLI_DIALOGUE_SUPPORT_THRESHOLD=0.009062" in lines
        assert "DIRECTOR_NLI_DIALOGUE_SCORING=raw_support" in lines
        assert "DIRECTOR_NLI_SUMMARIZATION_AGGREGATION=weakest_link" in lines

    def test_empty_points_render_empty_overlays(self):
        assert config_overlay([]) == {}
        assert format_env_overlay([]) == ""
