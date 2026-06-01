# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tuner Tests
"""Multi-angle tests for threshold tuner pipeline.

Covers: TuneResult structure, balanced accuracy, empty samples guard,
single threshold, CLI execution, output file, empty/malformed JSONL,
parametrised threshold ranges, pipeline integration, and performance.
"""

from __future__ import annotations

import json
import tempfile

import pytest

import director_ai.core.training.tuner as tuner_module
from director_ai.core.tuner import (
    BoundaryExample,
    ThresholdCandidate,
    TuneResult,
    format_confidence_report,
    format_profile_overlay,
    tune,
)


def _synthetic_samples():
    correct = [
        {"prompt": "sky color?", "response": "The sky is blue.", "label": True},
        {"prompt": "water wet?", "response": "Yes, water is wet.", "label": True},
        {"prompt": "2+2?", "response": "2+2 is 4.", "label": True},
        {"prompt": "fire hot?", "response": "Yes, fire is hot.", "label": True},
        {"prompt": "ice cold?", "response": "Yes, ice is cold.", "label": True},
    ]
    incorrect = [
        {"prompt": "sky color?", "response": "Mars has rings.", "label": False},
        {"prompt": "water wet?", "response": "Dolphins can fly.", "label": False},
        {"prompt": "2+2?", "response": "Purple elephants.", "label": False},
        {"prompt": "fire hot?", "response": "Snow is a fruit.", "label": False},
        {"prompt": "ice cold?", "response": "Moon is cheese.", "label": False},
    ]
    return correct + incorrect


class TestTuner:
    def test_returns_tune_result(self):
        result = tune(_synthetic_samples())
        assert isinstance(result, TuneResult)
        assert 0.30 <= result.threshold <= 0.90
        assert result.samples == 10

    def test_balanced_accuracy_above_chance(self):
        result = tune(_synthetic_samples())
        assert result.balanced_accuracy > 0.5

    def test_empty_samples_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            tune([])

    def test_single_threshold(self):
        result = tune(
            _synthetic_samples(),
            thresholds=[0.5],
            weight_pairs=[(0.6, 0.4)],
        )
        assert result.threshold == 0.5
        assert result.w_logic == 0.6

    def test_result_profile_overlay(self):
        result = tune(
            _synthetic_samples(),
            thresholds=[0.5],
            weight_pairs=[(0.6, 0.4)],
        )
        overlay = result.to_profile_overlay(
            profile="medical_tuned",
            base_profile="medical",
        )
        assert overlay["profile"] == "medical_tuned"
        assert overlay["coherence_threshold"] == 0.5
        assert overlay["hard_limit"] == 0.4
        assert overlay["soft_limit"] == 0.6
        assert overlay["w_logic"] == 0.6
        assert overlay["w_fact"] == 0.4
        assert overlay["extra"]["tuned_from_profile"] == "medical"
        assert overlay["extra"]["tune_confidence_level"] in {"low", "medium", "high"}
        assert "tune_confidence_intervals" in overlay["extra"]
        assert "balanced_accuracy=" in overlay["extra"]["tune_confidence_intervals"]
        assert "tune_confusion_matrix" in overlay["extra"]
        assert "tune_tradeoff_summary" in overlay["extra"]

    def test_format_profile_overlay_yaml(self):
        result = tune(
            _synthetic_samples(),
            thresholds=[0.5],
            weight_pairs=[(0.6, 0.4)],
        )
        content = format_profile_overlay(
            result,
            profile="finance_tuned",
            base_profile="finance",
        )
        assert 'profile: "finance_tuned"' in content
        assert "coherence_threshold: 0.5" in content
        assert 'tuned_from_profile: "finance"' in content
        assert "# Confidence report:" in content
        assert "tune_confidence_level" in content

    def test_confidence_report_contains_tradeoffs_and_boundary_examples(self):
        result = tune(
            _synthetic_samples(),
            thresholds=[0.45, 0.5, 0.55],
            weight_pairs=[(0.6, 0.4), (0.5, 0.5)],
        )
        report = format_confidence_report(result)

        assert result.confidence_level in {"low", "medium", "high"}
        assert result.selection_margin >= 0.0
        assert result.positive_samples == 5
        assert result.negative_samples == 5
        assert result.evaluated_candidates
        assert result.confidence_intervals["balanced_accuracy"][0] <= (
            result.balanced_accuracy
        )
        assert result.confidence_intervals["balanced_accuracy"][1] >= (
            result.balanced_accuracy
        )
        assert isinstance(result.evaluated_candidates[0], ThresholdCandidate)
        assert result.boundary_examples
        assert isinstance(result.boundary_examples[0], BoundaryExample)
        assert "Trade-off" in report
        assert "95% Wilson intervals" in report
        assert "Boundary examples" in report
        assert "flip_threshold" in report


class TestTunerInternals:
    def test_legacy_profile_overlay_and_empty_report_paths(self):
        class LegacyResult:
            threshold = 0.95
            w_logic = 0.7
            w_fact = 0.3
            balanced_accuracy = 0.75
            precision = 0.8
            recall = 0.6
            f1 = 0.6857
            samples = 12

        overlay = tuner_module._to_profile_overlay(
            LegacyResult(),
            profile="legacy_tuned",
            base_profile="legacy",
        )

        assert overlay["hard_limit"] == 0.85
        assert overlay["soft_limit"] == 1.0
        assert overlay["extra"]["tuned_from_profile"] == "legacy"
        assert overlay["extra"]["tune_confidence_level"] == "low"
        assert overlay["extra"]["tune_confidence_intervals"] == ""

        rendered = format_profile_overlay(
            LegacyResult(),
            profile="legacy_tuned",
            base_profile="",
        )
        assert 'profile: "legacy_tuned"' in rendered
        assert "# Confidence report:" in rendered

    def test_profile_overlay_rejects_non_mapping_extra(self):
        class BadOverlayResult:
            threshold = 0.5
            w_logic = 0.6
            w_fact = 0.4
            balanced_accuracy = 0.75
            precision = 0.8
            recall = 0.6
            f1 = 0.6857
            samples = 12

            def to_profile_overlay(self, *, profile: str, base_profile: str):
                return {"profile": profile, "extra": "not-a-mapping"}

        with pytest.raises(TypeError, match="extra metadata"):
            format_profile_overlay(BadOverlayResult())

    def test_candidate_metrics_cover_all_confusion_branches(self):
        scores = [
            tuner_module._ScoredSample(0, 0.8, True, "p0", "r0"),
            tuner_module._ScoredSample(1, 0.2, True, "p1", "r1"),
            tuner_module._ScoredSample(2, 0.7, False, "p2", "r2"),
            tuner_module._ScoredSample(3, 0.1, False, "p3", "r3"),
        ]

        candidate = tuner_module._evaluate_candidate(0.5, 0.6, 0.4, scores)

        assert candidate.tp == 1
        assert candidate.fn == 1
        assert candidate.fp == 1
        assert candidate.tn == 1
        assert candidate.precision == 0.5
        assert candidate.false_positive_rate == 0.5
        assert candidate.false_negative_rate == 0.5

    def test_candidate_metrics_handle_empty_denominators(self):
        positives_only = [
            tuner_module._ScoredSample(0, 0.1, True, "p", "r"),
        ]
        negatives_only = [
            tuner_module._ScoredSample(0, 0.9, False, "p", "r"),
        ]

        positive_candidate = tuner_module._evaluate_candidate(
            0.5,
            0.6,
            0.4,
            positives_only,
        )
        negative_candidate = tuner_module._evaluate_candidate(
            0.5,
            0.6,
            0.4,
            negatives_only,
        )

        assert positive_candidate.precision == 0.0
        assert positive_candidate.f1 == 0.0
        assert positive_candidate.false_positive_rate == 0.0
        assert negative_candidate.recall == 0.0
        assert negative_candidate.false_negative_rate == 0.0

    def test_confidence_levels_and_tradeoff_descriptions(self):
        assert (
            tuner_module._confidence_level(
                samples=50,
                positives=20,
                negatives=30,
                selection_margin=0.05,
            )
            == "high"
        )
        assert (
            tuner_module._confidence_level(
                samples=20,
                positives=3,
                negatives=17,
                selection_margin=0.02,
            )
            == "medium"
        )
        assert (
            tuner_module._confidence_level(
                samples=20,
                positives=1,
                negatives=19,
                selection_margin=0.02,
            )
            == "low"
        )

        fp_heavy = TuneResult(
            threshold=0.5,
            w_logic=0.6,
            w_fact=0.4,
            balanced_accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            samples=20,
            tp=8,
            fp=8,
            tn=2,
            fn=2,
        )
        fn_heavy = TuneResult(
            threshold=0.5,
            w_logic=0.6,
            w_fact=0.4,
            balanced_accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            samples=20,
            tp=2,
            fp=2,
            tn=8,
            fn=8,
        )

        assert "catching misses" in tuner_module._tradeoff_summary(fp_heavy)
        assert "fewer clean-response rejections" in tuner_module._tradeoff_summary(
            fn_heavy
        )

    def test_boundary_examples_excerpt_and_yaml_scalars(self):
        samples = [
            tuner_module._ScoredSample(
                index,
                score,
                bool(index % 2),
                f"prompt {index} " * 20,
                f"response {index} " * 20,
            )
            for index, score in enumerate([0.49, 0.51, 0.7, 0.2, 0.6])
        ]

        examples = tuner_module._boundary_examples(0.5, samples)

        assert len(examples) == 3
        assert examples[0].counterfactual_threshold == 0.49
        assert examples[0].prompt_excerpt.endswith("...")
        assert tuner_module._excerpt("short text") == "short text"
        assert tuner_module._yaml_scalar(True) == "true"
        assert tuner_module._yaml_scalar(3) == "3"
        assert tuner_module._yaml_scalar('a "quoted" path\\name') == (
            '"a \\"quoted\\" path\\\\name"'
        )

    def test_sum_int_python_and_rust_paths(self, monkeypatch):
        monkeypatch.setattr(tuner_module, "_RUST_TUNER", False)
        assert tuner_module._sum_int([1, 2, 3]) == 6

        calls = []

        def fake_rust_sum_i64(values):
            calls.append(values)
            return 10

        monkeypatch.setattr(tuner_module, "_RUST_TUNER", True)
        monkeypatch.setattr(tuner_module, "rust_sum_i64", fake_rust_sum_i64)

        assert tuner_module._sum_int([4, 6]) == 10
        assert calls == [[4, 6]]


class TestTuneCLI:
    def test_cli_runs(self, capsys):
        from director_ai.cli import main

        samples = _synthetic_samples()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            path = f.name

        main(["tune", path])
        captured = capsys.readouterr()
        assert "Confidence report" in captured.out
        assert "Selected threshold" in captured.out

    def test_cli_output_file(self):
        from director_ai.cli import main

        samples = _synthetic_samples()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            inpath = f.name

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as out:
            outpath = out.name

        main(["tune", inpath, "--output", outpath])
        with open(outpath, encoding="utf-8") as f:
            content = f.read()
        assert "coherence_threshold" in content
        assert "hard_limit" in content
        assert "extra:" in content

    def test_cli_dataset_profile_overlay(self):
        from director_ai.cli import main

        samples = _synthetic_samples()
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            inpath = f.name

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as out:
            outpath = out.name

        main(["tune", "--dataset", inpath, "--profile", "medical", "--output", outpath])
        with open(outpath, encoding="utf-8") as f:
            content = f.read()
        assert 'profile: "medical_tuned"' in content
        assert 'tuned_from_profile: "medical"' in content

    def test_cli_empty_file(self):
        from director_ai.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("")
            path = f.name

        with pytest.raises(SystemExit):
            main(["tune", path])

    def test_cli_malformed_jsonl(self, capsys):
        from director_ai.cli import main

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".jsonl",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("not json\n")
            f.write('{"prompt": "a", "response": "b", "label": true}\n')
            path = f.name

        main(["tune", path])
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "Best threshold" in captured.out


class TestTunerParametrised:
    """Parametrised tuner tests."""

    @pytest.mark.parametrize("n_thresholds", [1, 3, 5])
    def test_various_threshold_counts(self, n_thresholds):
        import numpy as np

        thresholds = np.linspace(0.3, 0.8, n_thresholds).tolist()
        result = tune(_synthetic_samples(), thresholds=thresholds)
        assert result.threshold in thresholds

    @pytest.mark.parametrize(
        "w_logic,w_fact",
        [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)],
    )
    def test_various_weight_pairs(self, w_logic, w_fact):
        result = tune(
            _synthetic_samples(),
            weight_pairs=[(w_logic, w_fact)],
        )
        assert result.w_logic == w_logic
        assert result.w_fact == w_fact


class TestTunerPerformanceDoc:
    """Document tuner pipeline performance."""

    def test_tune_result_fields(self):
        result = tune(_synthetic_samples())
        assert hasattr(result, "threshold")
        assert hasattr(result, "w_logic")
        assert hasattr(result, "w_fact")
        assert hasattr(result, "balanced_accuracy")
        assert hasattr(result, "samples")

    def test_tune_fast(self):
        import time

        t0 = time.perf_counter()
        tune(_synthetic_samples())
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 5000, f"Tuning took {elapsed_ms:.0f}ms"
