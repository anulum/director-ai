# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Tuner Tests (STRONG)
"""Multi-angle tests for threshold tuner pipeline.

Covers: TuneResult structure, balanced accuracy, empty samples guard,
single threshold, CLI execution, output file, empty/malformed JSONL,
parametrised threshold ranges, pipeline integration, and performance.
"""

from __future__ import annotations

import json
import tempfile

import pytest

from director_ai.core.tuner import TuneResult, tune


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


class TestTuneCLI:
    def test_cli_runs(self):
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
