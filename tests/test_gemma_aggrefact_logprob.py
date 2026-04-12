# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.gemma_aggrefact_logprob``.

Covers:

* ``compute_balanced_accuracy()`` — threshold parameter, None scores skipped.
* ``sweep_threshold()`` — finds optimal threshold.
* ``per_dataset_sweep()`` — per-dataset thresholds + average BA.
* ``main()`` — mocked LlamaCppLogprobBackend, JSON schema with logprob fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("_ensure_datasets_stub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from gemma_aggrefact_logprob import (  # noqa: E402
    compute_balanced_accuracy,
    per_dataset_sweep,
    sweep_threshold,
)

# ── compute_balanced_accuracy ───────────────────────────────────────────


class TestComputeBalancedAccuracy:
    def test_perfect_at_default_threshold(self):
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels) == 1.0

    def test_all_wrong(self):
        scores = [0.1, 0.9, 0.2, 0.8]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels) == 0.0

    def test_none_scores_skipped(self):
        scores = [0.9, None, None, 0.1]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels) == 1.0

    def test_custom_threshold(self):
        scores = [0.6, 0.4, 0.6, 0.4]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels, threshold=0.5) == 1.0
        # At threshold 0.7: all predicted 0 → BA=0.5
        assert compute_balanced_accuracy(scores, labels, threshold=0.7) == 0.5


# ── sweep_threshold ─────────────────────────────────────────────────────


class TestSweepThreshold:
    def test_finds_optimal(self):
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        t, ba = sweep_threshold(scores, labels)
        assert ba == 1.0
        assert 0.0 < t < 1.0

    def test_random_scores(self):
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0]
        t, ba = sweep_threshold(scores, labels)
        # All same score → BA 0.5 at any threshold
        assert ba == 0.5


# ── per_dataset_sweep ───────────────────────────────────────────────────


class TestPerDatasetSweep:
    def test_per_dataset_thresholds(self):
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        datasets = ["a", "a", "b", "b"]
        result, avg = per_dataset_sweep(scores, labels, datasets)
        assert "a" in result
        assert "b" in result
        assert result["a"]["balanced_accuracy"] == 1.0
        assert result["b"]["balanced_accuracy"] == 1.0
        assert avg == 1.0

    def test_samples_count(self):
        result, _ = per_dataset_sweep([0.5, 0.5, 0.5], [1, 0, 1], ["a", "a", "b"])
        assert result["a"]["samples"] == 2
        assert result["b"]["samples"] == 1


# ── main() ──────────────────────────────────────────────────────────────


class MockDataset:
    def __init__(self, rows):
        self._rows = rows

    def select(self, indices):
        return MockDataset([self._rows[i] for i in indices])

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


def _toy_dataset():
    return MockDataset(
        [
            {"doc": "ctx", "claim": "c1", "label": 1, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c2", "label": 0, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c3", "label": 1, "dataset": "RAGTruth"},
            {"doc": "ctx", "claim": "c4", "label": 0, "dataset": "RAGTruth"},
        ]
    )


def _mock_logprob_llm(score=0.9):
    """Mock Llama that returns logprobs for SUPPORTED."""
    import math

    mock = MagicMock()
    mock.tokenize.return_value = [100]  # fake token id
    mock.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"content": "SUPPORTED"},
                "logprobs": {
                    "content": [
                        {
                            "top_logprobs": [
                                {"token": "SUPPORTED", "logprob": math.log(score)},
                                {"token": "NOT", "logprob": math.log(1 - score)},
                            ],
                        }
                    ],
                },
            }
        ],
    }
    return mock


class TestMainCli:
    def _run_main(self, tmp_path, score=0.9):
        out_file = tmp_path / "logprob_result.json"
        mock = _mock_logprob_llm(score)

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model",
                    "/fake.gguf",
                    "--max-samples",
                    "4",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "2",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            from gemma_aggrefact_logprob import main

            main()

        return json.loads(out_file.read_text())

    def test_schema_completeness(self, tmp_path):
        r = self._run_main(tmp_path)
        for key in (
            "model",
            "samples",
            "global_balanced_accuracy_t05",
            "global_balanced_accuracy_optimal",
            "global_optimal_threshold",
            "per_dataset_avg_balanced_accuracy",
            "per_dataset",
            "invalid_scores",
            "scores",
            "labels",
            "datasets",
        ):
            assert key in r, f"missing {key!r}"

    def test_scores_are_continuous(self, tmp_path):
        r = self._run_main(tmp_path, score=0.8)
        # All scores should be ~0.8 (from logprob mock)
        for s in r["scores"]:
            assert 0.7 < s < 0.9

    def test_high_score_ba(self, tmp_path):
        r = self._run_main(tmp_path, score=0.9)
        # All scores ~0.9 → all pred=1 at t=0.5 → BA=0.5
        assert r["global_balanced_accuracy_t05"] == 0.5

    def test_invalid_scores_zero_on_success(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["invalid_scores"] == 0

    def test_per_dataset_has_threshold(self, tmp_path):
        r = self._run_main(tmp_path)
        for ds_metrics in r["per_dataset"].values():
            assert "threshold" in ds_metrics
            assert "balanced_accuracy" in ds_metrics
