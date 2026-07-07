# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Tests for ``benchmarks.gemma_aggrefact_logprob``."""

from __future__ import annotations

import importlib
import json
import math
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("_ensure_datasets_stub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

_logprob_module = importlib.import_module("gemma_aggrefact_logprob")


class ComputeBalancedAccuracy(Protocol):
    """Callable protocol for the public balanced-accuracy helper."""

    def __call__(
        self,
        scores: Sequence[float | None],
        labels: Sequence[int],
        threshold: float = 0.5,
    ) -> float:
        """Compute balanced accuracy for score/label pairs."""


class SweepThreshold(Protocol):
    """Callable protocol for the public threshold sweep helper."""

    def __call__(
        self,
        scores: Sequence[float | None],
        labels: Sequence[int],
    ) -> tuple[float, float]:
        """Return the best threshold and balanced accuracy."""


class PerDatasetSweep(Protocol):
    """Callable protocol for the public per-dataset sweep helper."""

    def __call__(
        self,
        scores: Sequence[float | None],
        labels: Sequence[int],
        datasets: Sequence[str],
    ) -> tuple[dict[str, dict[str, float | int]], float]:
        """Return per-dataset metrics and their average."""


compute_balanced_accuracy = cast(
    ComputeBalancedAccuracy,
    _logprob_module.compute_balanced_accuracy,
)
sweep_threshold = cast(SweepThreshold, _logprob_module.sweep_threshold)
per_dataset_sweep = cast(PerDatasetSweep, _logprob_module.per_dataset_sweep)


class TestComputeBalancedAccuracy:
    """Unit guard for balanced-accuracy calculations."""

    def test_perfect_at_default_threshold(self) -> None:
        """Return 1.0 when the default threshold separates all labels."""
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels) == 1.0

    def test_all_wrong(self) -> None:
        """Return 0.0 when every thresholded prediction is wrong."""
        scores = [0.1, 0.9, 0.2, 0.8]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels) == 0.0

    def test_none_scores_skipped(self) -> None:
        """Skip missing logprob scores during metric calculation."""
        scores: list[float | None] = [0.9, None, None, 0.1]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels) == 1.0

    def test_custom_threshold(self) -> None:
        """Respect caller-supplied decision thresholds."""
        scores = [0.6, 0.4, 0.6, 0.4]
        labels = [1, 0, 1, 0]
        assert compute_balanced_accuracy(scores, labels, threshold=0.5) == 1.0
        assert compute_balanced_accuracy(scores, labels, threshold=0.7) == 0.5


class TestSweepThreshold:
    """Unit guard for global threshold sweeps."""

    def test_finds_optimal(self) -> None:
        """Find a threshold that maximises balanced accuracy."""
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        threshold, balanced_accuracy = sweep_threshold(scores, labels)
        assert balanced_accuracy == 1.0
        assert 0.0 < threshold < 1.0

    def test_random_scores(self) -> None:
        """Return baseline balanced accuracy when all scores are tied."""
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0]
        _threshold, balanced_accuracy = sweep_threshold(scores, labels)
        assert balanced_accuracy == 0.5


class TestPerDatasetSweep:
    """Unit guard for per-dataset threshold sweeps."""

    def test_per_dataset_thresholds(self) -> None:
        """Compute independent thresholds for each dataset subset."""
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        datasets = ["a", "a", "b", "b"]
        result, average = per_dataset_sweep(scores, labels, datasets)
        assert "a" in result
        assert "b" in result
        assert result["a"]["balanced_accuracy"] == 1.0
        assert result["b"]["balanced_accuracy"] == 1.0
        assert average == 1.0

    def test_samples_count(self) -> None:
        """Preserve represented sample counts in each dataset metric."""
        result, _average = per_dataset_sweep(
            [0.5, 0.5, 0.5],
            [1, 0, 1],
            ["a", "a", "b"],
        )
        assert result["a"]["samples"] == 2
        assert result["b"]["samples"] == 1


class MockDataset:
    """Small in-memory dataset implementing the benchmark's dataset protocol."""

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        """Store rows in deterministic order."""
        self._rows = list(rows)

    def select(self, indices: range) -> MockDataset:
        """Return a subset selected by integer indices."""
        return MockDataset([self._rows[index] for index in indices])

    def __len__(self) -> int:
        """Return the number of rows."""
        return len(self._rows)

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Iterate over stored rows."""
        return iter(self._rows)

    def __getitem__(self, index: int) -> Mapping[str, object]:
        """Return one row by integer index."""
        return self._rows[index]


def _toy_dataset() -> MockDataset:
    """Return a balanced synthetic AggreFact split for unit guards."""
    return MockDataset(
        [
            {"doc": "ctx", "claim": "c1", "label": 1, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c2", "label": 0, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c3", "label": 1, "dataset": "RAGTruth"},
            {"doc": "ctx", "claim": "c4", "label": 0, "dataset": "RAGTruth"},
        ]
    )


def _mock_logprob_llm(score: float = 0.9) -> MagicMock:
    """Return a llama-cpp-compatible mock with next-token logprobs."""
    mock = MagicMock()
    mock.tokenize.return_value = [100]
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
    """Unit guard for the public benchmark CLI entry point."""

    def _run_main(self, tmp_path: Path, score: float = 0.9) -> dict[str, object]:
        """Run ``main()`` with mocked optional dependency surfaces."""
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
            main = cast(Callable[[], None], _logprob_module.main)
            main()

        return cast(dict[str, object], json.loads(out_file.read_text(encoding="utf-8")))

    def test_schema_completeness(self, tmp_path: Path) -> None:
        """Write all public report fields expected by downstream analyzers."""
        report = self._run_main(tmp_path)
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
            assert key in report, f"missing {key!r}"

    def test_scores_are_continuous(self, tmp_path: Path) -> None:
        """Persist continuous support probabilities from logprob metadata."""
        report = self._run_main(tmp_path, score=0.8)
        scores = cast(list[float], report["scores"])
        for score in scores:
            assert 0.7 < score < 0.9

    def test_high_score_ba(self, tmp_path: Path) -> None:
        """Report baseline balanced accuracy when all scores predict support."""
        report = self._run_main(tmp_path, score=0.9)
        assert report["global_balanced_accuracy_t05"] == 0.5

    def test_invalid_scores_zero_on_success(self, tmp_path: Path) -> None:
        """Report zero invalid scores when every model call returns metadata."""
        report = self._run_main(tmp_path)
        assert report["invalid_scores"] == 0

    def test_per_dataset_has_threshold(self, tmp_path: Path) -> None:
        """Include per-dataset optimal thresholds in the JSON payload."""
        report = self._run_main(tmp_path)
        per_dataset = cast(dict[str, dict[str, object]], report["per_dataset"])
        for metrics in per_dataset.values():
            assert "threshold" in metrics
            assert "balanced_accuracy" in metrics
