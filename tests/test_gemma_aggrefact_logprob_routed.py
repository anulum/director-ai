# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Tests for ``benchmarks.gemma_aggrefact_logprob_routed``."""

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

_routed_logprob_module = importlib.import_module("gemma_aggrefact_logprob_routed")


class BalancedAccuracy(Protocol):
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


balanced_accuracy = cast(
    BalancedAccuracy,
    _routed_logprob_module.balanced_accuracy,
)
sweep_threshold = cast(SweepThreshold, _routed_logprob_module.sweep_threshold)
per_dataset_sweep = cast(PerDatasetSweep, _routed_logprob_module.per_dataset_sweep)


class TestBalancedAccuracy:
    """Unit guard for balanced-accuracy calculations."""

    def test_perfect_at_default_threshold(self) -> None:
        """Return 1.0 when the default threshold separates all labels."""
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        assert balanced_accuracy(scores, labels) == 1.0

    def test_none_scores_skipped(self) -> None:
        """Skip missing logprob scores during metric calculation."""
        scores: list[float | None] = [0.9, None, None, 0.1]
        labels = [1, 0, 1, 0]
        assert balanced_accuracy(scores, labels) == 1.0

    def test_custom_threshold(self) -> None:
        """Respect caller-supplied decision thresholds."""
        scores = [0.6, 0.4]
        labels = [1, 0]
        assert balanced_accuracy(scores, labels, threshold=0.5) == 1.0
        assert balanced_accuracy(scores, labels, threshold=0.7) == 0.5


class TestSweepThreshold:
    """Unit guard for global threshold sweeps."""

    def test_finds_optimal(self) -> None:
        """Find a threshold that maximises balanced accuracy."""
        scores = [0.9, 0.1, 0.8, 0.2]
        labels = [1, 0, 1, 0]
        threshold, balanced_accuracy_value = sweep_threshold(scores, labels)
        assert balanced_accuracy_value == 1.0
        assert 0.0 < threshold < 1.0


class TestPerDatasetSweep:
    """Unit guard for per-dataset threshold sweeps."""

    def test_returns_per_dataset_metrics(self) -> None:
        """Compute independent thresholds for each dataset subset."""
        result, average = per_dataset_sweep(
            [0.9, 0.1, 0.8, 0.2],
            [1, 0, 1, 0],
            ["a", "a", "b", "b"],
        )
        assert result["a"]["balanced_accuracy"] == 1.0
        assert result["b"]["balanced_accuracy"] == 1.0
        assert average == 1.0


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
    """Return a balanced routed synthetic AggreFact split for unit guards."""
    return MockDataset(
        [
            {"doc": "ctx", "claim": "c1", "label": 1, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c2", "label": 0, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c3", "label": 1, "dataset": "RAGTruth"},
            {"doc": "ctx", "claim": "c4", "label": 0, "dataset": "RAGTruth"},
            {"doc": "ctx", "claim": "c5", "label": 1, "dataset": "Wice"},
            {"doc": "ctx", "claim": "c6", "label": 0, "dataset": "Wice"},
        ]
    )


def _mock_logprob_llm(score: float = 0.9) -> MagicMock:
    """Return a llama-cpp-compatible mock with next-token logprobs."""
    mock = MagicMock()
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
    """Unit guard for the public routed benchmark CLI entry point."""

    def _run_main(self, tmp_path: Path, score: float = 0.9) -> dict[str, object]:
        """Run ``main()`` with mocked optional dependency surfaces."""
        out_file = tmp_path / "routed_logprob_result.json"
        mock = _mock_logprob_llm(score)

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model",
                    "/fake.gguf",
                    "--max-samples",
                    "6",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "2",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            main = cast(Callable[[], None], _routed_logprob_module.main)
            main()

        return cast(dict[str, object], json.loads(out_file.read_text(encoding="utf-8")))

    def test_schema_completeness(self, tmp_path: Path) -> None:
        """Write all public report fields expected by downstream analyzers."""
        report = self._run_main(tmp_path)
        for key in (
            "schema_version",
            "model",
            "method",
            "samples",
            "global_balanced_accuracy_t05",
            "global_balanced_accuracy_optimal",
            "global_optimal_threshold",
            "per_dataset_avg_balanced_accuracy",
            "per_dataset",
            "per_family",
            "dataset_to_family",
            "invalid_scores",
            "scores",
            "labels",
            "datasets_per_sample",
            "families_per_sample",
        ):
            assert key in report, f"missing {key!r}"

    def test_scores_are_continuous(self, tmp_path: Path) -> None:
        """Persist continuous support probabilities from logprob metadata."""
        report = self._run_main(tmp_path, score=0.8)
        scores = cast(list[float], report["scores"])
        for score in scores:
            assert 0.7 < score < 0.9

    def test_per_family_has_thresholds(self, tmp_path: Path) -> None:
        """Include per-family routed optimal thresholds in the JSON payload."""
        report = self._run_main(tmp_path)
        per_family = cast(dict[str, dict[str, object]], report["per_family"])
        assert set(per_family) == {"summ", "rag", "claim"}
        for metrics in per_family.values():
            assert "threshold" in metrics
            assert "balanced_accuracy" in metrics

    def test_method_mentions_routing_and_logprob(self, tmp_path: Path) -> None:
        """Describe both routed prompting and logprob scoring in report metadata."""
        report = self._run_main(tmp_path)
        method = str(report["method"]).lower()
        assert "routing" in method
        assert "logprob" in method

    def test_families_per_sample(self, tmp_path: Path) -> None:
        """Persist the routed prompt family selected for every sample."""
        report = self._run_main(tmp_path)
        assert report["families_per_sample"] == [
            "summ",
            "summ",
            "rag",
            "rag",
            "claim",
            "claim",
        ]

    def test_invalid_scores_zero_on_success(self, tmp_path: Path) -> None:
        """Report zero invalid scores when every model call returns metadata."""
        report = self._run_main(tmp_path)
        assert report["invalid_scores"] == 0

    def test_exception_path_counts_invalid_scores(self, tmp_path: Path) -> None:
        """Convert backend runtime errors into invalid per-sample scores."""
        out_file = tmp_path / "routed_logprob_error.json"
        mock = MagicMock()
        mock.create_chat_completion.side_effect = RuntimeError("fail")

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model",
                    "/fake.gguf",
                    "--max-samples",
                    "6",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "100",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            main = cast(Callable[[], None], _routed_logprob_module.main)
            main()

        report = cast(
            dict[str, object],
            json.loads(out_file.read_text(encoding="utf-8")),
        )
        assert report["invalid_scores"] == 6
