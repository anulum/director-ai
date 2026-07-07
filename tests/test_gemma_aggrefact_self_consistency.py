# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Tests for ``benchmarks.gemma_aggrefact_self_consistency``."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("_ensure_datasets_stub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

_self_consistency_module = importlib.import_module("gemma_aggrefact_self_consistency")


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


class TestMainCli:
    """Unit guard for the public self-consistency benchmark CLI entry point."""

    def _run_main(
        self,
        tmp_path: Path,
        response: str = "SUPPORTED",
        *,
        k: int = 3,
        temperature: float = 0.4,
        top_p: float = 0.95,
    ) -> tuple[dict[str, object], MagicMock]:
        """Run ``main()`` with mocked optional dependency surfaces."""
        out_file = tmp_path / "self_consistency_result.json"
        mock = MagicMock()
        mock.create_chat_completion.return_value = {
            "choices": [{"message": {"content": response}}],
        }

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model",
                    "/fake.gguf",
                    "--max-samples",
                    "6",
                    "--k",
                    str(k),
                    "--temperature",
                    str(temperature),
                    "--top-p",
                    str(top_p),
                    "--output",
                    str(out_file),
                    "--log-every",
                    "2",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            main = cast(Callable[[], int], _self_consistency_module.main)
            assert main() == 0

        return (
            cast(dict[str, object], json.loads(out_file.read_text(encoding="utf-8"))),
            mock,
        )

    def test_schema_completeness(self, tmp_path: Path) -> None:
        """Write all public report fields expected by downstream analyzers."""
        report, _mock = self._run_main(tmp_path)
        for key in (
            "schema_version",
            "model",
            "method",
            "samples",
            "k",
            "temperature",
            "top_p",
            "global_balanced_accuracy",
            "per_dataset",
            "per_family",
            "dataset_to_family",
            "unknown_predictions",
            "predictions",
            "support_fractions",
            "labels",
            "datasets_per_sample",
            "families_per_sample",
        ):
            assert key in report, f"missing {key!r}"

    def test_k_value_stored(self, tmp_path: Path) -> None:
        """Persist the configured self-consistency sample count."""
        report, _mock = self._run_main(tmp_path, k=5)
        assert report["k"] == 5

    def test_temperature_stored(self, tmp_path: Path) -> None:
        """Persist the configured sampling temperature."""
        report, _mock = self._run_main(tmp_path, temperature=0.7)
        assert report["temperature"] == 0.7

    def test_top_p_stored(self, tmp_path: Path) -> None:
        """Persist the configured nucleus-sampling probability."""
        report, _mock = self._run_main(tmp_path, top_p=0.8)
        assert report["top_p"] == 0.8

    def test_llm_called_k_times_per_sample(self, tmp_path: Path) -> None:
        """Call the model exactly once per sample and self-consistency vote."""
        _report, mock = self._run_main(tmp_path, k=3)
        assert mock.create_chat_completion.call_count == 18

    def test_all_supported_gives_fraction_one(self, tmp_path: Path) -> None:
        """Compute support fraction 1.0 when every vote supports the claim."""
        report, _mock = self._run_main(tmp_path, "SUPPORTED", k=3)
        assert report["support_fractions"] == [1.0] * 6

    def test_all_not_supported_gives_fraction_zero(self, tmp_path: Path) -> None:
        """Compute support fraction 0.0 when every vote rejects the claim."""
        report, _mock = self._run_main(tmp_path, "NOT_SUPPORTED", k=3)
        assert report["support_fractions"] == [0.0] * 6

    def test_all_unknown_gives_none_fractions(self, tmp_path: Path) -> None:
        """Represent all-unparsable vote sets as unknown predictions."""
        report, _mock = self._run_main(tmp_path, "gibberish", k=3)
        assert report["unknown_predictions"] == 6
        assert report["support_fractions"] == [None] * 6

    def test_per_family_present(self, tmp_path: Path) -> None:
        """Report metrics for every routed prompt family represented."""
        report, _mock = self._run_main(tmp_path)
        per_family = cast(dict[str, object], report["per_family"])
        assert set(per_family) == {"summ", "rag", "claim"}

    def test_ba_with_all_supported(self, tmp_path: Path) -> None:
        """Report baseline balanced accuracy when all samples predict support."""
        report, _mock = self._run_main(tmp_path, "SUPPORTED")
        assert report["global_balanced_accuracy"] == 0.5

    def test_families_per_sample(self, tmp_path: Path) -> None:
        """Persist the routed prompt family selected for every sample."""
        report, _mock = self._run_main(tmp_path)
        assert report["families_per_sample"] == [
            "summ",
            "summ",
            "rag",
            "rag",
            "claim",
            "claim",
        ]

    def test_mixed_votes(self, tmp_path: Path) -> None:
        """Use majority voting when K sampled verdicts disagree."""
        out_file = tmp_path / "self_consistency_mixed.json"
        mock = MagicMock()
        responses = ["SUPPORTED", "NOT_SUPPORTED", "SUPPORTED"]
        call_index = [0]

        def side_effect(**_kwargs: object) -> dict[str, object]:
            response = responses[call_index[0] % 3]
            call_index[0] += 1
            return {"choices": [{"message": {"content": response}}]}

        mock.create_chat_completion.side_effect = side_effect

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model",
                    "/fake.gguf",
                    "--max-samples",
                    "6",
                    "--k",
                    "3",
                    "--temperature",
                    "0.4",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "100",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            main = cast(Callable[[], int], _self_consistency_module.main)
            assert main() == 0

        report = cast(
            dict[str, object],
            json.loads(out_file.read_text(encoding="utf-8")),
        )
        support_fractions = cast(list[float], report["support_fractions"])
        for support_fraction in support_fractions:
            assert abs(support_fraction - 2 / 3) < 0.01
        predictions = cast(list[int], report["predictions"])
        assert predictions == [1] * 6
