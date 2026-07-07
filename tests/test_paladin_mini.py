# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``benchmarks.paladin_mini_aggrefact``.

Covers:

* ``main()`` — mocked transformers AutoModelForCausalLM + AutoTokenizer +
  datasets, JSON schema, per-dataset metrics, BA correctness, exception path.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for paladin_mini mock")
# The CLI tests patch ``transformers.AutoTokenizer`` /
# ``transformers.AutoModelForCausalLM``; ``patch`` imports the target module, so
# transformers must be importable even though its classes are mocked. Skip
# cleanly on the base CI job, which ships without the NLI extra.
pytest.importorskip("transformers", reason="transformers required to patch backends")

pytestmark = pytest.mark.usefixtures("_ensure_datasets_stub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))


class SampleRow(TypedDict):
    """AggreFact sample row used by the Paladin benchmark tests."""

    doc: str
    claim: str
    label: int
    dataset: str


class MockDataset:
    """In-memory dataset implementing the benchmark's dataset protocol."""

    def __init__(self, rows: Sequence[SampleRow]) -> None:
        """Store benchmark sample rows."""
        self._rows = rows

    def select(self, indices: Sequence[int]) -> MockDataset:
        """Return rows selected by integer indices."""
        return MockDataset([self._rows[i] for i in indices])

    def __len__(self) -> int:
        """Return the number of sample rows."""
        return len(self._rows)

    def __iter__(self) -> Iterator[SampleRow]:
        """Iterate sample rows."""
        return iter(self._rows)

    def __getitem__(self, idx: int) -> SampleRow:
        """Return one sample row."""
        return self._rows[idx]


def _toy_dataset() -> MockDataset:
    """Build a two-dataset fixture with both binary labels."""
    return MockDataset(
        [
            {"doc": "ctx", "claim": "c1", "label": 1, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c2", "label": 0, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c3", "label": 1, "dataset": "RAGTruth"},
            {"doc": "ctx", "claim": "c4", "label": 0, "dataset": "RAGTruth"},
        ]
    )


def _mock_transformers(response_text: str = "SUPPORTED") -> tuple[MagicMock, MagicMock]:
    """Mock transformers pipeline: tokenizer + model."""
    input_tensor = torch.zeros(1, 10, dtype=torch.long)

    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = input_tensor
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.decode.return_value = response_text

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    # generate returns tensor of shape [1, 18]
    mock_model.generate.return_value = torch.zeros(1, 18, dtype=torch.long)

    return mock_tokenizer, mock_model


class TestMainCli:
    """Unit guards for the Paladin-mini benchmark main function."""

    def _run_main(
        self,
        tmp_path: Path,
        response_text: str = "SUPPORTED",
    ) -> dict[str, Any]:
        """Run the benchmark main function with patched external packages."""
        out_file = tmp_path / "paladin_result.json"
        mock_tok, mock_model = _mock_transformers(response_text)

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--max-samples",
                    "4",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "2",
                ],
            ),
            patch("transformers.AutoTokenizer") as tok_cls,
            patch("transformers.AutoModelForCausalLM") as model_cls,
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            tok_cls.from_pretrained.return_value = mock_tok
            model_cls.from_pretrained.return_value = mock_model
            from paladin_mini_aggrefact import main

            main()

        payload = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return cast(dict[str, Any], payload)

    def test_schema_completeness(self, tmp_path: Path) -> None:
        """The benchmark output should preserve the expected JSON schema."""
        r = self._run_main(tmp_path)
        for key in (
            "model",
            "backend",
            "samples",
            "global_balanced_accuracy",
            "per_dataset",
            "predictions",
            "labels",
            "datasets_per_sample",
            "unknown_predictions",
            "total_time_seconds",
            "p50_latency_ms",
            "p99_latency_ms",
        ):
            assert key in r, f"missing {key!r}"

    def test_samples_count(self, tmp_path: Path) -> None:
        """The benchmark should report the selected sample count."""
        r = self._run_main(tmp_path)
        assert r["samples"] == 4

    def test_backend_is_transformers(self, tmp_path: Path) -> None:
        """The benchmark should record the transformers backend."""
        r = self._run_main(tmp_path)
        assert r["backend"] == "transformers"

    def test_all_supported_ba(self, tmp_path: Path) -> None:
        """All-supported predictions should produce the expected balanced accuracy."""
        r = self._run_main(tmp_path, "SUPPORTED")
        assert r["global_balanced_accuracy"] == 0.5

    def test_all_not_supported_ba(self, tmp_path: Path) -> None:
        """All-not-supported predictions should produce the expected score."""
        r = self._run_main(tmp_path, "NOT_SUPPORTED")
        assert r["global_balanced_accuracy"] == 0.5

    def test_unknown_counted(self, tmp_path: Path) -> None:
        """Unparsable verdicts should be counted as unknown predictions."""
        r = self._run_main(tmp_path, "gibberish")
        assert r["unknown_predictions"] == 4

    def test_per_dataset_present(self, tmp_path: Path) -> None:
        """Per-dataset metrics should include each fixture dataset."""
        r = self._run_main(tmp_path)
        assert "AggreFact-CNN" in r["per_dataset"]
        assert "RAGTruth" in r["per_dataset"]
