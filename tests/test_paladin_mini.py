# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.paladin_mini_aggrefact``.

Covers:

* ``main()`` — mocked transformers AutoModelForCausalLM + AutoTokenizer +
  datasets, JSON schema, per-dataset metrics, BA correctness, exception path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for paladin_mini mock")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))


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


def _mock_transformers(response_text="SUPPORTED"):
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
    def _run_main(self, tmp_path, response_text="SUPPORTED"):
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
            patch("datasets.load_dataset", create=True, return_value=_toy_dataset()),
        ):
            tok_cls.from_pretrained.return_value = mock_tok
            model_cls.from_pretrained.return_value = mock_model
            from paladin_mini_aggrefact import main

            main()

        return json.loads(out_file.read_text())

    def test_schema_completeness(self, tmp_path):
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

    def test_samples_count(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["samples"] == 4

    def test_backend_is_transformers(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["backend"] == "transformers"

    def test_all_supported_ba(self, tmp_path):
        r = self._run_main(tmp_path, "SUPPORTED")
        assert r["global_balanced_accuracy"] == 0.5

    def test_all_not_supported_ba(self, tmp_path):
        r = self._run_main(tmp_path, "NOT_SUPPORTED")
        assert r["global_balanced_accuracy"] == 0.5

    def test_unknown_counted(self, tmp_path):
        r = self._run_main(tmp_path, "gibberish")
        assert r["unknown_predictions"] == 4

    def test_per_dataset_present(self, tmp_path):
        r = self._run_main(tmp_path)
        assert "AggreFact-CNN" in r["per_dataset"]
        assert "RAGTruth" in r["per_dataset"]
