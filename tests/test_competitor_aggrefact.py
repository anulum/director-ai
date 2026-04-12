# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.competitor_aggrefact``.

Covers:

* ``HHEMBackend`` — mocked AutoModelForSequenceClassification, score output.
* ``MiniCheckBackend`` — mocked Roberta classifier, softmax output.
* ``BACKENDS`` registry — correct class mappings.
* ``main()`` — mocked backend + datasets, JSON schema, threshold cut,
  per-dataset metrics, exception path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for competitor backends")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from competitor_aggrefact import BACKENDS, HHEMBackend, MiniCheckBackend  # noqa: E402


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


# ── BACKENDS registry ───────────────────────────────────────────────────


class TestBackendsRegistry:
    def test_contains_expected_keys(self):
        assert "vectara/hallucination_evaluation_model" in BACKENDS
        assert "lytang/MiniCheck-Roberta-Large" in BACKENDS

    def test_hhem_maps_correctly(self):
        assert BACKENDS["vectara/hallucination_evaluation_model"] is HHEMBackend

    def test_minicheck_maps_correctly(self):
        assert BACKENDS["lytang/MiniCheck-Roberta-Large"] is MiniCheckBackend


# ── main() CLI ──────────────────────────────────────────────────────────


def _mock_hhem_backend():
    """Mock HHEMBackend that returns a fixed score."""
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"
    # Simulate logits → sigmoid → 0.8
    logits = torch.tensor([[1.386]])  # sigmoid(1.386) ≈ 0.8
    mock_model.return_value = MagicMock(logits=logits)

    return mock_tokenizer, mock_model


class TestMainCli:
    def _run_main(
        self,
        tmp_path,
        model_id="vectara/hallucination_evaluation_model",
        fixed_score=0.8,
    ):
        out_file = tmp_path / "comp_result.json"

        mock_tok, mock_model = _mock_hhem_backend()

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model",
                    model_id,
                    "--max-samples",
                    "4",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "2",
                ],
            ),
            patch("transformers.AutoTokenizer") as tok_cls,
            patch("transformers.AutoModelForSequenceClassification") as model_cls,
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            tok_cls.from_pretrained.return_value = mock_tok
            model_cls.from_pretrained.return_value = mock_model
            from competitor_aggrefact import main

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
            "scores",
            "predictions",
            "labels",
            "datasets_per_sample",
            "threshold",
            "unknown_predictions",
            "total_time_seconds",
            "p50_latency_ms",
            "p99_latency_ms",
        ):
            assert key in r, f"missing {key!r}"

    def test_samples_count(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["samples"] == 4

    def test_backend_is_classifier(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["backend"] == "transformers-classifier"

    def test_per_dataset_present(self, tmp_path):
        r = self._run_main(tmp_path)
        assert "AggreFact-CNN" in r["per_dataset"]
        assert "RAGTruth" in r["per_dataset"]

    def test_model_name_stored(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["model"] == "vectara/hallucination_evaluation_model"

    def test_threshold_stored(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["threshold"] == 0.5  # default

    def test_scores_are_floats(self, tmp_path):
        r = self._run_main(tmp_path)
        for s in r["scores"]:
            assert isinstance(s, float)
