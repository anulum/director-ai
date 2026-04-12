# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.gemma_aggrefact_eval``.

Covers:

* ``LlamaCppBackend.judge()`` — prompt formatting, delegation to Llama.
* ``load_aggrefact()`` — max_samples truncation.
* ``main()`` (llama-cpp path) — JSON schema, per-dataset metrics, latencies,
  exception handling, all-supported / all-not-supported BA sanity.
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


# ── MockDataset ─────────────────────────────────────────────────────────


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
            {
                "doc": "Sky is blue.",
                "claim": "Sky is blue.",
                "label": 1,
                "dataset": "AggreFact-CNN",
                "document": "Sky is blue.",
                "hypothesis": "Sky is blue.",
                "annotations": 1,
            },
            {
                "doc": "Sky is blue.",
                "claim": "Sky is red.",
                "label": 0,
                "dataset": "AggreFact-CNN",
                "document": "Sky is blue.",
                "hypothesis": "Sky is red.",
                "annotations": 0,
            },
            {
                "doc": "Water is wet.",
                "claim": "Water is wet.",
                "label": 1,
                "dataset": "RAGTruth",
                "document": "Water is wet.",
                "hypothesis": "Water is wet.",
                "annotations": 1,
            },
            {
                "doc": "Water is wet.",
                "claim": "Fire is cold.",
                "label": 0,
                "dataset": "RAGTruth",
                "document": "Water is wet.",
                "hypothesis": "Fire is cold.",
                "annotations": 0,
            },
        ]
    )


def _mock_llm(response="SUPPORTED"):
    mock = MagicMock()
    mock.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response}}],
    }
    return mock


# ── LlamaCppBackend ─────────────────────────────────────────────────────


class TestLlamaCppBackend:
    def test_judge_delegates_to_llm(self):
        mock = _mock_llm("SUPPORTED")
        with patch("llama_cpp.Llama", return_value=mock):
            from gemma_aggrefact_eval import LlamaCppBackend

            backend = LlamaCppBackend("/fake.gguf")
            result = backend.judge("context", "claim")
        assert result == "SUPPORTED"
        mock.create_chat_completion.assert_called_once()

    def test_judge_passes_temperature_zero(self):
        mock = _mock_llm()
        with patch("llama_cpp.Llama", return_value=mock):
            from gemma_aggrefact_eval import LlamaCppBackend

            backend = LlamaCppBackend("/fake.gguf")
            backend.judge("ctx", "hyp")
        call_kwargs = mock.create_chat_completion.call_args
        assert (
            call_kwargs.kwargs.get("temperature") == 0.0
            or call_kwargs[1].get("temperature") == 0.0
        )


# ── load_aggrefact ──────────────────────────────────────────────────────


class TestLoadAggrefact:
    def test_max_samples_truncates(self):
        ds = _toy_dataset()
        with patch("datasets.load_dataset", return_value=ds):
            from gemma_aggrefact_eval import load_aggrefact

            result = load_aggrefact(max_samples=2)
        assert len(result) == 2

    def test_no_max_returns_all(self):
        ds = _toy_dataset()
        with patch("datasets.load_dataset", return_value=ds):
            from gemma_aggrefact_eval import load_aggrefact

            result = load_aggrefact()
        assert len(result) == 4


# ── main() CLI ──────────────────────────────────────────────────────────


class TestMainCli:
    def _run_main(self, tmp_path, response="SUPPORTED"):
        out_file = tmp_path / "eval_result.json"
        mock = _mock_llm(response)
        ds = _toy_dataset()

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--backend",
                    "llama-cpp",
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
            patch("datasets.load_dataset", return_value=ds),
        ):
            from gemma_aggrefact_eval import main

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
            "unknown_predictions",
            "total_time_seconds",
            "mean_latency_ms",
            "p50_latency_ms",
            "p99_latency_ms",
            "predictions",
            "labels",
            "datasets_per_sample",
        ):
            assert key in r, f"missing {key!r}"

    def test_samples_count(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["samples"] == 4
        assert len(r["predictions"]) == 4
        assert len(r["labels"]) == 4

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

    def test_backend_field(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["backend"] == "llama-cpp"

    def test_exception_path(self, tmp_path):
        out_file = tmp_path / "eval_err.json"
        mock = MagicMock()
        mock.create_chat_completion.side_effect = RuntimeError("boom")
        ds = _toy_dataset()

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--backend",
                    "llama-cpp",
                    "--model",
                    "/fake.gguf",
                    "--max-samples",
                    "4",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "100",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=ds),
        ):
            from gemma_aggrefact_eval import main

            main()

        r = json.loads(out_file.read_text())
        assert r["unknown_predictions"] == 4

    def test_latencies_non_negative(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["mean_latency_ms"] >= 0
        assert r["p50_latency_ms"] >= 0
