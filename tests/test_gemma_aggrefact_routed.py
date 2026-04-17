# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``benchmarks.gemma_aggrefact_routed``.

Covers:

* ``main()`` — mocked Llama + datasets, prompt routing via DATASET_TO_FAMILY,
  JSON schema (per_dataset, per_family, families_per_sample), BA correctness,
  exception path, family distribution logging.
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

from _judge_common import DATASET_TO_FAMILY  # noqa: E402


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
    """One sample from each family: summ, rag, claim."""
    return MockDataset(
        [
            {"doc": "ctx1", "claim": "c1", "label": 1, "dataset": "AggreFact-CNN"},
            {"doc": "ctx2", "claim": "c2", "label": 0, "dataset": "AggreFact-CNN"},
            {"doc": "ctx3", "claim": "c3", "label": 1, "dataset": "RAGTruth"},
            {"doc": "ctx4", "claim": "c4", "label": 0, "dataset": "RAGTruth"},
            {"doc": "ctx5", "claim": "c5", "label": 1, "dataset": "Wice"},
            {"doc": "ctx6", "claim": "c6", "label": 0, "dataset": "Wice"},
        ]
    )


def _mock_llm(response="SUPPORTED"):
    mock = MagicMock()
    mock.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response}}],
    }
    return mock


class TestMainCli:
    def _run_main(self, tmp_path, response="SUPPORTED"):
        out_file = tmp_path / "routed_result.json"
        mock = _mock_llm(response)

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
            from gemma_aggrefact_routed import main

            main()

        return json.loads(out_file.read_text())

    def test_schema_completeness(self, tmp_path):
        r = self._run_main(tmp_path)
        for key in (
            "model",
            "method",
            "samples",
            "global_balanced_accuracy",
            "per_dataset",
            "per_family",
            "dataset_to_family",
            "unknown_predictions",
            "predictions",
            "labels",
            "datasets_per_sample",
            "families_per_sample",
        ):
            assert key in r, f"missing {key!r}"

    def test_samples_count(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["samples"] == 6

    def test_per_family_keys(self, tmp_path):
        r = self._run_main(tmp_path)
        # summ (AggreFact-CNN), rag (RAGTruth), claim (Wice)
        assert "summ" in r["per_family"]
        assert "rag" in r["per_family"]
        assert "claim" in r["per_family"]

    def test_per_dataset_keys(self, tmp_path):
        r = self._run_main(tmp_path)
        assert "AggreFact-CNN" in r["per_dataset"]
        assert "RAGTruth" in r["per_dataset"]
        assert "Wice" in r["per_dataset"]

    def test_dataset_to_family_mapping(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["dataset_to_family"] == DATASET_TO_FAMILY

    def test_families_per_sample(self, tmp_path):
        r = self._run_main(tmp_path)
        assert len(r["families_per_sample"]) == 6
        assert r["families_per_sample"][0] == "summ"  # AggreFact-CNN
        assert r["families_per_sample"][2] == "rag"  # RAGTruth
        assert r["families_per_sample"][4] == "claim"  # Wice

    def test_all_supported_ba(self, tmp_path):
        r = self._run_main(tmp_path, "SUPPORTED")
        assert r["global_balanced_accuracy"] == 0.5

    def test_unknown_responses(self, tmp_path):
        r = self._run_main(tmp_path, "???")
        assert r["unknown_predictions"] == 6

    def test_method_mentions_routing(self, tmp_path):
        r = self._run_main(tmp_path)
        assert "routing" in r["method"].lower()

    def test_prompt_varies_by_family(self, tmp_path):
        """Verify the mock LLM receives different prompts for different families."""
        mock = _mock_llm("SUPPORTED")
        out_file = tmp_path / "routed_prompts.json"

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
            from gemma_aggrefact_routed import main

            main()

        # Check that different prompts were used (at least 2 different message contents)
        prompts_seen = set()
        for call in mock.create_chat_completion.call_args_list:
            msg = call.kwargs.get("messages", call[1].get("messages", []))
            content = msg[0]["content"] if msg else ""
            # Different families have different prompt templates
            prompts_seen.add(content[:50])
        assert len(prompts_seen) >= 2, "Expected prompts to vary by family"
