# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.gemma_aggrefact_self_consistency``.

Covers:

* ``main()`` — K-sample majority voting, support_fractions output,
  JSON schema (k, temperature, top_p, per_family), all-unknown path
  when every vote is unparseable, mixed vote scenarios.
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


class TestMainCli:
    def _run_main(self, tmp_path, response="SUPPORTED", k=3, temperature=0.4):
        out_file = tmp_path / "sc_result.json"
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
                    "4",
                    "--k",
                    str(k),
                    "--temperature",
                    str(temperature),
                    "--output",
                    str(out_file),
                    "--log-every",
                    "2",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            from gemma_aggrefact_self_consistency import main

            main()

        return json.loads(out_file.read_text()), mock

    def test_schema_completeness(self, tmp_path):
        r, _ = self._run_main(tmp_path)
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
            assert key in r, f"missing {key!r}"

    def test_k_value_stored(self, tmp_path):
        r, _ = self._run_main(tmp_path, k=5)
        assert r["k"] == 5

    def test_temperature_stored(self, tmp_path):
        r, _ = self._run_main(tmp_path, temperature=0.7)
        assert r["temperature"] == 0.7

    def test_llm_called_k_times_per_sample(self, tmp_path):
        """With 4 samples and K=3, expect 12 total LLM calls."""
        _, mock = self._run_main(tmp_path, k=3)
        assert mock.create_chat_completion.call_count == 12  # 4 * 3

    def test_all_supported_gives_fraction_one(self, tmp_path):
        r, _ = self._run_main(tmp_path, "SUPPORTED", k=3)
        for frac in r["support_fractions"]:
            assert frac == 1.0

    def test_all_not_supported_gives_fraction_zero(self, tmp_path):
        r, _ = self._run_main(tmp_path, "NOT_SUPPORTED", k=3)
        for frac in r["support_fractions"]:
            assert frac == 0.0

    def test_all_unknown_gives_none_fractions(self, tmp_path):
        r, _ = self._run_main(tmp_path, "gibberish", k=3)
        assert r["unknown_predictions"] == 4
        for frac in r["support_fractions"]:
            assert frac is None

    def test_per_family_present(self, tmp_path):
        r, _ = self._run_main(tmp_path)
        assert "summ" in r["per_family"]  # AggreFact-CNN
        assert "rag" in r["per_family"]  # RAGTruth

    def test_ba_with_all_supported(self, tmp_path):
        r, _ = self._run_main(tmp_path, "SUPPORTED")
        # All preds = 1 → BA = 0.5
        assert r["global_balanced_accuracy"] == 0.5

    def test_mixed_votes(self, tmp_path):
        """K=3, alternating SUPPORTED/NOT_SUPPORTED → majority wins."""
        out_file = tmp_path / "sc_mixed.json"
        mock = MagicMock()
        # Cycle: SUPPORTED, NOT_SUPPORTED, SUPPORTED → majority=SUPPORTED
        responses = ["SUPPORTED", "NOT_SUPPORTED", "SUPPORTED"]
        call_idx = [0]

        def side_effect(**kwargs):
            resp = responses[call_idx[0] % 3]
            call_idx[0] += 1
            return {"choices": [{"message": {"content": resp}}]}

        mock.create_chat_completion.side_effect = side_effect

        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    "--model",
                    "/fake.gguf",
                    "--max-samples",
                    "4",
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
            from gemma_aggrefact_self_consistency import main

            main()

        r = json.loads(out_file.read_text())
        # 2 SUPPORTED / 1 NOT → fraction = 2/3 ≈ 0.667, pred = 1
        for frac in r["support_fractions"]:
            assert abs(frac - 2 / 3) < 0.01
        assert all(p == 1 for p in r["predictions"])
