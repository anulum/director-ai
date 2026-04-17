# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``benchmarks.gemma_aggrefact_logprob_routed``.

Covers:

* ``balanced_accuracy()`` — threshold, None scores.
* ``sweep_threshold()`` — optimal threshold search.
* ``per_dataset_sweep()`` — per-dataset thresholds.
* ``GemmaRoutedLogprobBackend.judge()`` — logprob extraction, family routing.
* ``main()`` — JSON schema with routed + logprob fields, per_family metrics.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("_ensure_datasets_stub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from gemma_aggrefact_logprob_routed import (  # noqa: E402
    balanced_accuracy,
    per_dataset_sweep,
    sweep_threshold,
)


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


# ── Metric functions ────────────────────────────────────────────────────


class TestBalancedAccuracy:
    def test_perfect(self):
        assert balanced_accuracy([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0]) == 1.0

    def test_none_skipped(self):
        assert balanced_accuracy([0.9, None, None, 0.1], [1, 0, 1, 0]) == 1.0

    def test_custom_threshold(self):
        assert balanced_accuracy([0.6, 0.4], [1, 0], threshold=0.5) == 1.0


class TestSweepThreshold:
    def test_finds_best(self):
        t, ba = sweep_threshold([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
        assert ba == 1.0


class TestPerDatasetSweep:
    def test_returns_per_ds(self):
        result, avg = per_dataset_sweep(
            [0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0], ["a", "a", "b", "b"]
        )
        assert "a" in result
        assert "b" in result
        assert avg == 1.0


# ── main() ──────────────────────────────────────────────────────────────


def _toy_dataset():
    return MockDataset(
        [
            {"doc": "ctx", "claim": "c1", "label": 1, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c2", "label": 0, "dataset": "AggreFact-CNN"},
            {"doc": "ctx", "claim": "c3", "label": 1, "dataset": "RAGTruth"},
            {"doc": "ctx", "claim": "c4", "label": 0, "dataset": "Wice"},
        ]
    )


def _mock_logprob_llm(score=0.9):
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
    def _run_main(self, tmp_path, score=0.9):
        out_file = tmp_path / "lpr_result.json"
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
            from gemma_aggrefact_logprob_routed import main

            main()

        return json.loads(out_file.read_text())

    def test_schema_completeness(self, tmp_path):
        r = self._run_main(tmp_path)
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
            assert key in r, f"missing {key!r}"

    def test_scores_continuous(self, tmp_path):
        r = self._run_main(tmp_path, score=0.8)
        for s in r["scores"]:
            assert 0.7 < s < 0.9

    def test_per_family_has_threshold(self, tmp_path):
        r = self._run_main(tmp_path)
        for fam_m in r["per_family"].values():
            assert "threshold" in fam_m
            assert "balanced_accuracy" in fam_m

    def test_method_mentions_routing_and_logprob(self, tmp_path):
        r = self._run_main(tmp_path)
        assert "routing" in r["method"].lower()
        assert "logprob" in r["method"].lower()

    def test_families_per_sample(self, tmp_path):
        r = self._run_main(tmp_path)
        assert len(r["families_per_sample"]) == 4
        assert r["families_per_sample"][0] == "summ"  # AggreFact-CNN
        assert r["families_per_sample"][2] == "rag"  # RAGTruth
        assert r["families_per_sample"][3] == "claim"  # Wice

    def test_invalid_scores_zero(self, tmp_path):
        r = self._run_main(tmp_path)
        assert r["invalid_scores"] == 0

    def test_exception_path(self, tmp_path):
        out_file = tmp_path / "lpr_err.json"
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
                    "4",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "100",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            from gemma_aggrefact_logprob_routed import main

            main()

        r = json.loads(out_file.read_text())
        assert r["invalid_scores"] == 4
