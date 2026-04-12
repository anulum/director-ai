# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.gemma_aggrefact_hiss``.

Covers:

* ``parse_subclaims()`` — numbered lists, bullet lists, empty decomposition,
  fallback to original claim, max 5 cap, filtering of meta-labels.
* ``main()`` — mocked Llama + datasets, verifies JSON schema, HiSS-specific
  fields (subclaim_counts, first_10_samples), per-dataset metrics, exception path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from gemma_aggrefact_hiss import parse_subclaims  # noqa: E402

# ── MockDataset ─────────────────────────────────────────────────────────


class MockDataset:
    """Simulates a HuggingFace Dataset with .select() and iteration."""

    def __init__(self, rows: list[dict]):
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
            },
            {
                "doc": "Sky is blue.",
                "claim": "Sky is red.",
                "label": 0,
                "dataset": "AggreFact-CNN",
            },
            {
                "doc": "Water is wet.",
                "claim": "Water is wet.",
                "label": 1,
                "dataset": "RAGTruth",
            },
            {
                "doc": "Water is wet.",
                "claim": "Fire is cold.",
                "label": 0,
                "dataset": "RAGTruth",
            },
        ]
    )


# ── parse_subclaims ────────────────────────────────────────────────────


class TestParseSubclaims:
    def test_numbered_list(self):
        raw = "The sky is blue.\n2. The grass is green.\n3. Water flows."
        result = parse_subclaims(raw, "original")
        assert len(result) == 3
        assert "sky is blue" in result[0].lower()

    def test_bullet_list(self):
        raw = "- The sky is blue.\n- The grass is green."
        result = parse_subclaims(raw, "original")
        assert len(result) == 2

    def test_asterisk_list(self):
        raw = "* First claim\n* Second claim"
        result = parse_subclaims(raw, "original")
        assert len(result) == 2

    def test_paren_numbering(self):
        raw = "1) First sub\n2) Second sub"
        result = parse_subclaims(raw, "original")
        assert len(result) == 2

    def test_empty_returns_original(self):
        result = parse_subclaims("", "The original claim.")
        assert result == ["The original claim."]

    def test_gibberish_without_list_format_parsed_as_single(self):
        """Gibberish text gets '1. ' prepended and parsed as a single sub-claim."""
        result = parse_subclaims("I cannot break this down.", "The claim.")
        assert len(result) == 1
        assert "cannot break" in result[0]

    def test_truly_empty_returns_original(self):
        """Only whitespace → no regex matches → fallback to original."""
        result = parse_subclaims("   \n  \n", "The original claim.")
        assert result == ["The original claim."]

    def test_max_five_cap(self):
        raw = "\n".join(f"{i}. Sub-claim number {i}" for i in range(1, 10))
        result = parse_subclaims(raw, "original")
        assert len(result) == 5

    def test_meta_labels_filtered(self):
        """Lines containing just 'sub-claims' or 'claim' are filtered out."""
        raw = "1. sub-claims\n2. The actual sub-claim\n3. claim"
        result = parse_subclaims(raw, "original")
        assert len(result) == 1
        assert "actual" in result[0].lower()

    def test_leading_1_prefix_added(self):
        """If raw doesn't start with a list marker, '1. ' is prepended."""
        raw = "The claim is atomic and cannot be split."
        result = parse_subclaims(raw, "fallback")
        # Should try to parse "1. The claim is atomic..." as a single sub-claim
        assert len(result) == 1
        assert "atomic" in result[0].lower()

    def test_whitespace_stripped(self):
        raw = "1.   Lots of spaces   \n2.   Also padded   "
        result = parse_subclaims(raw, "original")
        assert not result[0].startswith(" ")
        assert not result[0].endswith(" ")


# ── main() CLI integration ──────────────────────────────────────────────


def _mock_hiss_llm():
    """Mock Llama for HiSS: decompose returns 2 subclaims, verify returns SUPPORTED."""
    mock = MagicMock()
    call_count = [0]

    def side_effect(**kwargs):
        call_count[0] += 1
        messages = kwargs.get("messages", [])
        content = messages[0]["content"] if messages else ""
        # Decompose prompt contains "Break the CLAIM"
        if "Break the CLAIM" in content:
            return {
                "choices": [
                    {"message": {"content": "First sub-claim\n2. Second sub-claim"}}
                ]
            }
        # Verify prompt contains "SUPPORTED or NOT_SUPPORTED"
        return {"choices": [{"message": {"content": "SUPPORTED"}}]}

    mock.create_chat_completion.side_effect = side_effect
    return mock


class TestMainCli:
    def _run_main(self, tmp_path, mock_llm=None):
        out_file = tmp_path / "hiss_result.json"
        if mock_llm is None:
            mock_llm = _mock_hiss_llm()
        mock_ds = _toy_dataset()

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
            patch("llama_cpp.Llama", return_value=mock_llm),
            patch("datasets.load_dataset", create=True, return_value=mock_ds),
        ):
            from gemma_aggrefact_hiss import main

            main()

        return json.loads(out_file.read_text())

    def test_output_schema_completeness(self, tmp_path):
        results = self._run_main(tmp_path)
        for key in (
            "model",
            "method",
            "samples",
            "global_balanced_accuracy",
            "per_dataset",
            "unknown_predictions",
            "mean_subclaims_per_sample",
            "total_time_seconds",
            "predictions",
            "labels",
            "datasets_per_sample",
            "subclaim_counts",
            "first_10_samples",
        ):
            assert key in results, f"missing key {key!r}"

    def test_samples_count(self, tmp_path):
        results = self._run_main(tmp_path)
        assert results["samples"] == 4

    def test_all_supported_subclaims_gives_supported(self, tmp_path):
        """Mock returns SUPPORTED for all subclaims → all preds = 1."""
        results = self._run_main(tmp_path)
        # All subclaims SUPPORTED → pred=1 for all
        assert all(p == 1 for p in results["predictions"])
        # BA = 0.5 (2 correct pos, 2 wrong neg → recall_pos=1, recall_neg=0)
        assert results["global_balanced_accuracy"] == 0.5

    def test_subclaim_counts_populated(self, tmp_path):
        results = self._run_main(tmp_path)
        assert len(results["subclaim_counts"]) == 4
        # Each decomposition should produce 2 subclaims (from our mock)
        assert all(c == 2 for c in results["subclaim_counts"])

    def test_first_10_samples_field(self, tmp_path):
        results = self._run_main(tmp_path)
        # With 4 samples, first_10_samples has 4 entries
        assert len(results["first_10_samples"]) == 4
        for entry in results["first_10_samples"]:
            assert "claim" in entry
            assert "n_sub" in entry
            assert "pred" in entry
            assert "label" in entry

    def test_per_dataset_keys(self, tmp_path):
        results = self._run_main(tmp_path)
        assert "AggreFact-CNN" in results["per_dataset"]
        assert "RAGTruth" in results["per_dataset"]

    def test_exception_in_decompose(self, tmp_path):
        """If Llama fails on decompose, sample gets pred=-1."""
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("OOM")
        results = self._run_main(tmp_path, mock_llm=mock_llm)
        assert results["unknown_predictions"] == 4
        assert all(p == -1 for p in results["predictions"])

    def test_method_mentions_hiss(self, tmp_path):
        results = self._run_main(tmp_path)
        assert "HiSS" in results["method"]

    def test_mean_subclaims_positive(self, tmp_path):
        results = self._run_main(tmp_path)
        assert results["mean_subclaims_per_sample"] > 0
