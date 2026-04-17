# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for ``benchmarks.gemma_aggrefact_hiss_routed``.

Covers:

* ``parse_subclaims()`` — max_n parameter, numbered/bullet/empty.
* ``main()`` — length gate (short claims skip decomposition), soft aggregation
  (--support-frac), routed verify prompts, JSON schema with HiSS-specific fields.
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

from gemma_aggrefact_hiss_routed import parse_subclaims  # noqa: E402


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


# ── parse_subclaims ────────────────────────────────────────────────────


class TestParseSubclaims:
    def test_max_n_caps(self):
        raw = "\n".join(f"{i}. sub {i}" for i in range(1, 10))
        result = parse_subclaims(raw, "orig", max_n=3)
        assert len(result) == 3

    def test_empty_falls_back(self):
        result = parse_subclaims("", "fallback claim")
        assert result == ["fallback claim"]

    def test_numbered_list(self):
        raw = "First thing\n2. Second thing\n3. Third thing"
        result = parse_subclaims(raw, "orig")
        assert len(result) == 3


# ── main() ────────────────────────────────────────────────────────────��─


def _toy_dataset():
    """Mix of short (<12 words) and long claims to test length gate."""
    return MockDataset(
        [
            {
                "doc": "ctx",
                "claim": "Short claim.",
                "label": 1,
                "dataset": "AggreFact-CNN",
            },
            {
                "doc": "ctx",
                "claim": "Another short one.",
                "label": 0,
                "dataset": "AggreFact-CNN",
            },
            {
                "doc": "ctx",
                "claim": "This is a much longer claim with many words that should trigger decomposition into subclaims.",
                "label": 1,
                "dataset": "RAGTruth",
            },
            {
                "doc": "ctx",
                "claim": "Yet another verbose claim that definitely has more than twelve words in it for testing.",
                "label": 0,
                "dataset": "RAGTruth",
            },
        ]
    )


def _mock_hiss_llm():
    mock = MagicMock()

    def side_effect(**kwargs):
        messages = kwargs.get("messages", [])
        content = messages[0]["content"] if messages else ""
        if "Break the CLAIM" in content:
            return {"choices": [{"message": {"content": "First sub\n2. Second sub"}}]}
        return {"choices": [{"message": {"content": "SUPPORTED"}}]}

    mock.create_chat_completion.side_effect = side_effect
    return mock


class TestMainCli:
    def _run_main(self, tmp_path, mock_llm=None, extra_args=None):
        out_file = tmp_path / "hiss_r.json"
        if mock_llm is None:
            mock_llm = _mock_hiss_llm()

        args = [
            "prog",
            "--model",
            "/fake.gguf",
            "--max-samples",
            "4",
            "--output",
            str(out_file),
            "--log-every",
            "2",
        ]
        if extra_args:
            args.extend(extra_args)

        with (
            patch("sys.argv", args),
            patch("llama_cpp.Llama", return_value=mock_llm),
            patch("datasets.load_dataset", return_value=_toy_dataset()),
        ):
            from gemma_aggrefact_hiss_routed import main

            main()

        return json.loads(out_file.read_text())

    def test_schema_completeness(self, tmp_path):
        r = self._run_main(tmp_path)
        for key in (
            "schema_version",
            "model",
            "method",
            "samples",
            "min_decompose_words",
            "support_frac",
            "max_subclaims",
            "skipped_decompose",
            "global_balanced_accuracy",
            "per_dataset",
            "per_family",
            "predictions",
            "support_fractions",
            "subclaim_counts",
            "decomposed_flags",
            "labels",
            "datasets_per_sample",
            "families_per_sample",
        ):
            assert key in r, f"missing {key!r}"

    def test_length_gate_skips_short_claims(self, tmp_path):
        r = self._run_main(tmp_path)
        # First 2 claims are short (<12 words) → decomposed_flags = False
        assert r["decomposed_flags"][0] is False
        assert r["decomposed_flags"][1] is False
        # Last 2 claims are long → decomposed_flags = True
        assert r["decomposed_flags"][2] is True
        assert r["decomposed_flags"][3] is True
        assert r["skipped_decompose"] == 2

    def test_subclaim_counts(self, tmp_path):
        r = self._run_main(tmp_path)
        # Short claims: subclaim_counts = 1 (direct judge)
        assert r["subclaim_counts"][0] == 1
        assert r["subclaim_counts"][1] == 1
        # Long claims: mock returns 2 subclaims
        assert r["subclaim_counts"][2] == 2
        assert r["subclaim_counts"][3] == 2

    def test_support_fractions_present(self, tmp_path):
        r = self._run_main(tmp_path)
        assert len(r["support_fractions"]) == 4
        # Short claims: SUPPORTED → 1.0; Long claims: all sub SUPPORTED → 1.0
        for frac in r["support_fractions"]:
            assert frac == 1.0

    def test_custom_support_frac(self, tmp_path):
        r = self._run_main(tmp_path, extra_args=["--support-frac", "1.0"])
        assert r["support_frac"] == 1.0

    def test_custom_min_decompose_words(self, tmp_path):
        r = self._run_main(tmp_path, extra_args=["--min-decompose-words", "100"])
        # All claims have <100 words → all skipped
        assert r["skipped_decompose"] == 4

    def test_exception_in_decompose(self, tmp_path):
        mock = MagicMock()
        mock.create_chat_completion.side_effect = RuntimeError("fail")
        r = self._run_main(tmp_path, mock_llm=mock)
        # Short claims fail at verify, long fail at decompose → all unknown
        assert r["unknown_predictions"] >= 2
