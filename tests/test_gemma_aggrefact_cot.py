# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""STRONG tests for ``benchmarks.gemma_aggrefact_cot``.

Covers:

* ``parse_cot()`` — explicit ANSWER line, fallback substring, NOT_SUPPORTED
  priority, unknown, mixed case, multi-line CoT reasoning.
* ``compute_ba()`` — edge cases mirroring _judge_common tests.
* ``main()`` — mocked Llama + datasets, verifies JSON output schema,
  per-dataset metrics, latency fields, sample_responses truncation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from gemma_aggrefact_cot import compute_ba, parse_cot  # noqa: E402

# ── parse_cot ───────────────────────────────────────────────────────────


class TestParseCot:
    """Multi-angle coverage of the CoT response parser."""

    def test_explicit_answer_supported(self):
        text = "The claim is correct.\nANSWER: SUPPORTED"
        assert parse_cot(text) == 1

    def test_explicit_answer_not_supported(self):
        text = "Missing evidence.\nANSWER: NOT_SUPPORTED"
        assert parse_cot(text) == 0

    def test_explicit_answer_not_space_supported(self):
        text = "Nope.\nANSWER: NOT SUPPORTED"
        assert parse_cot(text) == 0

    def test_explicit_answer_not_dash_supported(self):
        text = "Wrong.\nANSWER: NOT-SUPPORTED"
        assert parse_cot(text) == 0

    def test_explicit_answer_case_insensitive(self):
        text = "answer: supported"
        assert parse_cot(text) == 1

    def test_explicit_answer_with_colon_space(self):
        text = "Answer : NOT_SUPPORTED"
        assert parse_cot(text) == 0

    def test_fallback_not_supported_in_text(self):
        """No ANSWER line, but NOT_SUPPORTED appears in reasoning."""
        text = "The claim is NOT_SUPPORTED by the context."
        assert parse_cot(text) == 0

    def test_fallback_not_space_supported(self):
        text = "This is clearly not supported."
        assert parse_cot(text) == 0

    def test_fallback_supported_only(self):
        text = "Fully supported by the evidence provided."
        assert parse_cot(text) == 1

    def test_not_supported_priority_over_supported(self):
        """NOT_SUPPORTED substring check must fire before SUPPORTED."""
        text = "The claim is NOT_SUPPORTED even though some parts are SUPPORTED."
        assert parse_cot(text) == 0

    def test_unknown_gibberish(self):
        text = "I cannot determine the answer."
        assert parse_cot(text) == -1

    def test_empty_string(self):
        assert parse_cot("") == -1

    def test_error_string(self):
        assert parse_cot("ERROR: timeout") == -1

    def test_multiline_reasoning_with_answer(self):
        text = (
            "Step 1: The context mentions the sky is blue.\n"
            "Step 2: The claim says the sky is blue.\n"
            "These match.\n"
            "ANSWER: SUPPORTED"
        )
        assert parse_cot(text) == 1

    def test_answer_not_at_end(self):
        """ANSWER line in the middle of text is still captured."""
        text = "ANSWER: NOT_SUPPORTED\nBut I'm not sure."
        assert parse_cot(text) == 0

    def test_answer_takes_priority_over_fallback(self):
        """Explicit ANSWER line overrides fallback substring logic."""
        text = "The claim seems SUPPORTED but\nANSWER: NOT_SUPPORTED"
        assert parse_cot(text) == 0


# ── compute_ba ──────────────────────────────────────────────────────────


class TestComputeBa:
    """The CoT script has its own BA implementation — verify parity."""

    def test_all_correct(self):
        assert compute_ba([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0

    def test_all_wrong(self):
        assert compute_ba([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0

    def test_half_right(self):
        assert compute_ba([1, 0, 0, 1], [1, 0, 1, 0]) == 0.5

    def test_empty_returns_zero(self):
        assert compute_ba([], []) == 0.0

    def test_unknowns_filtered(self):
        assert compute_ba([1, -1, -1, 0], [1, 0, 1, 0]) == 1.0

    def test_only_positive_class(self):
        assert compute_ba([1, 1], [1, 1]) == 0.0  # no negatives → 0.0

    def test_only_negative_class(self):
        assert compute_ba([0, 0], [0, 0]) == 0.0  # no positives → 0.0

    def test_all_unknown(self):
        assert compute_ba([-1, -1], [1, 0]) == 0.0


# ── Toy dataset ─────────────────────────────────────────────────────────


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
    """Minimal synthetic AggreFact-like samples."""
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


def _mock_llama_cot(response_text: str = "ANSWER: SUPPORTED"):
    """Create a mock Llama that always returns the given response."""
    mock = MagicMock()
    mock.create_chat_completion.return_value = {
        "choices": [{"message": {"content": response_text}}],
    }
    return mock


# ── main() CLI integration ──────────────────────────────────────────────


class TestMainCli:
    """Integration tests for main() with mocked backends."""

    def _run_main(self, tmp_path, response_text="ANSWER: SUPPORTED", extra_args=None):
        """Helper: run main() with mocked Llama + dataset."""
        out_file = tmp_path / "cot_result.json"
        mock_llm = _mock_llama_cot(response_text)
        mock_ds = _toy_dataset()

        args = [
            "prog",
            "--model",
            "/fake/model.gguf",
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
            patch("datasets.load_dataset", return_value=mock_ds),
        ):
            from gemma_aggrefact_cot import main

            main()

        assert out_file.exists()
        return json.loads(out_file.read_text())

    def test_output_schema_completeness(self, tmp_path):
        results = self._run_main(tmp_path)
        for key in (
            "model",
            "prompt_style",
            "samples",
            "global_balanced_accuracy",
            "per_dataset",
            "unknown_predictions",
            "total_time_seconds",
            "p50_latency_ms",
            "p99_latency_ms",
            "sample_responses",
        ):
            assert key in results, f"missing key {key!r}"

    def test_all_supported_ba(self, tmp_path):
        """All samples predicted SUPPORTED → BA = 0.5 (all-1 preds)."""
        results = self._run_main(tmp_path, "ANSWER: SUPPORTED")
        assert results["samples"] == 4
        # 2 pos correct, 2 neg wrong → recall_pos=1.0, recall_neg=0.0 → BA=0.5
        assert results["global_balanced_accuracy"] == 0.5

    def test_all_not_supported_ba(self, tmp_path):
        results = self._run_main(tmp_path, "ANSWER: NOT_SUPPORTED")
        # 2 pos wrong, 2 neg correct → BA=0.5
        assert results["global_balanced_accuracy"] == 0.5

    def test_unknown_responses_counted(self, tmp_path):
        results = self._run_main(tmp_path, "I have no idea")
        assert results["unknown_predictions"] == 4

    def test_per_dataset_keys(self, tmp_path):
        results = self._run_main(tmp_path)
        assert "AggreFact-CNN" in results["per_dataset"]
        assert "RAGTruth" in results["per_dataset"]
        for ds_metrics in results["per_dataset"].values():
            assert "samples" in ds_metrics
            assert "balanced_accuracy" in ds_metrics

    def test_sample_responses_truncated(self, tmp_path):
        """sample_responses should contain at most first 20 entries, each max 80 chars."""
        results = self._run_main(tmp_path)
        assert len(results["sample_responses"]) <= 20
        for resp in results["sample_responses"]:
            assert len(resp) <= 80

    def test_latencies_positive(self, tmp_path):
        results = self._run_main(tmp_path)
        assert results["p50_latency_ms"] >= 0
        assert results["p99_latency_ms"] >= 0
        assert results["total_time_seconds"] >= 0

    def test_model_name_preserved(self, tmp_path):
        results = self._run_main(tmp_path)
        assert results["model"] == "/fake/model.gguf"

    def test_exception_in_llm_becomes_unknown(self, tmp_path):
        """If Llama raises during inference, the sample becomes unknown."""
        out_file = tmp_path / "cot_err.json"
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.side_effect = RuntimeError("GPU OOM")
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
                    "100",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock_llm),
            patch("datasets.load_dataset", return_value=mock_ds),
        ):
            from gemma_aggrefact_cot import main

            main()

        results = json.loads(out_file.read_text())
        # All samples should have ERROR text → parse_cot returns -1
        assert results["unknown_predictions"] == 4

    def test_max_tokens_cli_arg(self, tmp_path):
        """Verify --max-tokens is passed through to Llama."""
        out_file = tmp_path / "cot_maxtok.json"
        mock_llm = _mock_llama_cot("ANSWER: SUPPORTED")
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
                    "--max-tokens",
                    "32",
                    "--output",
                    str(out_file),
                    "--log-every",
                    "100",
                ],
            ),
            patch("llama_cpp.Llama", return_value=mock_llm),
            patch("datasets.load_dataset", return_value=mock_ds),
        ):
            from gemma_aggrefact_cot import main

            main()

        # Verify the mock was called with max_tokens=32
        call_kwargs = mock_llm.create_chat_completion.call_args
        assert (
            call_kwargs.kwargs.get("max_tokens") == 32
            or call_kwargs[1].get("max_tokens") == 32
        )
