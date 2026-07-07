# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Unit coverage for the Gemma AggreFact HiSS evaluator."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("_ensure_datasets_stub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from gemma_aggrefact_hiss import (  # noqa: E402
    build_report,
    evaluate_dataset,
    main,
    parse_subclaims,
)

AggreFactRow = dict[str, object]


class MockDataset:
    """Small Hugging Face Dataset-compatible in-memory dataset."""

    def __init__(self, rows: list[AggreFactRow]) -> None:
        """Store rows for iteration and selection."""
        self._rows = rows

    def select(self, indices: range) -> MockDataset:
        """Return a selected subset of rows."""
        return MockDataset([self._rows[index] for index in indices])

    def __len__(self) -> int:
        """Return the number of stored rows."""
        return len(self._rows)

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Iterate over stored AggreFact rows."""
        return iter(self._rows)


class StubLlama:
    """Small llama-cpp-compatible HiSS chat-completion stub."""

    def __init__(self, *, fail: bool = False) -> None:
        """Configure deterministic responses or a backend failure."""
        self._fail = fail
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Mapping[str, object]:
        """Return a llama-cpp-compatible chat-completion payload."""
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        if self._fail:
            raise RuntimeError("OOM")
        content = messages[0]["content"]
        if "Break the CLAIM" in content:
            return {
                "choices": [
                    {"message": {"content": "1. First sub-claim\n2. Second sub-claim"}}
                ]
            }
        return {"choices": [{"message": {"content": "SUPPORTED"}}]}


def _toy_dataset() -> MockDataset:
    """Return minimal AggreFact-like samples for CLI unit tests."""
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


@pytest.mark.parametrize(
    ("raw", "fallback", "expected_count", "expected_fragment"),
    [
        (
            "1. The sky is blue.\n2. The grass is green.\n3. Water flows.",
            "orig",
            3,
            "sky is blue",
        ),
        ("- The sky is blue.\n- The grass is green.", "orig", 2, "sky is blue"),
        ("* First claim\n* Second claim", "orig", 2, "First claim"),
        ("1) First sub\n2) Second sub", "orig", 2, "First sub"),
        ("I cannot break this down.", "The claim.", 1, "cannot break"),
        ("The claim is atomic and cannot be split.", "fallback", 1, "atomic"),
    ],
)
def test_parse_subclaims_formats(
    raw: str,
    fallback: str,
    expected_count: int,
    expected_fragment: str,
) -> None:
    """Parse common HiSS decomposition response formats."""
    result = parse_subclaims(raw, fallback)
    assert len(result) == expected_count
    assert expected_fragment in result[0]


def test_parse_subclaims_empty_returns_original() -> None:
    """Empty decomposition responses should fall back to the original claim."""
    assert parse_subclaims("", "The original claim.") == ["The original claim."]


def test_parse_subclaims_whitespace_returns_original() -> None:
    """Whitespace-only decomposition responses should fall back to the claim."""
    assert parse_subclaims("   \n  \n", "The original claim.") == [
        "The original claim."
    ]


def test_parse_subclaims_caps_at_five() -> None:
    """HiSS decomposition should cap parsed subclaims at five."""
    raw = "\n".join(f"{index}. Sub-claim number {index}" for index in range(1, 10))
    assert len(parse_subclaims(raw, "original")) == 5


def test_parse_subclaims_filters_meta_labels() -> None:
    """Meta-label lines should not become subclaims."""
    result = parse_subclaims("1. sub-claims\n2. The actual sub-claim\n3. claim", "orig")
    assert result == ["The actual sub-claim"]


def test_parse_subclaims_strips_padding() -> None:
    """Parsed subclaims should not retain surrounding whitespace."""
    result = parse_subclaims("1.   Lots of spaces   \n2.   Also padded   ", "orig")
    assert result == ["Lots of spaces", "Also padded"]


def _run_main(
    tmp_path: Path,
    *,
    stub_llm: StubLlama | None = None,
) -> dict[str, Any]:
    """Run the HiSS CLI with patched protocol modules and return its report."""
    output_path = tmp_path / "hiss_result.json"
    if stub_llm is None:
        stub_llm = StubLlama()

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
                str(output_path),
                "--log-every",
                "2",
            ],
        ),
        patch("llama_cpp.Llama", return_value=stub_llm),
        patch("datasets.load_dataset", return_value=_toy_dataset()),
    ):
        assert main() == 0

    return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))


def test_output_schema_completeness(tmp_path: Path) -> None:
    """The HiSS CLI report must include the complete public schema."""
    results = _run_main(tmp_path)
    assert set(results) >= {
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
    }


def test_samples_count(tmp_path: Path) -> None:
    """The report should preserve the selected sample count."""
    assert _run_main(tmp_path)["samples"] == 4


def test_all_supported_subclaims_give_supported_predictions(tmp_path: Path) -> None:
    """All-supported subclaim verdicts should yield supported predictions."""
    results = _run_main(tmp_path)
    assert results["predictions"] == [1, 1, 1, 1]
    assert results["global_balanced_accuracy"] == 0.5


def test_subclaim_counts_are_populated(tmp_path: Path) -> None:
    """The report should record parsed subclaim counts for every sample."""
    results = _run_main(tmp_path)
    assert results["subclaim_counts"] == [2, 2, 2, 2]


def test_first_10_samples_field(tmp_path: Path) -> None:
    """The trace field should include compact sample-level diagnostics."""
    results = _run_main(tmp_path)
    traces = cast(list[dict[str, object]], results["first_10_samples"])
    assert len(traces) == 4
    for entry in traces:
        assert set(entry) == {"claim", "n_sub", "sub_verdicts", "pred", "label"}


def test_per_dataset_keys(tmp_path: Path) -> None:
    """Per-dataset metrics should include both represented datasets."""
    per_dataset = cast(dict[str, object], _run_main(tmp_path)["per_dataset"])
    assert set(per_dataset) == {"AggreFact-CNN", "RAGTruth"}


def test_exception_in_decompose_becomes_unknown(tmp_path: Path) -> None:
    """Backend failures should produce unknown predictions without aborting."""
    results = _run_main(tmp_path, stub_llm=StubLlama(fail=True))
    assert results["unknown_predictions"] == 4
    assert results["predictions"] == [-1, -1, -1, -1]


def test_method_mentions_hiss(tmp_path: Path) -> None:
    """The method label should identify the HiSS prompt strategy."""
    assert "HiSS" in str(_run_main(tmp_path)["method"])


def test_mean_subclaims_positive(tmp_path: Path) -> None:
    """Successful decomposition should produce a positive mean subclaim count."""
    assert _run_main(tmp_path)["mean_subclaims_per_sample"] > 0


def test_evaluate_dataset_rejects_empty_dataset() -> None:
    """Direct HiSS evaluation calls should reject empty datasets early."""
    with pytest.raises(ValueError, match="dataset is empty"):
        evaluate_dataset(
            MockDataset([]),
            StubLlama(),
            max_decompose_tokens=16,
            max_verify_tokens=4,
            log_every=1,
        )


def test_build_report_rejects_empty_metrics() -> None:
    """Direct report calls should not divide by zero on empty metrics."""
    with pytest.raises(ValueError, match="requires at least one sample"):
        build_report(
            model_path="/fake.gguf",
            sample_count=0,
            preds=[],
            labels=[],
            datasets_per_sample=[],
            subclaim_counts=[],
            latencies=[],
            traces=[],
            unknown=0,
            total_time=0.0,
        )
