# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Unit coverage for the Gemma AggreFact CoT evaluator helpers and CLI."""

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

from gemma_aggrefact_cot import compute_ba, main, parse_cot  # noqa: E402

AggreFactRow = dict[str, object]


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The claim is correct.\nANSWER: SUPPORTED", 1),
        ("Missing evidence.\nANSWER: NOT_SUPPORTED", 0),
        ("Nope.\nANSWER: NOT SUPPORTED", 0),
        ("Wrong.\nANSWER: NOT-SUPPORTED", 0),
        ("answer: supported", 1),
        ("Answer : NOT_SUPPORTED", 0),
        ("The claim is NOT_SUPPORTED by the context.", 0),
        ("This is clearly not supported.", 0),
        ("Fully supported by the evidence provided.", 1),
        ("The claim is NOT_SUPPORTED even though parts are SUPPORTED.", 0),
        ("I cannot determine the answer.", -1),
        ("", -1),
        ("ERROR: timeout", -1),
        (
            "Step 1: The context mentions the sky is blue.\n"
            "Step 2: The claim says the sky is blue.\n"
            "ANSWER: SUPPORTED",
            1,
        ),
        ("ANSWER: NOT_SUPPORTED\nBut I am not sure.", 0),
        ("The claim seems SUPPORTED but\nANSWER: NOT_SUPPORTED", 0),
    ],
)
def test_parse_cot_responses(response: str, expected: int) -> None:
    """Parse explicit and fallback CoT verdict formats."""
    assert parse_cot(response) == expected


@pytest.mark.parametrize(
    ("preds", "labels", "expected"),
    [
        ([1, 0, 1, 0], [1, 0, 1, 0], 1.0),
        ([0, 1, 0, 1], [1, 0, 1, 0], 0.0),
        ([1, 0, 0, 1], [1, 0, 1, 0], 0.5),
        ([], [], 0.0),
        ([1, -1, -1, 0], [1, 0, 1, 0], 1.0),
        ([1, 1], [1, 1], 0.0),
        ([0, 0], [0, 0], 0.0),
        ([-1, -1], [1, 0], 0.0),
    ],
)
def test_compute_ba_cases(
    preds: list[int],
    labels: list[int],
    expected: float,
) -> None:
    """Compute balanced accuracy for normal and degenerate inputs."""
    assert compute_ba(preds, labels) == expected


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
    """Small llama-cpp-compatible chat completion stub."""

    def __init__(
        self,
        response_text: str = "ANSWER: SUPPORTED",
        failure: RuntimeError | None = None,
    ) -> None:
        """Configure the deterministic response or failure."""
        self._response_text = response_text
        self._failure = failure
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
        if self._failure is not None:
            raise self._failure
        return {"choices": [{"message": {"content": self._response_text}}]}


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


def _run_main(
    tmp_path: Path,
    *,
    response_text: str = "ANSWER: SUPPORTED",
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run ``main`` with protocol-module patches and return its JSON report."""
    output_path = tmp_path / "cot_result.json"
    stub_llm = StubLlama(response_text)
    args = [
        "prog",
        "--model",
        "/fake/model.gguf",
        "--max-samples",
        "4",
        "--output",
        str(output_path),
        "--log-every",
        "2",
    ]
    if extra_args is not None:
        args.extend(extra_args)

    with (
        patch("sys.argv", args),
        patch("llama_cpp.Llama", return_value=stub_llm),
        patch("datasets.load_dataset", return_value=_toy_dataset()),
    ):
        assert main() == 0

    assert output_path.exists()
    return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))


def test_output_schema_completeness(tmp_path: Path) -> None:
    """The CoT CLI report must include the complete public schema."""
    results = _run_main(tmp_path)
    assert set(results) >= {
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
        "predictions",
        "labels",
        "datasets_per_sample",
    }


def test_all_supported_balanced_accuracy(tmp_path: Path) -> None:
    """All-supported predictions should produce balanced accuracy 0.5."""
    results = _run_main(tmp_path, response_text="ANSWER: SUPPORTED")
    assert results["samples"] == 4
    assert results["global_balanced_accuracy"] == 0.5


def test_all_not_supported_balanced_accuracy(tmp_path: Path) -> None:
    """All-not-supported predictions should produce balanced accuracy 0.5."""
    results = _run_main(tmp_path, response_text="ANSWER: NOT_SUPPORTED")
    assert results["global_balanced_accuracy"] == 0.5


def test_unknown_responses_are_counted(tmp_path: Path) -> None:
    """Unparseable responses should increment the unknown counter."""
    results = _run_main(tmp_path, response_text="I have no idea")
    assert results["unknown_predictions"] == 4


def test_per_dataset_metric_payloads(tmp_path: Path) -> None:
    """Per-dataset report entries should include samples and accuracy."""
    results = _run_main(tmp_path)
    per_dataset = cast(dict[str, dict[str, object]], results["per_dataset"])
    assert set(per_dataset) == {"AggreFact-CNN", "RAGTruth"}
    for metrics in per_dataset.values():
        assert set(metrics) == {"samples", "balanced_accuracy"}


def test_sample_responses_are_truncated(tmp_path: Path) -> None:
    """Sample responses should contain at most 20 entries of 80 characters."""
    results = _run_main(tmp_path)
    responses = cast(list[str], results["sample_responses"])
    assert len(responses) <= 20
    assert all(len(response) <= 80 for response in responses)


def test_latency_fields_are_non_negative(tmp_path: Path) -> None:
    """Latency and elapsed-time fields should be non-negative."""
    results = _run_main(tmp_path)
    assert results["p50_latency_ms"] >= 0
    assert results["p99_latency_ms"] >= 0
    assert results["total_time_seconds"] >= 0


def test_model_name_is_preserved(tmp_path: Path) -> None:
    """The configured model path should be copied into the report."""
    results = _run_main(tmp_path)
    assert results["model"] == "/fake/model.gguf"


def test_exception_in_llm_becomes_unknown(tmp_path: Path) -> None:
    """Per-sample backend failures should become unknown predictions."""
    output_path = tmp_path / "cot_err.json"
    failing_llm = StubLlama(failure=RuntimeError("GPU OOM"))
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
                "100",
            ],
        ),
        patch("llama_cpp.Llama", return_value=failing_llm),
        patch("datasets.load_dataset", return_value=_toy_dataset()),
    ):
        assert main() == 0

    results = cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))
    assert results["unknown_predictions"] == 4


def test_max_tokens_cli_arg_is_forwarded(tmp_path: Path) -> None:
    """The ``--max-tokens`` argument should reach the llama-cpp call."""
    output_path = tmp_path / "cot_maxtok.json"
    stub_llm = StubLlama("ANSWER: SUPPORTED")
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
                str(output_path),
                "--log-every",
                "100",
            ],
        ),
        patch("llama_cpp.Llama", return_value=stub_llm),
        patch("datasets.load_dataset", return_value=_toy_dataset()),
    ):
        assert main() == 0

    assert {call["max_tokens"] for call in stub_llm.calls} == {32}
