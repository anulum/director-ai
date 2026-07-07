# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Unit coverage for the routed Gemma AggreFact HiSS evaluator."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.usefixtures("_ensure_datasets_stub")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

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
    """Small llama-cpp-compatible routed HiSS chat-completion stub."""

    def __init__(self, *, fail: bool = False) -> None:
        """Configure deterministic responses or backend failures."""
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
        verdict = "NOT_SUPPORTED" if "Sky is red" in content else "SUPPORTED"
        return {"choices": [{"message": {"content": verdict}}]}


class _ParseSubclaims(Protocol):
    def __call__(self, raw: str, original_claim: str, max_n: int = 5) -> list[str]:
        """Parse decomposition text into subclaims."""


class _Main(Protocol):
    def __call__(self, argv: Sequence[str] | None = None) -> int:
        """Run the routed HiSS CLI entry point."""


class _EvaluateDataset(Protocol):
    def __call__(
        self,
        dataset: MockDataset,
        llm: StubLlama,
        *,
        min_decompose_words: int,
        support_frac: float,
        max_subclaims: int,
        max_decompose_tokens: int,
        max_verify_tokens: int,
        log_every: int,
    ) -> object:
        """Run the routed HiSS evaluation loop."""


class _BuildReport(Protocol):
    def __call__(
        self,
        *,
        model_path: str,
        sample_count: int,
        min_decompose_words: int,
        support_frac: float,
        max_subclaims: int,
        skipped_decompose: int,
        preds: list[int],
        support_fractions: list[float | None],
        labels: list[int],
        datasets_per_sample: list[str],
        families_per_sample: list[str],
        subclaim_counts: list[int],
        decomposed_flags: list[bool],
        latencies: list[float],
        unknown_predictions: int,
        total_time: float,
    ) -> object:
        """Build the routed HiSS report payload."""


def _hiss_routed_module() -> ModuleType:
    """Import the benchmark module after making benchmarks importable."""
    return importlib.import_module("gemma_aggrefact_hiss_routed")


def _parse_subclaims() -> _ParseSubclaims:
    """Return the benchmark subclaim parser."""
    return cast(_ParseSubclaims, vars(_hiss_routed_module())["parse_subclaims"])


def _main() -> _Main:
    """Return the benchmark CLI entry point."""
    return cast(_Main, vars(_hiss_routed_module())["main"])


def _evaluate_dataset() -> _EvaluateDataset:
    """Return the benchmark evaluation function."""
    return cast(_EvaluateDataset, vars(_hiss_routed_module())["evaluate_dataset"])


def _build_report() -> _BuildReport:
    """Return the benchmark report builder."""
    return cast(_BuildReport, vars(_hiss_routed_module())["build_report"])


@pytest.mark.parametrize(
    ("raw", "fallback", "max_n", "expected"),
    [
        ("\n".join(f"{index}. sub {index}" for index in range(1, 10)), "orig", 3, 3),
        ("", "fallback claim", 5, 1),
        ("First thing\n2. Second thing\n3. Third thing", "orig", 5, 3),
    ],
)
def test_parse_subclaims_formats(
    raw: str,
    fallback: str,
    max_n: int,
    expected: int,
) -> None:
    """Parse routed HiSS decomposition response variants."""
    result = _parse_subclaims()(raw, fallback, max_n=max_n)
    assert len(result) == expected


def _toy_dataset() -> MockDataset:
    """Return short and long AggreFact-like samples for CLI unit tests."""
    return MockDataset(
        [
            {
                "doc": "ctx",
                "claim": "Short claim.",
                "label": 1,
                "dataset": "AggreFact-CNN",
            },
            {
                "document": "ctx",
                "hypothesis": "Sky is red.",
                "annotations": 0,
                "dataset": "AggreFact-CNN",
            },
            {
                "doc": "ctx",
                "claim": (
                    "This is a much longer claim with many words that should "
                    "trigger decomposition into subclaims."
                ),
                "label": 1,
                "dataset": "RAGTruth",
            },
            {
                "document": "ctx",
                "hypothesis": (
                    "Yet another verbose claim that definitely has more than "
                    "twelve words in it for testing."
                ),
                "annotations": 0,
                "dataset": "Wice",
            },
        ]
    )


def _run_main(
    tmp_path: Path,
    *,
    stub_llm: StubLlama | None = None,
    dataset: MockDataset | None = None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run the routed HiSS CLI with patched protocol modules."""
    output_path = tmp_path / "hiss_routed_result.json"
    if stub_llm is None:
        stub_llm = StubLlama()
    if dataset is None:
        dataset = _toy_dataset()
    args = [
        "prog",
        "--model",
        "/fake.gguf",
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
        patch("datasets.load_dataset", return_value=dataset),
    ):
        assert _main()() == 0

    return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))


def test_output_schema_completeness(tmp_path: Path) -> None:
    """The routed HiSS report must include every public schema field."""
    results = _run_main(tmp_path)
    assert set(results) >= {
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
    }


def test_length_gate_skips_short_claims(tmp_path: Path) -> None:
    """Short claims should use routed K=1 verification without decomposition."""
    results = _run_main(tmp_path)
    assert results["decomposed_flags"] == [False, False, True, True]
    assert results["skipped_decompose"] == 2


def test_subclaim_counts_are_populated(tmp_path: Path) -> None:
    """The report should record one routed claim or parsed subclaim counts."""
    results = _run_main(tmp_path)
    assert results["subclaim_counts"] == [1, 1, 2, 2]


def test_support_fractions_are_recorded(tmp_path: Path) -> None:
    """The report should preserve support fractions for every parsed sample."""
    results = _run_main(tmp_path)
    assert results["support_fractions"] == [1.0, 0.0, 1.0, 1.0]


def test_custom_support_fraction_is_reported(tmp_path: Path) -> None:
    """The configured soft-aggregation threshold should be reported."""
    results = _run_main(tmp_path, extra_args=["--support-frac", "1.0"])
    assert results["support_frac"] == 1.0


def test_custom_min_decompose_words_skips_all_claims(tmp_path: Path) -> None:
    """Raising the decomposition word gate should skip every toy claim."""
    results = _run_main(tmp_path, extra_args=["--min-decompose-words", "100"])
    assert results["skipped_decompose"] == 4


def test_backend_failure_becomes_unknown_predictions(tmp_path: Path) -> None:
    """Backend failures should produce unknown predictions without aborting."""
    results = _run_main(tmp_path, stub_llm=StubLlama(fail=True))
    assert results["unknown_predictions"] == 4
    assert results["predictions"] == [-1, -1, -1, -1]


def test_evaluate_dataset_rejects_empty_dataset() -> None:
    """Direct routed HiSS evaluation calls should reject empty datasets early."""
    with pytest.raises(ValueError, match="dataset is empty"):
        _evaluate_dataset()(
            MockDataset([]),
            StubLlama(),
            min_decompose_words=12,
            support_frac=0.75,
            max_subclaims=4,
            max_decompose_tokens=32,
            max_verify_tokens=8,
            log_every=1,
        )


def test_build_report_rejects_empty_metrics() -> None:
    """Direct report calls should not divide by zero on empty metrics."""
    with pytest.raises(ValueError, match="requires at least one sample"):
        _build_report()(
            model_path="/fake.gguf",
            sample_count=0,
            min_decompose_words=12,
            support_frac=0.75,
            max_subclaims=4,
            skipped_decompose=0,
            preds=[],
            support_fractions=[],
            labels=[],
            datasets_per_sample=[],
            families_per_sample=[],
            subclaim_counts=[],
            decomposed_flags=[],
            latencies=[],
            unknown_predictions=0,
            total_time=0.0,
        )
