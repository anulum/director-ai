# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — BatchProcessor public contract
"""Behavioural tests for the BatchProcessor request-processing contract."""

from __future__ import annotations

import asyncio
import time

import pytest

from director_ai.core.batch import BatchProcessor, BatchResult
from director_ai.core.exceptions import ValidationError
from director_ai.core.metrics import metrics
from director_ai.core.types import CoherenceScore, ReviewResult


class DeterministicAgent:
    def process(self, prompt: str) -> ReviewResult:
        if prompt == "explode":
            raise RuntimeError("model exploded")
        return ReviewResult(
            output=f"processed:{prompt}",
            coherence=CoherenceScore(
                score=0.9,
                approved=True,
                h_logical=0.05,
                h_factual=0.05,
                warning=False,
            ),
            halted=False,
            candidates_evaluated=1,
        )


class DeterministicReviewer:
    def review(self, prompt: str, response: str) -> tuple[bool, CoherenceScore]:
        if response == "explode":
            raise ValueError("bad response")
        approved = prompt in response
        return approved, CoherenceScore(
            score=0.8 if approved else 0.2,
            approved=approved,
            h_logical=0.1 if approved else 0.8,
            h_factual=0.1 if approved else 0.8,
            warning=not approved,
        )


class NativeMetricsReviewer:
    def review_batch(
        self,
        items: list[tuple[str, str]],
        tenant_id: str = "",
    ) -> list[tuple[bool, CoherenceScore]]:
        results = []
        for index, _item in enumerate(items):
            approved = index == 0
            score = 0.91 if approved else 0.21
            results.append(
                (
                    approved,
                    CoherenceScore(
                        score=score,
                        approved=approved,
                        h_logical=1.0 - score,
                        h_factual=1.0 - score,
                        warning=not approved,
                    ),
                )
            )
        return results

    def review(self, prompt: str, response: str) -> tuple[bool, CoherenceScore]:
        approved = prompt in response
        return approved, CoherenceScore(
            score=0.9 if approved else 0.2,
            approved=approved,
            h_logical=0.1 if approved else 0.8,
            h_factual=0.1 if approved else 0.8,
            warning=not approved,
        )


class SlowAgent:
    def process(self, prompt: str, tenant_id: str = "") -> ReviewResult:
        del prompt, tenant_id
        time.sleep(0.05)
        return ReviewResult(output="late", halted=False)


class SlowReviewer:
    def review(self, prompt: str, response: str) -> tuple[bool, CoherenceScore]:
        del prompt, response
        time.sleep(0.05)
        return True, CoherenceScore(
            score=0.9,
            approved=True,
            h_logical=0.1,
            h_factual=0.1,
        )


class NoCoherenceAgent:
    def process(self, prompt: str, tenant_id: str = "") -> ReviewResult:
        del tenant_id
        return ReviewResult(
            output=f"processed:{prompt}",
            coherence=None,
            halted=False,
            candidates_evaluated=1,
        )


def test_process_batch_preserves_success_counts_and_ordered_results() -> None:
    processor = BatchProcessor(DeterministicAgent(), max_concurrency=2)

    result = processor.process_batch(["alpha", "beta", "gamma"])

    assert isinstance(result, BatchResult)
    assert result.total == 3
    assert result.succeeded == 3
    assert result.failed == 0
    assert [item.output for item in result.results] == [
        "processed:alpha",
        "processed:beta",
        "processed:gamma",
    ]


def test_process_batch_records_backend_exceptions_without_losing_successes() -> None:
    processor = BatchProcessor(DeterministicAgent(), max_concurrency=1)

    result = processor.process_batch(["alpha", "explode", "beta"])

    assert result.total == 3
    assert result.succeeded == 2
    assert result.failed == 1
    assert [item.output for item in result.results] == [
        "processed:alpha",
        "processed:beta",
    ]
    assert result.errors == [(1, "model exploded")]


def test_process_batch_can_skip_metrics_and_accept_missing_coherence() -> None:
    metrics.reset()
    processor = BatchProcessor(NoCoherenceAgent(), max_concurrency=1)

    result = processor.process_batch(["alpha"], record_metrics=False)
    batch_telemetry = metrics.get_metrics()
    assert batch_telemetry["histograms"]["batch_size"]["count"] == 0

    direct = processor._process_one(0, "beta")
    telemetry = metrics.get_metrics()

    assert result.succeeded == 1
    assert result.results[0].coherence is None
    assert direct.output == "processed:beta"
    assert telemetry["counters"]["reviews_total"]["total"] == 1.0
    assert telemetry["histograms"]["coherence_score"]["count"] == 0


def test_review_batch_records_approved_rejected_and_failed_items() -> None:
    processor = BatchProcessor(DeterministicReviewer(), max_concurrency=1)

    result = processor.review_batch(
        [
            ("alpha", "alpha is present"),
            ("beta", "no matching term"),
            ("gamma", "explode"),
        ],
    )

    assert result.total == 3
    assert result.succeeded == 2
    assert result.failed == 1
    assert [approved for approved, _score in result.results] == [True, False]
    assert result.errors == [(2, "bad response")]


def test_process_batch_records_item_timeouts() -> None:
    processor = BatchProcessor(SlowAgent(), max_concurrency=1, item_timeout=0.001)

    result = processor.process_batch(["alpha"])

    assert result.total == 1
    assert result.succeeded == 0
    assert result.failed == 1
    assert result.errors == [(0, "item timeout")]
    assert result.results == []


def test_review_batch_records_item_timeouts() -> None:
    processor = BatchProcessor(SlowReviewer(), max_concurrency=1, item_timeout=0.001)

    result = processor.review_batch([("alpha", "alpha")])

    assert result.total == 1
    assert result.succeeded == 0
    assert result.failed == 1
    assert result.errors == [(0, "item timeout")]
    assert result.results == []


def test_native_review_batch_none_result_records_failure() -> None:
    class NativeNoneReviewer:
        def review_batch(self, items, tenant_id: str = ""):
            del tenant_id
            return [
                None,
                (
                    True,
                    CoherenceScore(
                        score=0.93,
                        approved=True,
                        h_logical=0.03,
                        h_factual=0.04,
                    ),
                ),
            ][: len(items)]

        def review(self, prompt: str, response: str):
            raise AssertionError("native path should not fall back")

    processor = BatchProcessor(NativeNoneReviewer(), max_concurrency=1)

    result = processor.review_batch([("missing", "missing"), ("ok", "ok")])

    assert result.succeeded == 1
    assert result.failed == 1
    assert result.errors == [(0, "scorer returned None")]
    assert len(result.results) == 1


def test_native_review_batch_invalid_result_falls_back_to_per_item() -> None:
    class InvalidNativeReviewer:
        def review_batch(self, items, tenant_id: str = ""):
            del items, tenant_id
            return []

        def review(self, prompt: str, response: str):
            approved = prompt in response
            return approved, CoherenceScore(
                score=0.9 if approved else 0.2,
                approved=approved,
                h_logical=0.1 if approved else 0.8,
                h_factual=0.1 if approved else 0.8,
            )

    processor = BatchProcessor(InvalidNativeReviewer(), max_concurrency=1)

    result = processor.review_batch([("alpha", "alpha"), ("beta", "miss")])

    assert result.total == 2
    assert result.succeeded == 2
    assert [approved for approved, _score in result.results] == [True, False]


def test_review_one_uses_nested_scorer_and_rejects_missing_reviewer() -> None:
    class Wrapper:
        scorer = DeterministicReviewer()

    wrapped = BatchProcessor(Wrapper(), max_concurrency=1)
    missing = BatchProcessor(object(), max_concurrency=1)

    approved, score = wrapped._review_one(0, "alpha", "alpha is present")

    assert approved is True
    assert score.score == 0.8
    with pytest.raises(AttributeError, match="backend has no review"):
        missing._review_one(0, "alpha", "alpha")


def test_review_one_falls_back_when_reviewer_has_no_tenant_parameter() -> None:
    processor = BatchProcessor(DeterministicReviewer(), max_concurrency=1)

    approved, score = processor._review_one(
        0,
        "alpha",
        "alpha is present",
        tenant_id="tenant-a",
    )

    assert approved is True
    assert score.score == 0.8


def test_review_one_can_skip_operational_metrics() -> None:
    metrics.reset()
    processor = BatchProcessor(DeterministicReviewer(), max_concurrency=1)

    approved, score = processor._review_one(
        0,
        "alpha",
        "alpha is present",
        record_metrics=False,
    )
    telemetry = metrics.get_metrics()

    assert approved is True
    assert score.score == 0.8
    assert telemetry["counters"]["reviews_total"]["total"] == 0.0


def test_async_native_review_batch_records_operational_metrics_once() -> None:
    metrics.reset()
    processor = BatchProcessor(NativeMetricsReviewer(), max_concurrency=1)

    result = asyncio.run(
        processor.review_batch_async(
            [
                ("alpha", "alpha is present"),
                ("beta", "no matching term"),
            ],
        )
    )
    telemetry = metrics.get_metrics()

    assert result.total == 2
    assert result.succeeded == 2
    assert telemetry["counters"]["reviews_total"]["total"] == 2.0
    assert telemetry["counters"]["reviews_approved"]["total"] == 1.0
    assert telemetry["counters"]["reviews_rejected"]["total"] == 1.0
    assert telemetry["histograms"]["coherence_score"]["count"] == 2
    assert telemetry["histograms"]["batch_size"]["count"] == 1
    assert telemetry["histograms"]["batch_size"]["total"] == 2.0


def test_async_process_batch_records_errors_and_timeouts() -> None:
    failing = BatchProcessor(DeterministicAgent(), max_concurrency=1)
    slow = BatchProcessor(SlowAgent(), max_concurrency=1, item_timeout=0.001)

    failed = asyncio.run(failing.process_batch_async(["alpha", "explode"]))
    timed_out = asyncio.run(slow.process_batch_async(["alpha"]))

    assert failed.succeeded == 1
    assert failed.failed == 1
    assert failed.errors == [(1, "model exploded")]
    assert timed_out.succeeded == 0
    assert timed_out.failed == 1
    assert timed_out.errors == [(0, "item timeout")]


def test_async_review_batch_falls_back_after_native_failure() -> None:
    class FailingNativeReviewer:
        def review_batch(self, items, tenant_id: str = ""):
            del items, tenant_id
            raise RuntimeError("coalesced scorer unavailable")

        def review(self, prompt: str, response: str):
            approved = prompt in response
            return approved, CoherenceScore(
                score=0.9 if approved else 0.2,
                approved=approved,
                h_logical=0.1 if approved else 0.8,
                h_factual=0.1 if approved else 0.8,
            )

    processor = BatchProcessor(FailingNativeReviewer(), max_concurrency=1)

    result = asyncio.run(
        processor.review_batch_async([("alpha", "alpha"), ("beta", "miss")])
    )

    assert result.total == 2
    assert result.succeeded == 2
    assert [approved for approved, _score in result.results] == [True, False]


def test_async_review_batch_falls_back_when_wrapper_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = BatchProcessor(NativeMetricsReviewer(), max_concurrency=1)

    def broken_review_batch(*_args, **_kwargs):
        raise RuntimeError("wrapper unavailable")

    monkeypatch.setattr(processor, "review_batch", broken_review_batch)

    result = asyncio.run(
        processor.review_batch_async([("alpha", "alpha"), ("beta", "miss")])
    )

    assert result.total == 2
    assert result.succeeded == 2
    assert [approved for approved, _score in result.results] == [True, False]


def test_async_review_batch_fallback_can_skip_metrics() -> None:
    metrics.reset()
    processor = BatchProcessor(DeterministicReviewer(), max_concurrency=1)

    result = asyncio.run(
        processor.review_batch_async(
            [("alpha", "alpha")],
            record_metrics=False,
        )
    )
    telemetry = metrics.get_metrics()

    assert result.total == 1
    assert result.succeeded == 1
    assert telemetry["histograms"]["batch_size"]["count"] == 0
    assert telemetry["counters"]["reviews_total"]["total"] == 0.0


def test_async_review_batch_records_errors_and_timeouts() -> None:
    failing = BatchProcessor(DeterministicReviewer(), max_concurrency=1)
    slow = BatchProcessor(SlowReviewer(), max_concurrency=1, item_timeout=0.001)

    failed = asyncio.run(
        failing.review_batch_async([("alpha", "alpha"), ("beta", "explode")])
    )
    timed_out = asyncio.run(slow.review_batch_async([("alpha", "alpha")]))

    assert failed.succeeded == 1
    assert failed.failed == 1
    assert failed.errors == [(1, "bad response")]
    assert timed_out.succeeded == 0
    assert timed_out.failed == 1
    assert timed_out.errors == [(0, "item timeout")]


@pytest.mark.parametrize(
    ("max_concurrency", "item_timeout", "message"),
    [
        (0, 1.0, "max_concurrency"),
        (1, 0.0, "item_timeout"),
        (1, -1.0, "item_timeout"),
    ],
)
def test_constructor_rejects_invalid_execution_bounds(
    max_concurrency: int,
    item_timeout: float,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        BatchProcessor(
            DeterministicAgent(),
            max_concurrency=max_concurrency,
            item_timeout=item_timeout,
        )


def test_async_processing_rejects_invalid_concurrency_override() -> None:
    processor = BatchProcessor(DeterministicAgent(), max_concurrency=1)

    with pytest.raises(ValidationError, match="max_concurrency"):
        asyncio.run(processor.process_batch_async(["alpha"], max_concurrency=0))


def test_async_review_rejects_invalid_concurrency_override() -> None:
    processor = BatchProcessor(DeterministicReviewer(), max_concurrency=1)

    with pytest.raises(ValidationError, match="max_concurrency"):
        asyncio.run(
            processor.review_batch_async(
                [("alpha", "alpha is present")],
                max_concurrency=0,
            ),
        )
