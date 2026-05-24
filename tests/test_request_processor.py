# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — BatchProcessor public contract
"""Behavioural tests for the BatchProcessor request-processing contract."""

from __future__ import annotations

import asyncio

import pytest

from director_ai.core.batch import BatchProcessor, BatchResult
from director_ai.core.exceptions import ValidationError
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
