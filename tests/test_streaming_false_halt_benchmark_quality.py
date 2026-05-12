# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI -- streaming halt-quality benchmark contract

from __future__ import annotations

from benchmarks.streaming_false_halt_bench import (
    BAD_PASSAGES,
    GOOD_PASSAGES,
    _expected_halt_index,
    _halt_quality_metrics,
)


def test_expected_halt_index_tracks_first_contradictory_fragment() -> None:
    passage_id, _facts, passage, expected_fragment = BAD_PASSAGES[0]

    index = _expected_halt_index(passage, expected_fragment)

    assert passage_id == "wrong_boiling"
    assert index == 3


def test_halt_quality_metrics_report_confusion_and_timing() -> None:
    good_count = len(GOOD_PASSAGES)
    bad_count = len(BAD_PASSAGES)
    good_results = [{"halted": False} for _ in range(good_count)]
    bad_results = [
        {"halted": True, "halt_index": 4, "expected_halt_index": 4},
        {"halted": True, "halt_index": 8, "expected_halt_index": 3},
        {"halted": False, "halt_index": -1, "expected_halt_index": 5},
    ]

    metrics = _halt_quality_metrics(
        good_results,
        bad_results,
        token_tolerance=5,
    )

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 1
    assert metrics["halt_precision"] == 1.0
    assert metrics["halt_recall"] == round(2 / bad_count, 4)
    assert metrics["token_of_halt_accuracy"] == 1.0
    assert metrics["median_halt_latency_tokens"] == 2.5
