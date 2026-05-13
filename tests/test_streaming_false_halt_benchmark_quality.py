# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
# Director-Class AI -- streaming halt-quality benchmark contract

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.streaming_false_halt_bench import (
    BAD_PASSAGES,
    GOOD_PASSAGES,
    _expected_halt_index,
    _halt_quality_metrics,
    _make_callbacks,
    run_benchmark,
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


def test_streaming_callbacks_treat_argument_as_accumulated_text() -> None:
    calls: list[tuple[str, str]] = []

    class RecordingScorer:
        def review(self, prompt: str, response: str):
            calls.append((prompt, response))
            return False, SimpleNamespace(score=0.875, evidence=["grounded"])

    coherence_cb, evidence_cb = _make_callbacks(
        RecordingScorer(),
        "Water science prompt",
    )

    assert coherence_cb("Water boils") == 0.5
    assert calls == []

    accumulated = "Water boils at 100 degrees Celsius"
    assert coherence_cb(accumulated) == 0.875
    assert calls == [("Water science prompt", accumulated)]

    assert evidence_cb(accumulated) == "score=0.875 chunks=['grounded']"
    assert calls[-1] == ("Water science prompt", accumulated)


def test_run_benchmark_resets_kernel_between_passages(monkeypatch) -> None:
    import benchmarks.streaming_false_halt_bench as bench
    import director_ai.core as core

    kernels: list[FakeStreamingKernel] = []

    class FakeGroundTruthStore:
        def add(self, _key: str, _value: str) -> None:
            return None

    class FakeScorer:
        def __init__(self, **_kwargs) -> None:
            return None

        def review(self, _prompt: str, _response: str):
            return False, SimpleNamespace(score=0.99, evidence=[])

    class FakeStreamingKernel:
        def __init__(self, **_kwargs) -> None:
            self.reset_calls = 0
            self.stream_calls = 0
            kernels.append(self)

        def stream_tokens(self, tokens, coherence_callback, evidence_callback=None):
            text = "".join(tokens)
            coherence_callback(text)
            if evidence_callback is not None:
                evidence_callback(text)
            self.stream_calls += 1
            return SimpleNamespace(
                halted=False,
                halt_reason="",
                halt_index=-1,
                halt_evidence=None,
                token_count=len(text.split()),
                avg_coherence=0.99,
                min_coherence=0.99,
            )

        def reset_state(self) -> None:
            self.reset_calls += 1

    monkeypatch.setattr(core, "GroundTruthStore", FakeGroundTruthStore)
    monkeypatch.setattr(core, "CoherenceScorer", FakeScorer)
    monkeypatch.setattr(core, "StreamingKernel", FakeStreamingKernel)
    monkeypatch.setattr(
        bench,
        "GOOD_PASSAGES",
        [("good", {"fact": "grounded fact"}, "Grounded fact remains stable.")],
    )
    monkeypatch.setattr(
        bench,
        "BAD_PASSAGES",
        [
            (
                "bad",
                {"fact": "grounded fact"},
                "Grounded fact flips at the contradiction.",
                "flips",
            ),
        ],
    )

    output = run_benchmark(use_nli=False)

    assert len(kernels) == 1
    assert kernels[0].stream_calls == 2
    assert kernels[0].reset_calls == 2
    assert output["total_passages"] == 1
    assert output["halt_quality"]["false_negatives"] == 1
