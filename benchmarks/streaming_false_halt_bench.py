# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming False-Halt Rate Benchmark

"""Measures false-halt rate: how often StreamingKernel incorrectly halts
on known-good text that should pass without interruption.

Feeds factually correct, coherent passages token-by-token through the
StreamingKernel with CoherenceScorer. A false halt is any halt on a
passage that should complete cleanly.

Usage::

    python -m benchmarks.streaming_false_halt_bench
    python -m benchmarks.streaming_false_halt_bench --nli
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

# The corpus (GOOD_PASSAGES / BAD_PASSAGES) lives in its own data module;
# re-exported here so ``benchmarks.streaming_false_halt_bench`` keeps its
# import surface.
from benchmarks.streaming_false_halt_corpus import (
    BAD_PASSAGES as BAD_PASSAGES,
)
from benchmarks.streaming_false_halt_corpus import (
    GOOD_PASSAGES as GOOD_PASSAGES,
)

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class FalseHaltResult:
    passage_id: str
    halted: bool
    halt_reason: str
    halt_index: int
    halt_evidence: str | None
    token_count: int
    avg_coherence: float
    min_coherence: float
    duration_ms: float


def _tokenize_simple(text: str) -> list[str]:
    """Split text into word-level tokens with leading spaces for join correctness.

    ``StreamingKernel.stream_tokens`` accumulates via ``"".join(tokens)``,
    so every token after the first carries a leading space.
    """
    words = text.split()
    if not words:
        return []
    return [words[0]] + [f" {w}" for w in words[1:]]


def _expected_halt_index(text: str, expected_fragment: str) -> int:
    """Return the first token index containing the labelled contradiction."""
    fragment = expected_fragment.strip()
    if not fragment:
        raise ValueError("expected_fragment must not be empty")
    for index, token in enumerate(_tokenize_simple(text)):
        if fragment in token.strip():
            return index
    raise ValueError(f"expected_fragment {expected_fragment!r} not found")


def _halt_quality_metrics(
    good_results: list[dict],
    bad_results: list[dict],
    *,
    token_tolerance: int = 8,
) -> dict[str, float | int]:
    """Compute confusion and token-timing metrics for streaming halt quality."""
    false_positives = sum(1 for result in good_results if result["halted"])
    true_negatives = len(good_results) - false_positives
    true_positives = sum(1 for result in bad_results if result["halted"])
    false_negatives = len(bad_results) - true_positives
    halt_precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    halt_recall = true_positives / len(bad_results) if bad_results else 0.0
    false_halt_rate = false_positives / len(good_results) if good_results else 0.0
    on_time_halts = 0
    halt_latencies: list[int] = []
    for result in bad_results:
        if not result["halted"]:
            continue
        latency = result["halt_index"] - result["expected_halt_index"]
        halt_latencies.append(latency)
        if 0 <= latency <= token_tolerance:
            on_time_halts += 1
    token_accuracy = on_time_halts / true_positives if true_positives else 0.0
    median_latency = statistics.median(halt_latencies) if halt_latencies else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "halt_precision": round(halt_precision, 4),
        "halt_recall": round(halt_recall, 4),
        "false_halt_rate": round(false_halt_rate, 4),
        "token_of_halt_accuracy": round(token_accuracy, 4),
        "median_halt_latency_tokens": round(float(median_latency), 4),
        "token_tolerance": token_tolerance,
    }


def _make_callbacks(scorer, prompt: str):
    """Factory to avoid B023 closure-in-loop binding issues.

    ``coherence_cb`` receives the **accumulated text so far** (not the
    individual token) from ``StreamingKernel.stream_tokens`` and runs it
    through the production :class:`StreamingCoherenceGate`, so the benchmark
    measures the same claim-boundary gating the server's ``agent.stream`` uses
    rather than scoring every half-finished fragment.
    """
    from director_ai.core.runtime.streaming_gate import StreamingCoherenceGate

    gate = StreamingCoherenceGate(lambda text: scorer.review(prompt, text)[1].score)

    def coherence_cb(text: str) -> float:
        return gate.update(text)

    def evidence_cb(text: str) -> str | None:
        _, sc = scorer.review(prompt, text)
        chunks = []
        if hasattr(sc, "evidence") and sc.evidence:
            chunks = sc.evidence
        return f"score={sc.score:.3f}" + (f" chunks={chunks}" if chunks else "")

    return coherence_cb, evidence_cb


def run_benchmark(use_nli: bool = False) -> dict:
    from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

    kernel = StreamingKernel(
        hard_limit=0.10,
        window_size=8,
        window_threshold=0.18,
        trend_window=5,
        trend_threshold=0.30,
        soft_limit=0.15,
    )

    results: list[FalseHaltResult] = []
    good_metrics: list[dict] = []
    bad_metrics: list[dict] = []
    n = len(GOOD_PASSAGES)
    print(f"Passages: {n}  |  NLI: {use_nli}")

    for pid, facts, passage in GOOD_PASSAGES:
        store = GroundTruthStore()
        for k, v in facts.items():
            store.add(k, v)

        scorer = CoherenceScorer(
            threshold=0.3,
            ground_truth_store=store,
            use_nli=use_nli,
        )

        tokens = _tokenize_simple(passage)
        coh_cb, ev_cb = _make_callbacks(scorer, passage[:50])

        t0 = time.perf_counter()
        session = kernel.stream_tokens(
            iter(tokens),
            coh_cb,
            evidence_callback=ev_cb,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        result = FalseHaltResult(
            passage_id=pid,
            halted=session.halted,
            halt_reason=session.halt_reason,
            halt_index=session.halt_index,
            halt_evidence=session.halt_evidence,
            token_count=session.token_count,
            avg_coherence=session.avg_coherence,
            min_coherence=session.min_coherence,
            duration_ms=elapsed,
        )
        results.append(result)
        good_metrics.append({"id": pid, "halted": session.halted})

        # Reset kernel for next passage
        kernel.reset_state()

    bad_results: list[FalseHaltResult] = []
    for pid, facts, passage, expected_fragment in BAD_PASSAGES:
        store = GroundTruthStore()
        for k, v in facts.items():
            store.add(k, v)

        scorer = CoherenceScorer(
            threshold=0.3,
            ground_truth_store=store,
            use_nli=use_nli,
        )

        expected_index = _expected_halt_index(passage, expected_fragment)
        tokens = _tokenize_simple(passage)
        coh_cb, ev_cb = _make_callbacks(scorer, passage[:50])

        t0 = time.perf_counter()
        session = kernel.stream_tokens(
            iter(tokens),
            coh_cb,
            evidence_callback=ev_cb,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        bad_result = FalseHaltResult(
            passage_id=pid,
            halted=session.halted,
            halt_reason=session.halt_reason,
            halt_index=session.halt_index,
            halt_evidence=session.halt_evidence,
            token_count=session.token_count,
            avg_coherence=session.avg_coherence,
            min_coherence=session.min_coherence,
            duration_ms=elapsed,
        )
        bad_results.append(bad_result)
        bad_metrics.append(
            {
                "id": pid,
                "halted": session.halted,
                "halt_index": session.halt_index,
                "expected_halt_index": expected_index,
            },
        )
        kernel.reset_state()

    false_halts = [r for r in results if r.halted]
    quality = _halt_quality_metrics(good_metrics, bad_metrics)
    fh_rate = len(false_halts) / n
    avg_coh = sum(r.avg_coherence for r in results) / n
    avg_ms = sum(r.duration_ms for r in results) / n

    print(f"\n{'=' * 55}")
    print("  Streaming False-Halt Benchmark")
    print(f"{'=' * 55}")
    print(f"  Passages:     {n}")
    print(f"  False halts:  {len(false_halts)} ({fh_rate:.1%})")
    print(f"  Halt precision: {quality['halt_precision']:.1%}")
    print(f"  Halt recall:    {quality['halt_recall']:.1%}")
    print(f"  Token accuracy: {quality['token_of_halt_accuracy']:.1%}")
    print(f"  Avg coherence: {avg_coh:.3f}")
    print(f"  Avg latency:  {avg_ms:.2f} ms/passage")
    print(f"{'=' * 55}")

    if false_halts:
        print(f"\n  False halts ({len(false_halts)}):")
        for fh in false_halts:
            print(f"    {fh.passage_id}: {fh.halt_reason}")
            print(
                f"      token {fh.halt_index}/{fh.token_count}"
                f"  avg_coh={fh.avg_coherence:.3f}",
            )
            if fh.halt_evidence:
                ev_str = str(fh.halt_evidence)[:120]
                print(f"      evidence: {ev_str.encode('ascii', 'replace').decode()}")

    output = {
        "benchmark": "streaming_false_halt",
        "nli": use_nli,
        "total_passages": n,
        "false_halts": len(false_halts),
        "false_halt_rate": round(fh_rate, 4),
        "halt_quality": quality,
        "avg_coherence": round(avg_coh, 4),
        "avg_latency_ms": round(avg_ms, 2),
        "per_passage": [
            {
                "id": r.passage_id,
                "halted": r.halted,
                "halt_reason": r.halt_reason,
                "halt_index": r.halt_index,
                "halt_evidence": r.halt_evidence,
                "token_count": r.token_count,
                "avg_coherence": round(r.avg_coherence, 4),
                "min_coherence": round(r.min_coherence, 4),
                "duration_ms": round(r.duration_ms, 3),
            }
            for r in results
        ],
        "bad_passages": [
            {
                "id": r.passage_id,
                "halted": r.halted,
                "halt_reason": r.halt_reason,
                "halt_index": r.halt_index,
                "expected_halt_index": bad_metrics[index]["expected_halt_index"],
                "halt_evidence": r.halt_evidence,
                "token_count": r.token_count,
                "avg_coherence": round(r.avg_coherence, 4),
                "min_coherence": round(r.min_coherence, 4),
                "duration_ms": round(r.duration_ms, 3),
            }
            for index, r in enumerate(bad_results)
        ],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "nli" if use_nli else "heuristic"
    path = RESULTS_DIR / f"streaming_false_halt_{tag}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")
    return output


def run_window_sweep(use_nli: bool = False) -> dict:
    """Sweep window_size and measure false-halt / correct-halt rates."""
    from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

    window_sizes = [3, 5, 8, 10, 15, 20]
    sweep_results = []

    for ws in window_sizes:
        kernel = StreamingKernel(
            hard_limit=0.10,
            window_size=ws,
            window_threshold=0.18,
            trend_window=5,
            trend_threshold=0.30,
            soft_limit=0.15,
        )

        false_halts = 0
        for _pid, facts, passage in GOOD_PASSAGES:
            store = GroundTruthStore()
            for k, v in facts.items():
                store.add(k, v)
            scorer = CoherenceScorer(
                threshold=0.3,
                ground_truth_store=store,
                use_nli=use_nli,
            )
            tokens = _tokenize_simple(passage)
            coh_cb, ev_cb = _make_callbacks(scorer, passage[:50])
            session = kernel.stream_tokens(iter(tokens), coh_cb, ev_cb)
            if session.halted:
                false_halts += 1
            kernel.reset_state()

        correct_halts = 0
        halt_coherences = []
        for _pid, facts, passage, _expected_fragment in BAD_PASSAGES:
            store = GroundTruthStore()
            for k, v in facts.items():
                store.add(k, v)
            scorer = CoherenceScorer(
                threshold=0.3,
                ground_truth_store=store,
                use_nli=use_nli,
            )
            tokens = _tokenize_simple(passage)
            coh_cb, ev_cb = _make_callbacks(scorer, passage[:50])
            session = kernel.stream_tokens(iter(tokens), coh_cb, ev_cb)
            if session.halted:
                correct_halts += 1
                halt_coherences.append(session.avg_coherence)
            kernel.reset_state()

        n_good = len(GOOD_PASSAGES)
        n_bad = len(BAD_PASSAGES)
        avg_halt_coh = (
            sum(halt_coherences) / len(halt_coherences) if halt_coherences else 0.0
        )
        sweep_results.append(
            {
                "window_size": ws,
                "false_halt_rate": false_halts / n_good,
                "correct_halt_rate": correct_halts / n_bad if n_bad else 0.0,
                "avg_coherence_at_halt": round(avg_halt_coh, 4),
            },
        )

    print(f"\n{'=' * 65}")
    print("  Window Size Sweep")
    print(f"{'=' * 65}")
    print(
        f"  {'Window':>6} {'FalseHalt%':>10} {'CorrectHalt%':>12} {'AvgCoh@Halt':>12}",
    )
    print(f"  {'-' * 44}")
    for r in sweep_results:
        print(
            f"  {r['window_size']:>6}"
            f" {r['false_halt_rate']:>9.1%}"
            f" {r['correct_halt_rate']:>11.1%}"
            f" {r['avg_coherence_at_halt']:>11.4f}",
        )
    print(f"{'=' * 65}")

    output = {"benchmark": "window_sweep", "nli": use_nli, "results": sweep_results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = "nli" if use_nli else "heuristic"
    path = RESULTS_DIR / f"window_sweep_{tag}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Streaming false-halt rate benchmark",
    )
    parser.add_argument(
        "--nli",
        action="store_true",
        help="Use NLI scorer (requires director-ai[nli])",
    )
    parser.add_argument(
        "--sweep-window",
        action="store_true",
        help="Sweep window_size [3,5,8,10,15,20] and measure halt rates",
    )
    args = parser.parse_args()
    if args.sweep_window:
        run_window_sweep(use_nli=args.nli)
    else:
        run_benchmark(use_nli=args.nli)


if __name__ == "__main__":
    main()
