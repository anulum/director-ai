# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Streaming Overhead Benchmark

"""Measure tokens/sec with vs without guard at cadences 1, 4, 8, adaptive.

Usage::

    python -m benchmarks.streaming_overhead_bench
"""

from __future__ import annotations

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
N_TOKENS = 200
ITERATIONS = 5


def _make_tokens(n: int) -> list[str]:
    return [f"tok{i} " for i in range(n)]


def _run_baseline(n_tokens: int, iterations: int) -> dict:
    """Raw iteration, no scoring."""
    tokens = _make_tokens(n_tokens)
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        for _ in tokens:
            pass
        times.append(time.perf_counter() - t0)
    avg_s = sum(times) / len(times)
    return {
        "cadence": "none",
        "tokens_per_sec": round(n_tokens / avg_s) if avg_s > 0 else 0,
        "wall_ms": round(avg_s * 1000, 2),
        "callbacks": 0,
    }


def _run_guard(cadence_label: str, n_tokens: int, iterations: int, **kwargs) -> dict:
    """StreamingKernel + heuristic CoherenceScorer at given cadence."""
    from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

    store = GroundTruthStore()
    store.add("sky", "The sky is blue due to Rayleigh scattering")
    scorer = CoherenceScorer(threshold=0.3, ground_truth_store=store, use_nli=False)

    kernel = StreamingKernel(
        hard_limit=0.1,
        window_size=10,
        window_threshold=0.15,
        trend_window=5,
        trend_threshold=0.3,
        soft_limit=0.15,
        **kwargs,
    )

    tokens = _make_tokens(n_tokens)
    times = []
    last_cb_count = 0

    for _ in range(iterations):
        cb_count = 0
        accumulated = ""

        def coherence_cb(token, _s=scorer):
            nonlocal cb_count, accumulated
            cb_count += 1
            accumulated += token
            _, sc = _s.review("sky", accumulated)
            return sc.score

        t0 = time.perf_counter()
        kernel.stream_tokens(iter(tokens), coherence_cb)
        times.append(time.perf_counter() - t0)
        last_cb_count = cb_count
        kernel._active = True
        accumulated = ""

    avg_s = sum(times) / len(times)
    return {
        "cadence": cadence_label,
        "tokens_per_sec": round(n_tokens / avg_s) if avg_s > 0 else 0,
        "wall_ms": round(avg_s * 1000, 2),
        "callbacks": last_cb_count,
    }


def _run_delta() -> dict:
    """Hallucination reduction: catch_rate with guard vs without."""
    from benchmarks.regression_suite import _E2E_DELTA_SAMPLES
    from director_ai.core import CoherenceScorer, GroundTruthStore

    tp = fn = 0
    for prompt, response, facts, is_hallucinated in _E2E_DELTA_SAMPLES:
        if not is_hallucinated:
            continue
        store = GroundTruthStore()
        for k, v in facts.items():
            store.add(k, v)
        scorer = CoherenceScorer(threshold=0.4, ground_truth_store=store, use_nli=False)
        approved, _ = scorer.review(prompt, response)
        if not approved:
            tp += 1
        else:
            fn += 1

    catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"catch_rate": round(catch_rate, 3), "tp": tp, "fn": fn}


def main():
    print("=" * 65)
    print("  Director-AI Streaming Overhead Benchmark")
    print("=" * 65)

    baseline = _run_baseline(N_TOKENS, ITERATIONS)
    cadence_1 = _run_guard("1", N_TOKENS, ITERATIONS, score_every_n=1)
    cadence_4 = _run_guard("4", N_TOKENS, ITERATIONS, score_every_n=4)
    cadence_8 = _run_guard("8", N_TOKENS, ITERATIONS, score_every_n=8)
    cadence_a = _run_guard(
        "adaptive",
        N_TOKENS,
        ITERATIONS,
        score_every_n=1,
        adaptive=True,
        max_cadence=8,
    )

    rows = [baseline, cadence_1, cadence_4, cadence_8, cadence_a]

    # Compute overhead relative to baseline
    base_ms = baseline["wall_ms"]
    for row in rows:
        if row["cadence"] == "none":
            row["overhead"] = "—"
        elif base_ms > 0:
            pct = ((row["wall_ms"] - base_ms) / base_ms) * 100
            row["overhead"] = f"+{pct:.0f}%"
        else:
            row["overhead"] = "N/A"

    cols = ["Cadence", "Tokens/s", "Wall (ms)", "Overhead", "Callbacks"]
    header = "  ".join(f"{c:>10}" for c in cols)
    sep = "  ".join("─" * 10 for _ in cols)
    print(f"\n{header}")
    print(sep)
    for row in rows:
        print(
            f"{row['cadence']:<10} {row['tokens_per_sec']:>10,} {row['wall_ms']:>10} "
            f"{row['overhead']:>10} {row['callbacks']:>10}",
        )

    delta = _run_delta()
    print(
        f"\nHallucination delta: catch_rate={delta['catch_rate']:.1%} "
        f"(tp={delta['tp']} fn={delta['fn']})",
    )

    results = {"throughput": rows, "delta": delta}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "streaming_overhead.json"
    path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
