# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — word-overlap Python/Rust crossover benchmark

"""Find where the Rust ``rust_word_overlap`` kernel overtakes pure Python.

The ``_word_overlap`` heuristic (Jaccard over whitespace-split tokens) is
dispatched to the Rust kernel unconditionally in many modules, but the kernel is
not always the faster path: Python's ``set``/``split`` is highly optimised, and
the FFI marshalling has a fixed per-call cost. This benchmark sweeps the input
size and reports, per word count, the mean Python and Rust per-call time and the
speedup, so each call site can default to the *measured* faster path with a
crossover threshold rather than assuming Rust.

This is a **relative, non-isolated** measurement (same-process Python-vs-Rust on
a shared workstation) intended to pick a runtime default — not a published
absolute performance claim. Output:
``benchmarks/results/word_overlap_crossover.json``. Reproduce with
``python -m benchmarks.word_overlap_crossover``.
"""

from __future__ import annotations

import json
import random
import time

from benchmarks._common import RESULTS_DIR

_WORD_COUNTS = (5, 10, 20, 50, 100, 200, 500, 1000, 2000)
_REPEATS = 4000
_VOCAB = [f"token{i:04d}" for i in range(600)]


def _py_word_overlap(a: str, b: str) -> float:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _make_pair(n_words: int, rng: random.Random) -> tuple[str, str]:
    a = " ".join(rng.choice(_VOCAB) for _ in range(n_words))
    # ~50% shared vocabulary so the Jaccard is non-trivial.
    b = " ".join(rng.choice(_VOCAB) for _ in range(n_words))
    return a, b


def _time(fn, pairs, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        for a, b in pairs:
            fn(a, b)
    return (time.perf_counter() - t0) / (repeats * len(pairs)) * 1e6  # µs/call


def run(*, repeats: int = _REPEATS) -> dict:
    try:
        from backfire_kernel import rust_word_overlap
    except ImportError:
        rust_word_overlap = None

    rng = random.Random(20260615)
    rows = []
    for n in _WORD_COUNTS:
        pairs = [_make_pair(n, rng) for _ in range(8)]
        reps = max(50, repeats // max(1, n))
        py_us = round(_time(_py_word_overlap, pairs, reps), 4)
        if rust_word_overlap is None:
            rows.append({"words": n, "py_us": py_us, "rust_us": None, "speedup": None})
            continue
        rust_us = round(
            _time(lambda a, b: float(rust_word_overlap(a, b)), pairs, reps), 4
        )
        speedup = round(py_us / rust_us, 3) if rust_us else None
        rows.append(
            {"words": n, "py_us": py_us, "rust_us": rust_us, "speedup": speedup}
        )

    # A *sustained* crossover: the smallest size at which Rust is faster and stays
    # faster for every larger size measured. A single faster point (sub-µs noise)
    # does not count. None => no input size where Rust reliably wins.
    sizes_with_rust = [r for r in rows if r["rust_us"] is not None]
    sustained: int | None = None
    for idx, row in enumerate(sizes_with_rust):
        if all(
            r["speedup"] is not None and r["speedup"] > 1.0
            for r in sizes_with_rust[idx:]
        ):
            sustained = row["words"]
            break

    return {
        "benchmark": "word_overlap_crossover",
        "measurement": "relative, non-isolated (Python vs Rust, same process)",
        "kernel_available": rust_word_overlap is not None,
        "sustained_crossover_words": sustained,
        "recommended_default": "rust" if sustained == 0 else "python",
        "rows": rows,
        "note": (
            "sustained_crossover_words is the smallest input size at which Rust is "
            "faster and stays faster for all larger sizes; None means no size where "
            "Rust reliably wins, so the runtime default for these call sites should "
            "be pure Python (their inputs sit in the 10-500-word range)."
        ),
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "word_overlap_crossover.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("\nword_overlap Python/Rust crossover (relative, non-isolated):")
    print(f"  {'words':>6} {'py µs':>10} {'rust µs':>10} {'rust speedup':>13}")
    for row in result["rows"]:
        rust = "n/a" if row["rust_us"] is None else f"{row['rust_us']:.3f}"
        sp = "n/a" if row["speedup"] is None else f"{row['speedup']:.2f}x"
        print(f"  {row['words']:>6} {row['py_us']:>10.3f} {rust:>10} {sp:>13}")
    print(
        f"  sustained crossover: {result['sustained_crossover_words']} words "
        f"-> recommended default: {result['recommended_default']}"
    )


if __name__ == "__main__":
    main()
