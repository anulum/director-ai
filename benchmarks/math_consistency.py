# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — arithmetic verification accuracy + Rust-vs-Python benchmark

"""Measure chain-of-thought arithmetic verification and compare backends.

Two things are reported:

* **Detection accuracy** — a labelled set of reasoning snippets, each tagged with
  whether its stated equation is correct, is run through ``verify_arithmetic``;
  we report how often the verdict matches the label, plus error precision/recall.
* **Backend comparison** — the ``_eval_arithmetic`` evaluator is run with the Rust
  ``rust_eval_arithmetic`` kernel and with the pure-Python ``ast`` fallback over
  an expression corpus; the two must agree exactly (bit-for-bit parity, NaN
  included) and both throughputs are recorded so the dispatch is chosen on
  measured speed.

Output: ``benchmarks/results/math_consistency.json``. Reproduce with
``python -m benchmarks.math_consistency``.
"""

from __future__ import annotations

import json
import math
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.verification import math_consistency as mc
from director_ai.core.verification.math_consistency import verify_arithmetic

# (text, equation_is_wrong) — one stated equation each.
_LABELLED: list[tuple[str, bool]] = [
    ("Adding them, 3 + 4 = 7, so the total is seven.", False),
    ("We multiply 12 × 5 = 60 to get the count.", False),
    ("Dividing, 100 / 4 = 25 gives the share.", False),
    ("So (120 - 20) / 4 = 25, the per-unit cost.", False),
    ("The revenue 1,200 + 800 = 2,000 in total.", False),
    ("Then 2 + 2 is equal to 4, trivially.", False),
    ("Summing, 3 + 4 = 8, the subtotal.", True),
    ("Hence 12 × 5 = 55 widgets in all.", True),
    ("Dividing, 100 / 4 = 20 per group.", True),
    ("So (120 - 20) / 4 = 30, the cost each.", True),
    ("Revenue 1,200 + 800 = 1,900 overall.", True),
    ("Therefore 7 * 8 = 54 in the grid.", True),
]

_EXPR_CORPUS = [
    "3 + 4",
    "2 + 3 * 4",
    "(120 - 20) / 4",
    "12 × 5",
    "100 ÷ 4",
    "10 / 3",
    "1,200 + 800",
    "2 + 2 * 2 - 1",
    "1 / 0",
    "-5 + 8",
]


def detection_accuracy() -> dict:
    tp = fp = tn = fn = 0
    for text, wrong in _LABELLED:
        detected = not verify_arithmetic(text).valid
        if wrong and detected:
            tp += 1
        elif wrong and not detected:
            fn += 1
        elif not wrong and detected:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "n": len(_LABELLED),
        "accuracy": round((tp + tn) / len(_LABELLED), 4),
        "error_precision": round(precision, 4),
        "error_recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def backend_comparison(repeats: int) -> dict:
    rust_available = mc._RUST_MATH and mc.rust_eval_arithmetic is not None
    rust_vals = [mc._eval_arithmetic(e) for e in _EXPR_CORPUS]

    def _tps() -> float:
        t0 = time.perf_counter()
        for _ in range(repeats):
            for expr in _EXPR_CORPUS:
                mc._eval_arithmetic(expr)
        elapsed = time.perf_counter() - t0
        return (len(_EXPR_CORPUS) * repeats) / elapsed if elapsed else 0.0

    rust_tps = _tps() if rust_available else 0.0
    saved_flag, saved_fn = mc._RUST_MATH, mc.rust_eval_arithmetic
    mc._RUST_MATH, mc.rust_eval_arithmetic = False, None
    try:
        py_vals = [mc._eval_arithmetic(e) for e in _EXPR_CORPUS]
        py_tps = _tps()
    finally:
        mc._RUST_MATH, mc.rust_eval_arithmetic = saved_flag, saved_fn

    parity = all(
        (math.isnan(r) and math.isnan(p)) or r == p
        for r, p in zip(rust_vals, py_vals, strict=True)
    )
    return {
        "rust_available": rust_available,
        "parity_rust_equals_python": parity,
        "rust_exprs_per_sec": round(rust_tps, 1),
        "python_exprs_per_sec": round(py_tps, 1),
        "rust_speedup": round(rust_tps / py_tps, 3)
        if (py_tps and rust_available)
        else None,
        "fastest": "rust" if (rust_available and rust_tps >= py_tps) else "python",
    }


def run(*, repeats: int = 5000) -> dict:
    return {
        "benchmark": "math_consistency",
        "detection": detection_accuracy(),
        "backends": backend_comparison(repeats),
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "math_consistency.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    d = result["detection"]
    b = result["backends"]
    print(f"\nMath consistency (n={d['n']}):")
    print(
        f"  detection accuracy={d['accuracy']:.2f} "
        f"error P={d['error_precision']:.2f} R={d['error_recall']:.2f}"
    )
    print(
        f"  backend rust_available={b['rust_available']} "
        f"parity={b['parity_rust_equals_python']}"
    )
    print(
        f"  throughput rust={b['rust_exprs_per_sec']:.0f}/s "
        f"python={b['python_exprs_per_sec']:.0f}/s "
        f"speedup={b['rust_speedup']} fastest={b['fastest']}"
    )


if __name__ == "__main__":
    main()
