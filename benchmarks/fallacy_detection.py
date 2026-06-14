# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — fallacy detection accuracy + Rust-vs-Python benchmark

"""Measure informal-fallacy detection and compare the Rust and Python backends.

Two things are reported:

* **Detection accuracy** — a labelled set of sentences, each tagged with the
  fallacy family it contains (or ``none`` for clean control sentences), is run
  through ``detect_fallacies``; we report whether the expected family is flagged
  (and clean sentences stay clean), plus false-positive rate on the controls.
* **Backend comparison** — the marker scan is run with the Rust
  ``rust_detect_fallacies`` kernel and with the pure-Python regex pass over the
  corpus; the two must return identical matches (parity) and both throughputs are
  recorded so the dispatch is chosen on measured speed.

Output: ``benchmarks/results/fallacy_detection.json``. Reproduce with
``python -m benchmarks.fallacy_detection``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.verification import fallacy_detector as fd
from director_ai.core.verification.fallacy_detector import detect_fallacies

# (text, expected_family or "none")
_LABELLED: list[tuple[str, str]] = [
    ("You're just biased, so the point is wrong.", "ad_hominem"),
    ("He is incompetent and cannot be trusted on this.", "ad_hominem"),
    ("Because experts say so, the policy is correct.", "appeal_to_authority"),
    ("Since scientists agree, we need not check further.", "appeal_to_authority"),
    ("Everyone knows this is the right call.", "bandwagon"),
    ("Nobody believes the alternative anymore.", "bandwagon"),
    ("You're either with us or against us.", "false_dichotomy"),
    ("There are only two options here, really.", "false_dichotomy"),
    ("This proves that all of them are corrupt.", "hasty_generalization"),
    ("It shows that nobody can be trusted.", "hasty_generalization"),
    ("This will inevitably lead to total collapse.", "slippery_slope"),
    ("Next thing you know, everything is banned.", "slippery_slope"),
    ("Think of the children before you vote.", "appeal_to_emotion"),
    ("You should be ashamed for even asking.", "appeal_to_emotion"),
    ("Sales rose after the ad, therefore the ad caused it.", "post_hoc"),
    # Clean controls — must produce no fallacy.
    ("The capital of France is Paris.", "none"),
    ("Water boils at 100 degrees Celsius at sea level.", "none"),
    ("The report is due on Tuesday afternoon.", "none"),
    ("Three studies measured the same effect size.", "none"),
    ("The function returns a sorted list of integers.", "none"),
]


def detection_accuracy() -> dict:
    hits = 0
    fp_controls = 0
    n_controls = 0
    for text, expected in _LABELLED:
        families = detect_fallacies(text).types
        if expected == "none":
            n_controls += 1
            if families:
                fp_controls += 1
            else:
                hits += 1
        elif expected in families:
            hits += 1
    return {
        "n": len(_LABELLED),
        "accuracy": round(hits / len(_LABELLED), 4),
        "control_false_positive_rate": round(fp_controls / n_controls, 4) if n_controls else 0.0,
        "n_controls": n_controls,
    }


def backend_comparison(repeats: int) -> dict:
    corpus = [t for t, _ in _LABELLED]
    rust_available = fd._RUST_FALLACY and fd.rust_detect_fallacies is not None

    rust_out = {t: fd._scan(t) for t in corpus}
    saved_flag, saved_fn = fd._RUST_FALLACY, fd.rust_detect_fallacies

    def _tps() -> float:
        t0 = time.perf_counter()
        for _ in range(repeats):
            for text in corpus:
                fd._scan(text)
        elapsed = time.perf_counter() - t0
        return (len(corpus) * repeats) / elapsed if elapsed else 0.0

    rust_tps = _tps() if rust_available else 0.0
    fd._RUST_FALLACY, fd.rust_detect_fallacies = False, None
    try:
        py_out = {t: fd._scan(t) for t in corpus}
        py_tps = _tps()
    finally:
        fd._RUST_FALLACY, fd.rust_detect_fallacies = saved_flag, saved_fn

    parity = rust_out == py_out
    return {
        "rust_available": rust_available,
        "parity_rust_equals_python": parity,
        "rust_texts_per_sec": round(rust_tps, 1),
        "python_texts_per_sec": round(py_tps, 1),
        "rust_speedup": round(rust_tps / py_tps, 3) if (py_tps and rust_available) else None,
        "fastest": "rust" if (rust_available and rust_tps >= py_tps) else "python",
    }


def run(*, repeats: int = 3000) -> dict:
    return {
        "benchmark": "fallacy_detection",
        "detection": detection_accuracy(),
        "backends": backend_comparison(repeats),
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "fallacy_detection.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    d = result["detection"]
    b = result["backends"]
    print(f"\nFallacy detection (n={d['n']}):")
    print(
        f"  accuracy={d['accuracy']:.2f} "
        f"control_FPR={d['control_false_positive_rate']:.2f}"
    )
    print(
        f"  backend rust_available={b['rust_available']} "
        f"parity={b['parity_rust_equals_python']}"
    )
    print(
        f"  throughput rust={b['rust_texts_per_sec']:.0f}/s "
        f"python={b['python_texts_per_sec']:.0f}/s "
        f"speedup={b['rust_speedup']} fastest={b['fastest']}"
    )


if __name__ == "__main__":
    main()
