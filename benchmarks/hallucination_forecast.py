# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — hallucination-forecast separation + Rust-vs-Python benchmark

"""Measure how well the pre-generation forecaster separates risky prompts from
safe ones, and compare the Rust and Python lexical-overlap backends.

Two things are reported:

* **Separation** — a labelled set of prompts tagged ``high`` (under-specified
  and/or ungrounded) or ``low`` (specific and grounded by a matching fact) is
  scored through :class:`HallucinationForecaster`. We report the mean risk per
  class, the gap between them, the pairwise ranking accuracy (fraction of
  ``low`` < ``high`` prompt pairs ordered correctly — an AUROC equivalent), and
  the recommendation-band accuracy.
* **Backend comparison** — the ``_lexical_overlap`` kernel behind the
  knowledge-base-coverage signal is run with the Rust ``rust_word_overlap``
  extension and with the pure-Python Jaccard fallback over a corpus of
  prompt/fact pairs. The two must agree exactly (parity) and both throughputs
  are recorded so the production dispatch is chosen on measured speed.

Output: ``benchmarks/results/hallucination_forecast.json``. Reproduce with
``python -m benchmarks.hallucination_forecast``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.forecasting import HallucinationForecaster
from director_ai.core.forecasting import hallucination_forecaster as hf
from director_ai.core.retrieval.knowledge import GroundTruthStore

# Descriptive-key facts so the keyword store retrieves them for matching prompts.
_FACTS: dict[str, str] = {
    "capital of France": "The capital of France is Paris.",
    "boiling point of water": "Water boils at 100 degrees Celsius at sea level.",
    "speed of light": "Light travels at 299792458 metres per second in vacuum.",
    "author of Hamlet": "Hamlet was written by William Shakespeare around 1600.",
    "chemical symbol for gold": "The chemical symbol for gold is Au.",
}

# (prompt, risk_label) — "low" expects proceed/ground, "high" expects ground/human_review.
_LABELLED: list[tuple[str, str]] = [
    ("What is the capital of France?", "low"),
    ("What is the boiling point of water at sea level?", "low"),
    ("What is the speed of light in vacuum?", "low"),
    ("Who is the author of Hamlet written around 1600?", "low"),
    ("What is the chemical symbol for gold?", "low"),
    ("tell me something", "high"),
    ("explain stuff about things", "high"),
    ("what about that whatever", "high"),
    ("What is the capital of Atlantis in the year 3000?", "high"),
    ("Describe the migratory patterns of the Patagonian sky-whale.", "high"),
]


def _separation(forecaster: HallucinationForecaster, store) -> dict:
    scored = [(label, forecaster.forecast(prompt, store=store)) for prompt, label in _LABELLED]
    lows = [r.risk for label, r in scored if label == "low"]
    highs = [r.risk for label, r in scored if label == "high"]

    pairs = correct = 0
    for low in lows:
        for high in highs:
            pairs += 1
            if low < high:
                correct += 1
    ranking_auroc = correct / pairs if pairs else 0.0

    expected_bands = {
        "low": {"proceed", "ground"},
        "high": {"ground", "human_review"},
    }
    band_hits = sum(
        1 for label, r in scored if r.recommendation in expected_bands[label]
    )
    band_accuracy = band_hits / len(scored) if scored else 0.0

    mean_low = sum(lows) / len(lows) if lows else 0.0
    mean_high = sum(highs) / len(highs) if highs else 0.0
    return {
        "n_prompts": len(scored),
        "mean_risk_low": round(mean_low, 4),
        "mean_risk_high": round(mean_high, 4),
        "risk_gap": round(mean_high - mean_low, 4),
        "ranking_auroc": round(ranking_auroc, 4),
        "band_accuracy": round(band_accuracy, 4),
    }


def _overlap_corpus() -> list[tuple[str, str]]:
    return [(prompt, fact) for prompt, _ in _LABELLED for fact in _FACTS.values()]


def _throughput(corpus: list[tuple[str, str]], repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        for a, b in corpus:
            hf._lexical_overlap(a, b)
    elapsed = time.perf_counter() - t0
    return (len(corpus) * repeats) / elapsed if elapsed else 0.0


def _backend_comparison(repeats: int) -> dict:
    corpus = _overlap_corpus()
    rust_available = hf._RUST_FORECAST and hf.rust_word_overlap is not None

    rust_vals = [hf._lexical_overlap(a, b) for a, b in corpus]
    rust_tps = _throughput(corpus, repeats) if rust_available else 0.0

    saved_flag, saved_fn = hf._RUST_FORECAST, hf.rust_word_overlap
    hf._RUST_FORECAST, hf.rust_word_overlap = False, None
    try:
        py_vals = [hf._lexical_overlap(a, b) for a, b in corpus]
        py_tps = _throughput(corpus, repeats)
    finally:
        hf._RUST_FORECAST, hf.rust_word_overlap = saved_flag, saved_fn

    parity = all(
        abs(r - p) < 1e-9 for r, p in zip(rust_vals, py_vals, strict=True)
    )
    return {
        "rust_available": rust_available,
        "parity_rust_equals_python": parity,
        "rust_pairs_per_sec": round(rust_tps, 1),
        "python_pairs_per_sec": round(py_tps, 1),
        "rust_speedup": round(rust_tps / py_tps, 3) if (py_tps and rust_available) else None,
        "fastest": "rust" if (rust_available and rust_tps >= py_tps) else "python",
    }


def run(*, repeats: int = 2000) -> dict:
    store = GroundTruthStore()
    for key, value in _FACTS.items():
        store.add(key, value)
    forecaster = HallucinationForecaster()
    return {
        "benchmark": "hallucination_forecast",
        "separation": _separation(forecaster, store),
        "backends": _backend_comparison(repeats),
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "hallucination_forecast.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    s = result["separation"]
    b = result["backends"]
    print(f"\nHallucination forecast (n={s['n_prompts']}):")
    print(
        f"  risk low={s['mean_risk_low']:.3f} high={s['mean_risk_high']:.3f} "
        f"gap={s['risk_gap']:.3f}"
    )
    print(
        f"  ranking AUROC={s['ranking_auroc']:.3f} "
        f"band_accuracy={s['band_accuracy']:.3f}"
    )
    print(
        f"  backend rust_available={b['rust_available']} "
        f"parity={b['parity_rust_equals_python']}"
    )
    print(
        f"  throughput rust={b['rust_pairs_per_sec']:.0f}/s "
        f"python={b['python_pairs_per_sec']:.0f}/s "
        f"speedup={b['rust_speedup']} fastest={b['fastest']}"
    )


if __name__ == "__main__":
    main()
