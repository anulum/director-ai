# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — temporal-refresh parser/lexical/Rust-vs-Python benchmark

"""Measure the offline parts of the live temporal-claim refresher.

The live web-search path cannot run deterministically in a committed benchmark,
so every measurement here is offline and reproducible:

* **HTML parsing** — throughput of the DuckDuckGo result parser on a fixed
  fragment, and a correctness check on the parsed hits;
* **Backend comparison** — the ``topical_overlap`` lexical kernel run with the
  Rust ``rust_word_overlap`` extension and with the pure-Python Jaccard fallback
  over a claim/evidence corpus; the two must agree exactly (parity) and both
  throughputs are recorded;
* **Lexical drift accuracy** — the dependency-free verdict heuristic (no NLI) run
  over a labelled set of position claims whose asserted incumbent either does or
  does not appear in the supplied top result. This measures the *triage* path on
  clean cases; the documented co-occurrence weakness (a former office-holder's
  name persisting in current coverage) is out of scope here and is why the NLI
  engine exists.

Output: ``benchmarks/results/temporal_refresh.json``. Reproduce with
``python -m benchmarks.temporal_refresh``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.scoring import temporal_refresh as tr
from director_ai.core.scoring.temporal_refresh import SearchHit, TemporalRefresher

_DDG_FRAGMENT = b"""
<div class="result results_links">
  <h2 class="result__title">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSam_Altman&amp;rut=a">Sam Altman - Wikipedia</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSam_Altman&amp;rut=a">Sam Altman is the <b>CEO</b> of OpenAI.</a>
</div>
<div class="result results_links">
  <h2 class="result__title">
    <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.forbes.com%2Fprofile%2Fsam-altman%2F&amp;rut=b">Sam Altman - Forbes</a>
  </h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.forbes.com%2Fprofile%2Fsam-altman%2F&amp;rut=b">Sam Altman is the CEO of OpenAI and an investor.</a>
</div>
"""

# (response, top-result text, drift expected) — clean position cases.
_LABELLED: list[tuple[str, str, bool]] = [
    ("The CEO of OpenAI is Sam Altman.", "Sam Altman - Wikipedia: Sam Altman is CEO of OpenAI", False),
    ("The CEO of Apple is Tim Cook.", "Tim Cook - Apple Leadership: Tim Cook is CEO of Apple", False),
    ("The CEO of Microsoft is Satya Nadella.", "Satya Nadella is the chief executive of Microsoft", False),
    ("The CEO of Tesla is Elon Musk.", "Elon Musk leads Tesla as chief executive officer", False),
    ("The CEO of NVIDIA is Jensen Huang.", "Jensen Huang is the founder and CEO of NVIDIA", False),
    ("The CEO of Twitter is Jack Dorsey.", "Linda Yaccarino - Wikipedia: Linda Yaccarino is CEO of X", True),
    ("The CEO of Google is Eric Schmidt.", "Sundar Pichai is the chief executive of Google", True),
    ("The CEO of Disney is Bob Chapek.", "Bob Iger returned as Disney chief executive", True),
    ("The CEO of Starbucks is Kevin Johnson.", "Brian Niccol is the chief executive of Starbucks", True),
    ("The CEO of Intel is Pat Gelsinger.", "Lip-Bu Tan is the chief executive of Intel", True),
]


class _OneHitProvider:
    def __init__(self, text: str) -> None:
        self._text = text

    def search(self, query: str, *, max_results: int) -> list[SearchHit]:
        return [SearchHit(title=self._text, snippet="", url="https://example", rank=0)]


def parser_throughput(repeats: int) -> dict:
    hits = tr._ddg_hits(_DDG_FRAGMENT, max_results=5)
    t0 = time.perf_counter()
    for _ in range(repeats):
        tr._ddg_hits(_DDG_FRAGMENT, max_results=5)
    elapsed = time.perf_counter() - t0
    return {
        "hits_parsed": len(hits),
        "first_url": hits[0].url if hits else "",
        "parses_per_sec": round(repeats / elapsed, 1) if elapsed else 0.0,
    }


def _overlap_corpus() -> list[tuple[str, str]]:
    return [(resp, ev) for resp, ev, _ in _LABELLED]


def backend_comparison(repeats: int) -> dict:
    corpus = _overlap_corpus()
    rust_available = tr._RUST_REFRESH and tr.rust_word_overlap is not None

    rust_vals = [tr._lexical_overlap(a, b) for a, b in corpus]

    def _tps() -> float:
        t0 = time.perf_counter()
        for _ in range(repeats):
            for a, b in corpus:
                tr._lexical_overlap(a, b)
        elapsed = time.perf_counter() - t0
        return (len(corpus) * repeats) / elapsed if elapsed else 0.0

    rust_tps = _tps() if rust_available else 0.0
    saved_flag, saved_fn = tr._RUST_REFRESH, tr.rust_word_overlap
    tr._RUST_REFRESH, tr.rust_word_overlap = False, None
    try:
        py_vals = [tr._lexical_overlap(a, b) for a, b in corpus]
        py_tps = _tps()
    finally:
        tr._RUST_REFRESH, tr.rust_word_overlap = saved_flag, saved_fn

    parity = all(abs(r - p) < 1e-9 for r, p in zip(rust_vals, py_vals, strict=True))
    return {
        "rust_available": rust_available,
        "parity_rust_equals_python": parity,
        "rust_pairs_per_sec": round(rust_tps, 1),
        "python_pairs_per_sec": round(py_tps, 1),
        "rust_speedup": round(rust_tps / py_tps, 3) if (py_tps and rust_available) else None,
        "fastest": "rust" if (rust_available and rust_tps >= py_tps) else "python",
    }


def lexical_drift_accuracy() -> dict:
    tp = fp = tn = fn = 0
    for response, evidence, expect_drift in _LABELLED:
        refresher = TemporalRefresher(provider=_OneHitProvider(evidence))
        report = refresher.refresh_response(response)
        drift = any(r.verdict == "drift_suspected" for r in report.refreshes)
        if expect_drift and drift:
            tp += 1
        elif expect_drift and not drift:
            fn += 1
        elif not expect_drift and drift:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "n": len(_LABELLED),
        "accuracy": round((tp + tn) / len(_LABELLED), 4),
        "drift_precision": round(precision, 4),
        "drift_recall": round(recall, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def run(*, parser_repeats: int = 20000, overlap_repeats: int = 2000) -> dict:
    return {
        "benchmark": "temporal_refresh",
        "parser": parser_throughput(parser_repeats),
        "backends": backend_comparison(overlap_repeats),
        "lexical_drift": lexical_drift_accuracy(),
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "temporal_refresh.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    p = result["parser"]
    b = result["backends"]
    d = result["lexical_drift"]
    print("\nTemporal refresh:")
    print(f"  parser hits={p['hits_parsed']} {p['parses_per_sec']:.0f}/s")
    print(
        f"  backend rust_available={b['rust_available']} "
        f"parity={b['parity_rust_equals_python']} fastest={b['fastest']}"
    )
    print(
        f"  lexical drift accuracy={d['accuracy']:.2f} "
        f"precision={d['drift_precision']:.2f} recall={d['drift_recall']:.2f} "
        f"(n={d['n']})"
    )


if __name__ == "__main__":
    main()
