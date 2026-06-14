# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-model consensus benchmark

"""Measure the cross-model consensus engine on its offline lexical path.

The semantic (NLI) path is validated by the unit tests with a deterministic
scorer; here we measure the parts that run everywhere without loading a model:

* **Discrimination accuracy** — labelled answer panels, each tagged ``accept``
  (the models give the same answer) or ``escalate`` (they give unrelated
  answers), are run through the lexical consensus; we report how often the
  recommendation matches the label.
* **Rust/Python parity** — the lexical Jaccard backend is computed with the Rust
  kernel and with the forced Python fallback; we report whether every pair is
  bit-for-bit identical (the dispatch is a pure speed choice).
* **Throughput** — consensus computations per second over the panels.

Output: ``benchmarks/results/cross_model_consensus.json``. Reproduce with
``python -m benchmarks.cross_model_consensus``.
"""

from __future__ import annotations

import json
import time

from benchmarks._common import RESULTS_DIR
from director_ai.core.consensus import CrossModelConsensus, ModelResponse
from director_ai.core.consensus import cross_model_consensus as _cmc

# (answers, expected_recommendation) — lexical agreement only.
_PANELS: list[tuple[list[str], str]] = [
    (
        [
            "The capital of France is Paris.",
            "The capital of France is Paris.",
            "The capital of France is Paris.",
        ],
        "accept",
    ),
    (
        [
            "Water boils at 100 degrees Celsius at sea level.",
            "Water boils at 100 degrees Celsius at sea level.",
        ],
        "accept",
    ),
    (
        [
            "The mitochondria is the powerhouse of the cell.",
            "Photosynthesis occurs in chloroplasts of plants.",
        ],
        "escalate",
    ),
    (
        [
            "Quantum entanglement links two distant particles.",
            "The French Revolution began in the year 1789.",
            "Maple syrup is produced from tree sap in spring.",
        ],
        "escalate",
    ),
    (
        [
            "Photosynthesis converts light into chemical energy.",
            "Photosynthesis converts light into chemical energy.",
        ],
        "accept",
    ),
]


def _panels() -> list[list[ModelResponse]]:
    return [
        [ModelResponse(model_id=f"m{i}", text=t) for i, t in enumerate(answers)]
        for answers, _ in _PANELS
    ]


def discrimination_accuracy() -> dict:
    engine = CrossModelConsensus()  # lexical mode
    hits = sum(
        1
        for (answers, expected), panel in zip(_PANELS, _panels(), strict=True)
        if engine.consensus(panel).recommendation == expected
    )
    return {"n": len(_PANELS), "accuracy": round(hits / len(_PANELS), 4)}


def parity() -> dict:
    """Bit-exact agreement between the Rust kernel and the Python fallback."""
    saved_flag, saved_fn = _cmc._RUST_CONSENSUS, _cmc.rust_word_overlap
    pairs = [
        (a, b) for answers, _ in _PANELS for a in answers for b in answers if a != b
    ]
    try:
        rust_vals = [_cmc._lexical_overlap(a, b) for a, b in pairs]
        _cmc._RUST_CONSENSUS = False
        _cmc.rust_word_overlap = None
        py_vals = [_cmc._lexical_overlap(a, b) for a, b in pairs]
    finally:
        _cmc._RUST_CONSENSUS, _cmc.rust_word_overlap = saved_flag, saved_fn
    exact = all(r == p for r, p in zip(rust_vals, py_vals, strict=True))
    return {"pairs": len(pairs), "bit_exact": exact, "kernel_active": saved_flag}


def throughput(repeats: int) -> dict:
    engine = CrossModelConsensus()
    panels = _panels()
    t0 = time.perf_counter()
    for _ in range(repeats):
        for panel in panels:
            engine.consensus(panel)
    elapsed = time.perf_counter() - t0
    rate = (len(panels) * repeats) / elapsed if elapsed else 0.0
    return {"consensus_per_sec": round(rate, 1)}


def run(*, repeats: int = 2000) -> dict:
    return {
        "benchmark": "cross_model_consensus",
        "discrimination": discrimination_accuracy(),
        "parity": parity(),
        "throughput": throughput(repeats),
        "backend": "rust_word_overlap kernel with bit-exact Python Jaccard fallback",
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "cross_model_consensus.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    d = result["discrimination"]
    print(f"\nCross-model consensus (n={d['n']}):")
    print(f"  lexical discrimination accuracy={d['accuracy']:.2f}")
    print(f"  parity bit_exact={result['parity']['bit_exact']}")
    print(f"  throughput {result['throughput']['consensus_per_sec']:.0f}/s")


if __name__ == "__main__":
    main()
