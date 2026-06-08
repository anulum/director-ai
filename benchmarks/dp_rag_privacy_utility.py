# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DP-RAG privacy/utility benchmark
"""Measure the privacy-utility tradeoff of the DP-RAG pipeline.

Differential privacy trades utility for a privacy budget ``ε``: smaller ``ε``
means stronger privacy but noisier output. This harness quantifies that curve
for the two noised pipeline stages, so a deployment can pick ``ε`` from measured
evidence rather than a guess:

* **Retrieval** — top-``k`` overlap between the DP ranking and the noise-free
  ranking (how often the right documents still surface).
* **Decoding** — top-1 agreement between the exponential-mechanism token and the
  true argmax (how often the intended token is still chosen).

Each metric is averaged over many seeded trials per ``ε``. Utility rises
monotonically with ``ε`` (less noise), bounded by the no-privacy baseline of
``1.0``. This is a utility measurement, not a compute-latency or Rust-parity
benchmark; it is deterministic given the seed and carries no host-load claim.

Usage::

    python -m benchmarks.dp_rag_privacy_utility
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

from director_ai.core.dp_rag import DPRagPipeline, ScoredItem

RESULTS_DIR = Path(__file__).parent / "results"

EPSILONS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
TRIALS = 400
CANDIDATES = 20
TOP_K = 5
VOCAB = 32


def _true_items() -> list[ScoredItem]:
    # Distinct, separated similarity scores so the noise-free ranking is unique.
    return [ScoredItem(f"doc-{i}", 1.0 - i / CANDIDATES) for i in range(CANDIDATES)]


def _true_top_k(items: list[ScoredItem], k: int) -> set[str]:
    ranked = sorted(items, key=lambda it: it.score, reverse=True)
    return {it.item_id for it in ranked[:k]}


def _logits(argmax: int) -> list[float]:
    # A clearly separated peak so the noise-free decode has a unique argmax.
    return [3.0 if i == argmax else 0.0 for i in range(VOCAB)]


def _retrieval_overlap(epsilon: float) -> float:
    items = _true_items()
    target = _true_top_k(items, TOP_K)
    overlap = 0
    for trial in range(TRIALS):
        pipe = DPRagPipeline(max_epsilon=1e9, seed=1_000 + trial)
        ranking = pipe.rank(items, tenant_id="bench", epsilon=epsilon)
        dp_top = {it.item_id for it in ranking.items[:TOP_K]}
        overlap += len(dp_top & target)
    return overlap / (TRIALS * TOP_K)


def _decode_accuracy(epsilon: float) -> float:
    true_argmax = VOCAB // 2
    logits = _logits(true_argmax)
    hits = 0
    for trial in range(TRIALS):
        pipe = DPRagPipeline(max_epsilon=1e9, seed=5_000 + trial)
        choice = pipe.decode(logits, tenant_id="bench", epsilon=epsilon)
        hits += int(choice.index == true_argmax)
    return hits / TRIALS


def run() -> dict:
    rows = []
    for epsilon in EPSILONS:
        rows.append(
            {
                "epsilon": epsilon,
                "retrieval_top_k_overlap": round(_retrieval_overlap(epsilon), 4),
                "decode_top1_accuracy": round(_decode_accuracy(epsilon), 4),
            }
        )
    return {
        "benchmark": "dp_rag_privacy_utility",
        "kind": "privacy_utility_tradeoff",
        "isolated": False,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "config": {
            "trials": TRIALS,
            "candidates": CANDIDATES,
            "top_k": TOP_K,
            "vocab": VOCAB,
        },
        "rows": rows,
    }


def main() -> None:
    result = run()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "dp_rag_privacy_utility.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"{'epsilon':>8}  {'retrieval@5':>12}  {'decode top1':>12}")
    for row in result["rows"]:
        print(
            f"{row['epsilon']:>8}  "
            f"{row['retrieval_top_k_overlap']:>12}  "
            f"{row['decode_top1_accuracy']:>12}"
        )


if __name__ == "__main__":
    main()
