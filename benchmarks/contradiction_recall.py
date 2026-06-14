# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — clean contradiction recall via counterfactual injection

"""Measure the streaming-halt contradiction signal on synthesised contradictions.

AggreFact's ``unsupported`` class mixes contradictions with merely-unsupported
claims, so recall against it understates the true contradiction recall. This
benchmark builds a clean test instead: every supported (label 1) claim is paired
with a meaning-flipped variant (see :mod:`benchmarks.contradiction_injection`)
that contradicts the same document. Then:

* the **original** supported claim is the false-halt probe — ``P(contradiction)``
  must stay below the halt threshold;
* the **injected** claim is the recall probe — it genuinely contradicts the
  document, so ``P(contradiction)`` must rise above the threshold.

Both are scored against the same grounding at the same granularity as
``contradiction_aggrefact`` (whole document, or top-k relevant passages), so the
two benchmarks are directly comparable. Recall is broken down by injection
strategy so it is clear which contradiction types the model catches.

Output: ``benchmarks/results/contradiction_recall_<granularity>.json``.
Reproduce with ``python -m benchmarks.contradiction_recall`` (``--device`` for
GPU, ``--granularity passage --top-k K`` for passage grounding).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from benchmarks._common import save_results
from benchmarks.contradiction_aggrefact import (
    _auc,
    _device_label,
    _metrics,
    _score_rows,
)
from benchmarks.contradiction_injection import ContradictionInjector
from director_ai.core.scoring.contradiction import ContradictionScorer

_DATA = Path(__file__).resolve().parent / "aggrefact_test.jsonl"


def _load_supported(max_samples: int | None) -> list[dict]:
    rows = [json.loads(line) for line in _DATA.read_text(encoding="utf-8").splitlines()]
    supported = [r for r in rows if int(r["label"]) == 1]
    return supported[:max_samples] if max_samples is not None else supported


def run(
    max_samples: int | None = None,
    *,
    batch_size: int = 64,
    device: int = -1,
    granularity: str = "document",
    top_k: int = 5,
    model_id: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
) -> dict:
    supported = _load_supported(max_samples)
    injector = ContradictionInjector()

    # Each injectable claim contributes one original (false-halt probe) and one
    # injected variant per applicable strategy (recall probes).
    original_rows: list[dict] = []
    injected_rows: list[dict] = []
    injected_strategy: list[str] = []
    for r in supported:
        variants = injector.inject_all(r["claim"])
        if not variants:
            continue
        original_rows.append({"doc": r["doc"], "claim": r["claim"]})
        for v in variants:
            injected_rows.append({"doc": r["doc"], "claim": v.perturbed})
            injected_strategy.append(v.strategy)

    scorer = ContradictionScorer.from_pretrained(model_id, device=device)
    orig_scores = _score_rows(
        scorer,
        original_rows,
        granularity=granularity,
        top_k=top_k,
        batch_size=batch_size,
    )
    print("  scored originals; scoring injected", flush=True)
    inj_scores = _score_rows(
        scorer,
        injected_rows,
        granularity=granularity,
        top_k=top_k,
        batch_size=batch_size,
    )

    by_strategy: dict[str, list[float]] = defaultdict(list)
    for strat, s in zip(injected_strategy, inj_scores, strict=True):
        by_strategy[strat].append(s)
    recall_by_strategy = {
        strat: {
            "n": len(scores),
            "recall@0.2": round(sum(1 for x in scores if x >= 0.2) / len(scores), 4),
            "mean_contradiction": round(sum(scores) / len(scores), 4),
        }
        for strat, scores in sorted(by_strategy.items())
    }

    labels = [1] * len(inj_scores) + [0] * len(orig_scores)
    return {
        "benchmark": "contradiction_recall",
        "model": model_id,
        "device": _device_label(device),
        "granularity": granularity,
        "top_k": top_k if granularity == "passage" else None,
        "n_supported_seen": len(supported),
        "n_injectable": len(original_rows),
        "n_injected_variants": len(injected_rows),
        "injection_strategy_counts": dict(Counter(injected_strategy)),
        "signal": "injected=clean contradiction (recall), original=supported (false-halt)",
        "auc_injected_vs_original": round(_auc(labels, inj_scores + orig_scores), 4),
        "overall": _metrics(inj_scores, orig_scores),
        "recall_by_strategy": recall_by_strategy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("max_samples", nargs="?", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument(
        "--granularity", choices=("document", "passage"), default="document"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model",
        default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        help="NLI model id or local path (e.g. a fine-tuned merged model).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix for the result filename, to keep before/after runs apart.",
    )
    args = parser.parse_args()
    result = run(
        args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        granularity=args.granularity,
        top_k=args.top_k,
        model_id=args.model,
    )
    suffix = f"_{args.tag}" if args.tag else ""
    save_results(result, f"contradiction_recall_{args.granularity}{suffix}.json")
    o = result["overall"]
    print(
        f"\nClean contradiction recall "
        f"(n_injected={result['n_injected_variants']}, {result['granularity']}, "
        f"{result['device']}):"
    )
    print(f"  AUC injected-vs-original: {result['auc_injected_vs_original']}")
    for thr, m in o["thresholds"].items():
        print(
            f"  thr={thr}: false-halt(original)={m['false_halt_rate']:.1%} "
            f"| recall(injected)={m['recall']:.1%}"
        )
    print("  recall@0.2 by strategy:")
    for strat, m in result["recall_by_strategy"].items():
        print(f"    {strat:11s} n={m['n']:5d}  recall={m['recall@0.2']:.1%}")


if __name__ == "__main__":
    main()
