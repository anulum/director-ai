# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — self-consistency eval on wiki_bio_gpt3_hallucination

"""WCA-3 acceptance eval: the semantic-entropy signal on a public set.

Dataset: ``potsawee/wiki_bio_gpt3_hallucination`` (SelfCheckGPT,
Manakul et al. 2023) — 238 GPT-3 WikiBio passages, each with 20
independent samples for the same prompt and sentence-level
``accurate / minor_inaccurate / major_inaccurate`` annotations.

Protocol (passage level — our signal scores whole responses):

- ground truth per passage = fraction of sentences annotated
  inaccurate (``major`` = 1.0, ``minor`` = 0.5, ``accurate`` = 0.0,
  averaged) — the SelfCheckGPT passage-level convention;
- signal per passage = ``1 − consistency_score`` of the passage
  against its 20 samples (higher = more suspect);
- metrics: Pearson + Spearman correlation between signal and ground
  truth, and AUROC for detecting strongly hallucinated passages
  (ground truth ≥ 0.5), computed rank-based without sklearn.

Both entailment backends are measured: ``lexical`` (dependency-free
floor) and ``nli`` (shipped FactCG scorer) — the honest comparison of
what a bare install gets versus the production path.

Usage::

    python -m benchmarks.self_consistency_wikibio_bench --backends lexical
    python -m benchmarks.self_consistency_wikibio_bench  # both backends
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from benchmarks.grounded_ann_bench import RESULTS_DIR
from benchmarks.retrieval_model_refresh_ab import _environment

DATASET = "potsawee/wiki_bio_gpt3_hallucination"

_LABEL_SCORES = {
    "accurate": 0.0,
    "minor_inaccurate": 0.5,
    "major_inaccurate": 1.0,
}


def _passage_truth(annotations: list[str]) -> float:
    """SelfCheckGPT passage-level ground truth: mean sentence score."""
    if not annotations:
        return 0.0
    return sum(_LABEL_SCORES[a] for a in annotations) / len(annotations)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based) with ties shared, for Spearman/AUROC."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        tie_end = index
        while (
            tie_end + 1 < len(order)
            and values[order[tie_end + 1]] == values[order[index]]
        ):
            tie_end += 1
        average_rank = (index + tie_end) / 2 + 1
        for position in range(index, tie_end + 1):
            ranks[order[position]] = average_rank
        index = tie_end + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / (var_x**0.5 * var_y**0.5)


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _auroc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based AUROC (Mann-Whitney U with tie correction)."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = _ranks(scores)
    rank_sum = sum(r for r, flag in zip(ranks, labels, strict=True) if flag)
    u_statistic = rank_sum - n_pos * (n_pos + 1) / 2
    return u_statistic / (n_pos * n_neg)


def _load_passages() -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="evaluation")
    passages: list[dict[str, Any]] = []
    for row in rows:
        passages.append(
            {
                "passage": row["gpt3_text"],
                "samples": list(row["gpt3_text_samples"]),
                "truth": _passage_truth(list(row["annotation"])),
            },
        )
    return passages


def _run_backend(
    backend: str,
    passages: list[dict[str, Any]],
) -> dict[str, Any]:
    from director_ai.core.scoring.self_consistency import SelfConsistencyScorer

    nli = None
    if backend == "nli":
        from director_ai.core.scoring.nli import NLIScorer

        nli = NLIScorer(use_model=True)
    scorer = SelfConsistencyScorer(nli_scorer=nli)

    signals: list[float] = []
    truths: list[float] = []
    latencies: list[float] = []
    for row in passages:
        t0 = time.perf_counter()
        result = scorer.score(row["passage"], row["samples"])
        latencies.append(time.perf_counter() - t0)
        signals.append(1.0 - result.consistency_score)
        truths.append(row["truth"])

    strong_labels = [truth >= 0.5 for truth in truths]
    return {
        "entailment_backend": backend,
        "n_passages": len(passages),
        "pearson_vs_truth": round(_pearson(signals, truths), 4),
        "spearman_vs_truth": round(_spearman(signals, truths), 4),
        "auroc_strong_hallucination": round(_auroc(signals, strong_labels), 4),
        "n_strong_hallucinated": sum(strong_labels),
        "mean_seconds_per_passage": round(sum(latencies) / len(latencies), 3),
    }


def main() -> None:
    """Run the requested backends and write the JSON artefact."""
    parser = argparse.ArgumentParser(
        description="Self-consistency signal eval on wiki_bio_gpt3_hallucination",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["lexical", "nli"],
        default=["lexical", "nli"],
        help="entailment backends to evaluate (default: both)",
    )
    args = parser.parse_args()

    path = RESULTS_DIR / "self_consistency_wikibio_bench.json"
    if path.is_file():
        output = json.loads(path.read_text(encoding="utf-8"))
    else:
        output = {
            "benchmark": "self_consistency_wikibio_bench",
            "dataset": DATASET,
            "note": (
                "Passage-level SelfCheckGPT protocol: signal = 1 - "
                "consistency_score of each GPT-3 passage against its 20 "
                "dataset-shipped samples; truth = mean sentence "
                "annotation score (accurate 0, minor 0.5, major 1). "
                "AUROC target = passages with truth >= 0.5. Rank-based "
                "metrics, no sklearn."
            ),
            "backends": {},
        }
    output["environment"] = _environment()

    passages = _load_passages()
    for backend in args.backends:
        print(f"=== backend {backend}: {len(passages)} passages ===")
        result = _run_backend(backend, passages)
        output["backends"][backend] = result
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(
            f"  pearson={result['pearson_vs_truth']:.4f}  "
            f"spearman={result['spearman_vs_truth']:.4f}  "
            f"auroc={result['auroc_strong_hallucination']:.4f}  "
            f"({result['mean_seconds_per_passage']:.3f}s/passage)",
        )

    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
