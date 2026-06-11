# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - Patronus HaluBench text benchmark

"""Evaluate Director-AI on the PatronusAI/HaluBench text dataset.

Dataset source: ``PatronusAI/HaluBench`` on Hugging Face. As of the
2026-05-18 integration check, the public test split exposes
``id``, ``passage``, ``question``, ``answer``, ``label``, and
``source_ds`` fields. ``label=FAIL`` denotes a hallucinated answer;
``label=PASS`` denotes a grounded answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger("DirectorAI.Benchmark.PatronusHaluBench")

DATASET_ID = "PatronusAI/HaluBench"
DEFAULT_SPLIT = "test"


def _load_rows(split: str, max_samples: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split, trust_remote_code=False)
    rows = list(ds)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def _normalise_label(label: object) -> bool | None:
    if label is None:
        return None
    value = str(label).strip().upper()
    if value == "FAIL":
        return True
    if value == "PASS":
        return False
    return None


def _rate(num: int, denom: int) -> float:
    return num / denom if denom else 0.0


def run_patronus_halubench(
    *,
    split: str = DEFAULT_SPLIT,
    max_samples: int | None = None,
    threshold: float = 0.5,
    soft_limit: float = 0.6,
    use_nli: bool = True,
    nli_model: str | None = None,
    nli_torch_dtype: str | None = None,
) -> dict:
    """Run HaluBench and return aggregate metrics only."""

    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import VectorGroundTruthStore

    rows = _load_rows(split, max_samples)
    scorer = CoherenceScorer(
        threshold=threshold,
        soft_limit=soft_limit,
        use_nli=use_nli,
        nli_model=nli_model,
        nli_torch_dtype=nli_torch_dtype,
        ground_truth_store=VectorGroundTruthStore(),
    )

    counts = {
        "total": 0,
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "skipped": 0,
    }
    latency_ms: list[float] = []
    by_source: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )

    for row in rows:
        passage = str(row.get("passage") or "")
        question = str(row.get("question") or "")
        answer = str(row.get("answer") or "")
        is_hallucinated = _normalise_label(row.get("label"))
        source = str(row.get("source_ds") or "unknown")
        if not passage or not answer or is_hallucinated is None:
            counts["skipped"] += 1
            continue

        prompt = f"{question}\n\n{passage}" if question else passage
        scorer._ground_truth_store = VectorGroundTruthStore()
        scorer._ground_truth_store.ingest([passage])

        start = time.perf_counter()
        approved, _score = scorer.review(prompt, answer)
        latency_ms.append((time.perf_counter() - start) * 1000.0)

        flagged = not approved
        counts["total"] += 1
        by_source[source]["total"] += 1
        if is_hallucinated and flagged:
            bucket = "tp"
        elif not is_hallucinated and flagged:
            bucket = "fp"
        elif not is_hallucinated and not flagged:
            bucket = "tn"
        else:
            bucket = "fn"
        counts[bucket] += 1
        by_source[source][bucket] += 1

    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    latencies = np.asarray(latency_ms, dtype=float)
    return {
        "benchmark": "PatronusAI/HaluBench",
        "dataset": {
            "source": DATASET_ID,
            "split": split,
            "label_policy": "FAIL=hallucinated, PASS=grounded",
        },
        "total": counts["total"],
        "skipped": counts["skipped"],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "catch_rate": round(_rate(tp, tp + fn), 4),
        "false_positive_rate": round(_rate(fp, fp + tn), 4),
        "precision": round(_rate(tp, tp + fp), 4),
        "f1": round(
            _rate(2 * tp, (2 * tp) + fp + fn),
            4,
        ),
        "accuracy": round(_rate(tp + tn, counts["total"]), 4),
        "avg_latency_ms": round(float(latencies.mean()), 2) if latencies.size else 0.0,
        "p95_latency_ms": (
            round(float(np.percentile(latencies, 95)), 2) if latencies.size else 0.0
        ),
        "threshold": threshold,
        "soft_limit": soft_limit,
        "use_nli": use_nli,
        "nli_model": nli_model,
        "per_source": dict(sorted(by_source.items())),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--soft-limit", type=float, default=0.6)
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--nli-torch-dtype", default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("benchmarks/results/patronus_halubench_eval.json"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_patronus_halubench(
        split=args.split,
        max_samples=args.max_samples,
        threshold=args.threshold,
        soft_limit=args.soft_limit,
        use_nli=not args.no_nli,
        nli_model=args.model,
        nli_torch_dtype=args.nli_torch_dtype,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.output_json}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
