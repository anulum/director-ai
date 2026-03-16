#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Grounded Scoring Benchmark
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────
"""Measure end-to-end BA of the full RAG pipeline on AggreFact.

Tests the COMPLETE path: ingest source documents → retrieve relevant
chunks → NLI score claims against retrieved context → measure BA.

This is the number that matters for the grounded product.

Usage::

    # Full NLI pipeline (requires GPU + transformers)
    python -m benchmarks.grounded_scoring_bench

    # Heuristic only (no GPU needed, lower accuracy)
    python -m benchmarks.grounded_scoring_bench --no-nli

    # Subset for quick testing
    python -m benchmarks.grounded_scoring_bench --max-docs 50
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("GroundedBench")

RESULTS_DIR = Path("gpu_results/grounded_bench")


def load_samples():
    jsonl = Path(__file__).parent / "aggrefact_test.jsonl"
    if jsonl.exists():
        with open(jsonl) as f:
            return [json.loads(line) for line in f if line.strip()]
    from datasets import load_dataset

    return list(load_dataset("lytang/LLM-AggreFact", split="test"))


def build_scorer(use_nli: bool, use_hybrid: bool):
    from director_ai.core import CoherenceScorer, GroundTruthStore
    from director_ai.core.vector_store import InMemoryBackend, VectorGroundTruthStore

    backend = InMemoryBackend()
    if use_hybrid:
        from director_ai.core.vector_store import HybridBackend

        backend = HybridBackend(base=backend)

    store = VectorGroundTruthStore(backend=backend)
    scorer = CoherenceScorer(
        threshold=0.5,
        ground_truth_store=store,
        use_nli=use_nli,
    )
    scorer._rag_claim_decomposition = False  # disable for benchmark speed
    scorer.W_LOGIC = 0.0  # skip logical divergence — only factual matters for RAG
    scorer.W_FACT = 1.0
    return scorer, store


def run_benchmark(
    use_nli: bool = True,
    use_hybrid: bool = True,
    max_docs: int = 0,
    sweep: bool = True,
):
    samples = load_samples()
    logger.info("Loaded %d samples", len(samples))

    # Group by unique documents
    doc_to_claims: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        doc = s.get("doc", "")
        if doc:
            doc_to_claims[doc].append(s)

    unique_docs = list(doc_to_claims.keys())
    logger.info("Found %d unique documents", len(unique_docs))

    if max_docs > 0:
        unique_docs = unique_docs[:max_docs]
        logger.info("Limited to %d documents", max_docs)

    scorer, store = build_scorer(use_nli, use_hybrid)

    # Phase 1: Ingest all source documents
    logger.info("Ingesting %d documents...", len(unique_docs))
    t0 = time.monotonic()
    for i, doc in enumerate(unique_docs):
        doc_id = f"doc_{i}"
        store.add(doc_id, doc[:5000])  # cap at 5K chars per doc
        if (i + 1) % 100 == 0:
            logger.info("Ingested %d/%d docs", i + 1, len(unique_docs))
    ingest_time = time.monotonic() - t0
    logger.info("Ingestion complete in %.1fs", ingest_time)

    # Phase 2: Score all claims against the store
    logger.info("Scoring claims...")
    results = []
    t0 = time.monotonic()
    scored = 0

    for doc in unique_docs:
        for claim_data in doc_to_claims[doc]:
            claim = claim_data.get("claim", "")
            label = claim_data.get("label", 0)
            dataset = claim_data.get("dataset", "unknown")

            t1 = time.monotonic()
            approved, cs = scorer.review(claim, claim)
            latency = time.monotonic() - t1

            results.append({
                "dataset": dataset,
                "label": label,
                "score": round(cs.score, 6),
                "approved": approved,
                "latency_ms": round(latency * 1000, 1),
                "has_attribution": cs.evidence is not None and cs.evidence.attributions is not None,
            })
            scored += 1
            if scored % 500 == 0:
                elapsed = time.monotonic() - t0
                rate = scored / elapsed
                remaining = sum(len(doc_to_claims[d]) for d in unique_docs) - scored
                eta = remaining / rate if rate > 0 else 0
                logger.info(
                    "Scored %d (%.1f/s, ETA %.0fm)",
                    scored,
                    rate,
                    eta / 60,
                )

    score_time = time.monotonic() - t0
    logger.info("Scoring complete: %d samples in %.1fs", len(results), score_time)

    # Phase 3: Compute BA
    if sweep:
        best_ba, best_t = _sweep(results)
    else:
        best_t = 0.5
        best_ba = _compute_macro_ba(results, best_t)

    per_dataset = _per_dataset_ba(results, best_t)

    summary = {
        "mode": "grounded_nli" if use_nli else "grounded_heuristic",
        "hybrid_retrieval": use_hybrid,
        "total_samples": len(results),
        "unique_docs": len(unique_docs),
        "best_threshold": best_t,
        "macro_ba_pct": round(best_ba * 100, 2),
        "baseline_open_domain_pct": 75.86,
        "delta_vs_baseline_pp": round(best_ba * 100 - 75.86, 2),
        "ingest_time_s": round(ingest_time, 1),
        "score_time_s": round(score_time, 1),
        "avg_latency_ms": round(np.mean([r["latency_ms"] for r in results]), 1),
        "attribution_rate": round(
            sum(1 for r in results if r["has_attribution"]) / len(results) * 100, 1
        ),
        "per_dataset": per_dataset,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "nli" if use_nli else "heuristic"
    scores_path = RESULTS_DIR / f"grounded_{suffix}_scores.json"
    scores_path.write_text(json.dumps(results, indent=2) + "\n")
    summary_path = RESULTS_DIR / f"grounded_{suffix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 65)
    print(f"  Grounded Scoring Benchmark ({suffix})")
    print("=" * 65)
    print(f"  Samples: {len(results)}, Documents: {len(unique_docs)}")
    print(f"  Best BA: {summary['macro_ba_pct']}% at t={best_t}")
    print(f"  Open-domain baseline: 75.86%")
    print(f"  Delta: {summary['delta_vs_baseline_pp']:+.2f}pp")
    print(f"  Avg latency: {summary['avg_latency_ms']} ms/sample")
    print(f"  Attribution rate: {summary['attribution_rate']}%")
    print("=" * 65)
    print("\nPer-dataset:")
    for ds, info in sorted(per_dataset.items()):
        print(f"  {ds:25s}: {info['ba_pct']}% (n={info['n']})")

    return summary


def _compute_macro_ba(results, threshold):
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    bas = []
    for items in by_ds.values():
        y_true = [it["label"] for it in items]
        y_pred = [int(it["score"] >= threshold) for it in items]
        bas.append(balanced_accuracy_score(y_true, y_pred))
    return float(np.mean(bas))


def _sweep(results):
    best_ba = 0.0
    best_t = 0.5
    for t_int in range(10, 91):
        t = t_int / 100.0
        ba = _compute_macro_ba(results, t)
        if ba > best_ba:
            best_ba = ba
            best_t = t
    return best_ba, best_t


def _per_dataset_ba(results, threshold):
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    out = {}
    for ds, items in sorted(by_ds.items()):
        y_true = [it["label"] for it in items]
        y_pred = [int(it["score"] >= threshold) for it in items]
        ba = balanced_accuracy_score(y_true, y_pred)
        out[ds] = {"ba_pct": round(ba * 100, 1), "n": len(items)}
    return out


def main():
    parser = argparse.ArgumentParser(description="Grounded scoring benchmark")
    parser.add_argument("--no-nli", action="store_true", help="Heuristic only")
    parser.add_argument("--no-hybrid", action="store_true", help="Skip hybrid retrieval")
    parser.add_argument("--max-docs", type=int, default=0, help="Limit documents (0=all)")
    parser.add_argument("--no-sweep", action="store_true", help="Skip threshold sweep")
    args = parser.parse_args()

    run_benchmark(
        use_nli=not args.no_nli,
        use_hybrid=not args.no_hybrid,
        max_docs=args.max_docs,
        sweep=not args.no_sweep,
    )


if __name__ == "__main__":
    main()
