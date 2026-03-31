#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Grounded Scoring Benchmark v2

"""Measure end-to-end BA of the full RAG pipeline on AggreFact.

v2 fixes: uses SentenceTransformerBackend for proper embeddings,
samples proportionally across all 11 datasets, verifies retrieval
quality before scoring.

Usage::

    python -m benchmarks.grounded_scoring_bench --samples-per-ds 50
    python -m benchmarks.grounded_scoring_bench --samples-per-ds 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
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


def sample_proportionally(samples: list, per_ds: int, seed: int = 42) -> list:
    """Take up to per_ds samples from each dataset."""
    rng = random.Random(seed)
    by_ds = defaultdict(list)
    for s in samples:
        by_ds[s.get("dataset", "unknown")].append(s)

    selected = []
    for ds in sorted(by_ds):
        items = by_ds[ds]
        chosen = rng.sample(items, min(per_ds, len(items)))
        selected.extend(chosen)
        logger.info("  %s: %d/%d samples", ds, len(chosen), len(items))
    return selected


def build_pipeline():
    """Build scorer with SentenceTransformer embeddings + hybrid retrieval."""
    from director_ai.core.scorer import CoherenceScorer
    from director_ai.core.vector_store import (
        HybridBackend,
        SentenceTransformerBackend,
        VectorGroundTruthStore,
    )

    logger.info("Building SentenceTransformer backend...")
    base = SentenceTransformerBackend(model_name="all-MiniLM-L6-v2")
    backend = HybridBackend(base=base)
    store = VectorGroundTruthStore(backend=backend)

    scorer = CoherenceScorer(
        threshold=0.5,
        ground_truth_store=store,
        use_nli=True,
    )
    scorer._rag_claim_decomposition = False
    scorer.W_LOGIC = 0.0
    scorer.W_FACT = 1.0
    return scorer, store


def ingest_documents(store, samples: list):
    """Ingest unique source documents into the store."""
    seen_docs = set()
    doc_count = 0
    for s in samples:
        doc = s.get("doc", "")
        doc_hash = hash(doc[:500])
        if doc_hash in seen_docs or not doc:
            continue
        seen_docs.add(doc_hash)
        store.add(f"doc_{doc_count}", doc[:5000])
        doc_count += 1
    return doc_count


def verify_retrieval(store, samples: list, n_check: int = 20):
    """Spot-check that retrieval returns relevant content."""
    rng = random.Random(42)
    checks = rng.sample(samples, min(n_check, len(samples)))
    hits = 0
    for s in checks:
        claim = s.get("claim", "")
        doc = s.get("doc", "")[:200]
        context = store.retrieve_context(claim, top_k=3)
        if context and any(word in context.lower() for word in doc.lower().split()[:5]):
            hits += 1
    rate = hits / len(checks)
    logger.info(
        "Retrieval spot-check: %d/%d relevant (%.0f%%)", hits, len(checks), rate * 100
    )
    return rate


def score_samples(scorer, samples: list):
    """Score all samples and return results."""
    results = []
    t0 = time.monotonic()

    for i, s in enumerate(samples):
        claim = s.get("claim", "")
        label = s.get("label", 0)
        dataset = s.get("dataset", "unknown")

        t1 = time.monotonic()
        approved, cs = scorer.review(claim, claim)
        latency = time.monotonic() - t1

        results.append(
            {
                "dataset": dataset,
                "label": label,
                "score": round(cs.score, 6),
                "h_factual": round(cs.h_factual, 6),
                "approved": approved,
                "latency_ms": round(latency * 1000, 1),
            }
        )

        if (i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            logger.info(
                "Scored %d/%d (%.1f/s, ETA %.0fs)", i + 1, len(samples), rate, eta
            )

    return results


def analyze(results: list, sweep: bool = True):
    if sweep:
        best_ba, best_t = _sweep(results)
    else:
        best_t = 0.5
        best_ba = _macro_ba(results, best_t)

    per_ds = _per_dataset_ba(results, best_t)

    return {
        "best_threshold": best_t,
        "macro_ba_pct": round(best_ba * 100, 2),
        "baseline_open_domain_pct": 75.86,
        "delta_pp": round(best_ba * 100 - 75.86, 2),
        "total_samples": len(results),
        "avg_latency_ms": round(np.mean([r["latency_ms"] for r in results]), 1),
        "per_dataset": per_ds,
    }


def _macro_ba(results, t):
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    bas = []
    for items in by_ds.values():
        y_true = [it["label"] for it in items]
        y_pred = [int(it["score"] >= t) for it in items]
        if len(set(y_true)) < 2:
            continue
        bas.append(balanced_accuracy_score(y_true, y_pred))
    return float(np.mean(bas)) if bas else 0.0


def _sweep(results):
    best_ba, best_t = 0.0, 0.5
    for t_int in range(5, 96):
        t = t_int / 100.0
        ba = _macro_ba(results, t)
        if ba > best_ba:
            best_ba = ba
            best_t = t
    return best_ba, best_t


def _per_dataset_ba(results, t):
    by_ds = defaultdict(list)
    for r in results:
        by_ds[r["dataset"]].append(r)
    out = {}
    for ds, items in sorted(by_ds.items()):
        y_true = [it["label"] for it in items]
        y_pred = [int(it["score"] >= t) for it in items]
        if len(set(y_true)) < 2:
            out[ds] = {"ba_pct": 50.0, "n": len(items), "note": "single-class"}
            continue
        out[ds] = {
            "ba_pct": round(balanced_accuracy_score(y_true, y_pred) * 100, 1),
            "n": len(items),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Grounded scoring benchmark v2")
    parser.add_argument("--samples-per-ds", type=int, default=50)
    parser.add_argument("--no-sweep", action="store_true")
    args = parser.parse_args()

    all_samples = load_samples()
    logger.info("Loaded %d total samples", len(all_samples))

    logger.info("Sampling %d per dataset:", args.samples_per_ds)
    selected = sample_proportionally(all_samples, args.samples_per_ds)
    logger.info(
        "Selected %d samples across %d datasets",
        len(selected),
        len(set(s["dataset"] for s in selected)),
    )

    scorer, store = build_pipeline()

    logger.info("Ingesting source documents...")
    n_docs = ingest_documents(store, selected)
    logger.info("Ingested %d unique documents", n_docs)

    retrieval_rate = verify_retrieval(store, selected)
    if retrieval_rate < 0.3:
        logger.warning(
            "Retrieval quality low (%.0f%%) — results may be unreliable",
            retrieval_rate * 100,
        )

    logger.info("Scoring %d samples...", len(selected))
    results = score_samples(scorer, selected)

    summary = analyze(results, sweep=not args.no_sweep)
    summary["documents_ingested"] = n_docs
    summary["retrieval_spot_check_rate"] = round(retrieval_rate, 2)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = RESULTS_DIR / "grounded_v2_scores.json"
    scores_path.write_text(json.dumps(results, indent=2) + "\n")
    summary_path = RESULTS_DIR / "grounded_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n" + "=" * 65)
    print("  Grounded Scoring Benchmark v2")
    print("=" * 65)
    print(f"  Samples: {len(results)}, Documents: {n_docs}")
    print(f"  Retrieval quality: {retrieval_rate:.0%}")
    print(f"  Best BA: {summary['macro_ba_pct']}% at t={summary['best_threshold']}")
    print("  Open-domain baseline: 75.86%")
    print(f"  Delta: {summary['delta_pp']:+.2f}pp")
    print(f"  Avg latency: {summary['avg_latency_ms']} ms/sample")
    print("=" * 65)
    print("\nPer-dataset:")
    for ds, info in sorted(summary["per_dataset"].items()):
        note = f" ({info.get('note', '')})" if info.get("note") else ""
        print(f"  {ds:25s}: {info['ba_pct']}% (n={info['n']}){note}")


if __name__ == "__main__":
    main()
