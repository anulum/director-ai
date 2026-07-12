# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — BEIR fusion-strategy evaluation (WCA-9)

"""BEIR evidence for the hybrid fusion strategies beyond RRF.

Indexes each BEIR corpus once through the unreranked ``grounded()``
recipe (bge-large embedder), then evaluates every fusion arm as a
``HybridBackend.with_fusion()`` view over that single shared index —
the arms differ only in the query-time fusion of the BM25 and dense
runs, so per-arm cost is queries only, never re-embedding.

The ``rrf__s50_d50`` arm is the shipped default (weighted RRF, equal
weights, k=60) and must reproduce the ``bge_large__none`` arm of
``beir_competitive_bench`` — it doubles as a regression check that
the fusion refactor left the default path bit-identical.

Scoring, datasets and caveats match ``beir_competitive_bench``:
nDCG@10 via pytrec_eval cross-checked against the built-in
implementation, title+text as one indexed field, no reranker, no
query instruction prefix. Datasets are the standard BEIR zips
extracted under ``benchmarks/data/beir/<dataset>/`` (not committed).

Usage::

    python -m benchmarks.beir_fusion_bench
    python -m benchmarks.beir_fusion_bench --datasets nfcorpus \
        --arms convex__s30_d70 zscore__s50_d50
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

from benchmarks.beir_competitive_bench import (
    DATA_DIR,
    DATASETS,
    RESULTS_DIR,
    TOP_K,
    _build_indexed_store,
    _load_dataset,
    _run_queries,
)
from benchmarks.retrieval_model_refresh_ab import (
    BGE_LARGE,
    _cached_revision,
    _environment,
)

RRF_K = 60

# arm name -> (fusion method, sparse weight, dense weight); every arm
# is a with_fusion() view over the single bge-large indexed store.
FUSION_ARMS: dict[str, tuple[str, float, float]] = {
    "rrf__s50_d50": ("rrf", 1.0, 1.0),
    "rrf__s30_d70": ("rrf", 0.3, 0.7),
    "rrf__s70_d30": ("rrf", 0.7, 0.3),
    "convex__s50_d50": ("convex", 0.5, 0.5),
    "convex__s30_d70": ("convex", 0.3, 0.7),
    "convex__s70_d30": ("convex", 0.7, 0.3),
    "combmnz__s50_d50": ("combmnz", 0.5, 0.5),
    "zscore__s50_d50": ("zscore", 0.5, 0.5),
}

# The default-arm reference this run must reproduce: nDCG@10 of the
# bge_large__none arm in results/beir_competitive_bench.json (canonical
# GPU artefact, cross-checked identical on CPU for NFCorpus).
BASELINE_ARTEFACT = "beir_competitive_bench.json"


def _write_artefact(output: dict[str, Any], path: Path) -> None:
    """Persist after every arm so long runs are resumable.

    Model revisions are refreshed at write time — on a fresh host the
    embedder downloads during the first arm, so a start-up snapshot
    would record ``None``.
    """
    output["model_revisions"] = {BGE_LARGE: _cached_revision(BGE_LARGE)}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def _baseline_reference() -> dict[str, Any]:
    """Pull the bge_large__none nDCG@10 per dataset from the canonical
    competitive artefact, when present, for in-artefact comparison."""
    path = RESULTS_DIR / BASELINE_ARTEFACT
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    reference: dict[str, Any] = {}
    for dataset, ds_data in data.get("datasets", {}).items():
        arm = ds_data.get("arms", {}).get("bge_large__none")
        if arm:
            reference[dataset] = {
                "arm": "bge_large__none",
                "ndcg_at_10": arm["ndcg_at_10"],
                "artefact": BASELINE_ARTEFACT,
            }
    return reference


def main() -> None:
    """Run the requested dataset/fusion-arm grid and write the artefact."""
    parser = argparse.ArgumentParser(
        description="BEIR evaluation of hybrid fusion strategies",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=list(DATASETS),
        help="BEIR datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(FUSION_ARMS),
        default=sorted(FUSION_ARMS),
        help="fusion arms to evaluate (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run arms already present in the artefact",
    )
    args = parser.parse_args()

    path = RESULTS_DIR / "beir_fusion_bench.json"
    if path.is_file():
        output = json.loads(path.read_text(encoding="utf-8"))
    else:
        output = {
            "benchmark": "beir_fusion_bench",
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "embedding_model": BGE_LARGE,
            "note": (
                "BEIR test splits through the unreranked grounded() "
                "recipe (bge-large), one shared index per dataset, "
                "fusion arms as HybridBackend.with_fusion() views — "
                "arms differ only in query-time fusion of the BM25 and "
                "dense runs. rrf__s50_d50 is the shipped default and "
                "must match the bge_large__none arm of "
                "beir_competitive_bench. Same caveats as that "
                "benchmark: title+text one field, no reranker, no "
                "query instruction prefix."
            ),
            "baseline_reference": _baseline_reference(),
            "datasets": {},
        }
    output["environment"] = _environment()

    for dataset in args.datasets:
        docs, queries, qrels = _load_dataset(DATA_DIR / dataset)
        ds_out = output["datasets"].setdefault(
            dataset,
            {"n_docs": len(docs), "n_test_queries": len(queries), "arms": {}},
        )
        wanted = [
            name for name in args.arms if args.force or name not in ds_out["arms"]
        ]
        if not wanted:
            continue
        print(f"\n=== {dataset}: {len(docs)} docs, {len(queries)} queries ===")
        store, ingest_seconds = _build_indexed_store(BGE_LARGE, docs)
        for name in wanted:
            method, sparse_w, dense_w = FUSION_ARMS[name]
            backend = store.backend.with_fusion(
                method,
                rrf_k=RRF_K,
                sparse_weight=sparse_w,
                dense_weight=dense_w,
            )
            t0 = time.perf_counter()
            arm = {
                "fusion_method": method,
                "sparse_weight": sparse_w,
                "dense_weight": dense_w,
                "rrf_k": RRF_K,
                "ingest_seconds": round(ingest_seconds, 1),
                **_run_queries(backend, queries, qrels),
            }
            arm["arm_seconds"] = round(time.perf_counter() - t0, 1)
            ds_out["arms"][name] = arm
            _write_artefact(output, path)
            print(
                f"  {name:<18} nDCG@10={arm['ndcg_at_10']:.4f}  "
                f"p50={arm['p50_ms']:.0f} ms  "
                f"({arm['arm_seconds']:.0f}s)",
            )
        del store
        gc.collect()

    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
