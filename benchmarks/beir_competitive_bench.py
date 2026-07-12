# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — BEIR public-benchmark retrieval evaluation (WCA-8)

"""Public-benchmark evidence for the ``grounded()`` retrieval pipeline.

Runs the shipped ``VectorGroundTruthStore.grounded()`` recipe — hybrid
BM25 + dense retrieval with RRF fusion, optionally cross-encoder
reranked — on BEIR test splits (NFCorpus, SciFact) and reports nDCG@10
next to published baseline numbers so the results are comparable across
systems, unlike the internal ``retrieval_bench`` evaluation set.

Scoring uses ``pytrec_eval`` (the scorer the BEIR paper uses) when
installed, cross-checked against a built-in linear-gain nDCG@10
implementation; the JSON artefact records both plus the maximum
disagreement. Published baselines embedded below were verified at source
on 2026-07-12 (BEIR paper Table 2; bge-large-en-v1.5 model-card MTEB
metrics).

Caveats recorded in the artefact: the recipe indexes ``title + text`` as
one field, fuses BM25 with dense via RRF (k=60), reranks the top 30
candidates only, and does not add the bge-large query instruction prefix
— it measures the pipeline users get, not a leaderboard submission.

Datasets are the standard BEIR zips extracted under
``benchmarks/data/beir/<dataset>/`` (corpus.jsonl, queries.jsonl,
qrels/test.tsv). They are not committed to the repository.

Usage::

    python -m benchmarks.beir_competitive_bench
    python -m benchmarks.beir_competitive_bench --datasets scifact \
        --arms bge_m3__none bge_m3__bge_v2_m3
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

from benchmarks.grounded_ann_bench import _latency_summary
from benchmarks.retrieval_model_refresh_ab import (
    BGE_LARGE,
    BGE_M3,
    BGE_RERANKER,
    MSMARCO,
    _cached_revision,
    _environment,
)

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data" / "beir"
TOP_K = 10

EMBEDDERS = {"bge_large": BGE_LARGE, "bge_m3": BGE_M3}
RERANKERS = {"msmarco": MSMARCO, "bge_v2_m3": BGE_RERANKER}

# arm name -> (embedder key, reranker key or None); arms sharing an
# embedder reuse one indexed store, so the corpus is embedded once.
ARMS: dict[str, tuple[str, str | None]] = {
    "bge_large__none": ("bge_large", None),
    "bge_large__msmarco": ("bge_large", "msmarco"),
    "bge_m3__none": ("bge_m3", None),
    "bge_m3__msmarco": ("bge_m3", "msmarco"),
    "bge_m3__bge_v2_m3": ("bge_m3", "bge_v2_m3"),
}

DATASETS = ("nfcorpus", "scifact")

# Verified at source 2026-07-12. BEIR paper rows are nDCG@10 from
# Thakur et al. 2021 (arXiv:2104.08663), Table 2 ("In-domain and
# zero-shot performances on beir benchmark. All scores denote nDCG@10.");
# BM25+CE there reranks BM25 top-100 with the ms-marco MiniLM
# cross-encoder. bge-large-en-v1.5 rows are the model card's MTEB
# model-index ndcg_at_10 entries (0-100 scale, divided by 100 here);
# MTEB evaluates pure dense retrieval with the model's query
# instruction prefix, which the grounded() recipe does not add.
PUBLISHED_BASELINES: dict[str, dict[str, dict[str, Any]]] = {
    "nfcorpus": {
        "bm25": {
            "ndcg_at_10": 0.325,
            "source": "arXiv:2104.08663 Table 2",
        },
        "bm25_ce_msmarco_minilm": {
            "ndcg_at_10": 0.350,
            "source": "arXiv:2104.08663 Table 2",
        },
        "bge_large_en_v15_dense": {
            "ndcg_at_10": 0.38129,
            "source": "https://huggingface.co/BAAI/bge-large-en-v1.5 "
            "model-index (MTEB NFCorpus)",
        },
    },
    "scifact": {
        "bm25": {
            "ndcg_at_10": 0.665,
            "source": "arXiv:2104.08663 Table 2",
        },
        "bm25_ce_msmarco_minilm": {
            "ndcg_at_10": 0.688,
            "source": "arXiv:2104.08663 Table 2",
        },
        "bge_large_en_v15_dense": {
            "ndcg_at_10": 0.74607,
            "source": "https://huggingface.co/BAAI/bge-large-en-v1.5 "
            "model-index (MTEB SciFact)",
        },
    },
}


def _load_dataset(
    root: Path,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, int]]]:
    """Load a BEIR dataset directory into docs, test queries, and qrels."""
    docs: dict[str, str] = {}
    with (root / "corpus.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = f"{row.get('title', '')} {row.get('text', '')}".strip()
            docs[str(row["_id"])] = text

    qrels: dict[str, dict[str, int]] = {}
    with (root / "qrels" / "test.tsv").open(encoding="utf-8") as fh:
        header = fh.readline()
        if not header.startswith("query-id"):
            raise ValueError(f"unexpected qrels header: {header!r}")
        for line in fh:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)

    queries: dict[str, str] = {}
    with (root / "queries.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            qid = str(row["_id"])
            if qid in qrels:
                queries[qid] = row["text"]
    return docs, queries, qrels


def _ndcg_at_k(ranked_ids: list[str], rels: dict[str, int], k: int) -> float:
    """Linear-gain nDCG@k (trec_eval convention, as used by BEIR)."""
    dcg = sum(
        rels.get(doc_id, 0) / math.log2(rank + 2)
        for rank, doc_id in enumerate(ranked_ids[:k])
    )
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _score_run(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Score a retrieval run with pytrec_eval plus the built-in nDCG@10."""
    own_ndcg = {}
    for qid, doc_scores in run.items():
        ranked = sorted(doc_scores, key=doc_scores.__getitem__, reverse=True)
        own_ndcg[qid] = _ndcg_at_k(ranked, qrels[qid], TOP_K)
    result: dict[str, Any] = {
        "ndcg_at_10_own": round(sum(own_ndcg.values()) / len(own_ndcg), 5),
    }
    try:
        import pytrec_eval
    except ImportError:
        result["scorer"] = "own"
        result["ndcg_at_10"] = result["ndcg_at_10_own"]
        return result

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {"ndcg_cut.10", "recall.10"},
    )
    scores = evaluator.evaluate(run)
    n = len(scores)
    ndcg = sum(q["ndcg_cut_10"] for q in scores.values()) / n
    recall = sum(q["recall_10"] for q in scores.values()) / n
    result.update(
        {
            "scorer": "pytrec_eval",
            "ndcg_at_10": round(ndcg, 5),
            "recall_at_10": round(recall, 5),
            "own_vs_pytrec_max_abs_diff": round(
                max(abs(own_ndcg[qid] - scores[qid]["ndcg_cut_10"]) for qid in scores),
                8,
            ),
        },
    )
    return result


def _run_queries(
    backend: Any,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Query every test topic, returning scored metrics and latencies."""
    run: dict[str, dict[str, float]] = {}
    latencies: list[float] = []
    for qid, text in queries.items():
        t0 = time.perf_counter()
        rows = backend.query(text, n_results=TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        # Descending pseudo-scores preserve the backend's ranking order.
        run[qid] = {
            str(row["id"]): float(len(rows) - rank) for rank, row in enumerate(rows)
        }
    return {
        "n_queries": len(queries),
        **_score_run(run, {qid: qrels[qid] for qid in run}),
        **_latency_summary(latencies),
    }


def _build_indexed_store(embedding_model: str, docs: dict[str, str]) -> Any:
    """Index one BEIR corpus through the grounded() recipe, unreranked."""
    from director_ai.core.vector_store import VectorGroundTruthStore

    store = VectorGroundTruthStore.grounded(
        embedding_model=embedding_model,
        use_reranker=False,
    )
    t0 = time.perf_counter()
    for doc_id, text in docs.items():
        store.backend.add(doc_id, text)
    ingest_seconds = time.perf_counter() - t0
    return store, ingest_seconds


def _write_artefact(output: dict[str, Any], path: Path) -> None:
    """Persist the artefact after every arm so long runs are resumable."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the requested dataset/arm grid and write the JSON artefact."""
    parser = argparse.ArgumentParser(
        description="BEIR public-benchmark evaluation of grounded()",
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
        choices=sorted(ARMS),
        default=sorted(ARMS),
        help="pipeline arms to evaluate (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run arms already present in the artefact",
    )
    args = parser.parse_args()

    path = RESULTS_DIR / "beir_competitive_bench.json"
    if path.is_file():
        output = json.loads(path.read_text(encoding="utf-8"))
    else:
        output = {
            "benchmark": "beir_competitive_bench",
            "top_k": TOP_K,
            "note": (
                "BEIR test splits through the public grounded() recipe "
                "(hybrid BM25+dense RRF k=60, title+text as one field, "
                "rerank arms rescore the top 30 candidates). Published "
                "baselines were measured by their authors under their own "
                "pipelines — see each source; the bge-large MTEB rows use "
                "a query instruction prefix grounded() does not add."
            ),
            "published_baselines": PUBLISHED_BASELINES,
            "datasets": {},
        }
    output["environment"] = _environment()
    output["model_revisions"] = {
        model: _cached_revision(model)
        for model in (*EMBEDDERS.values(), *RERANKERS.values())
    }

    from director_ai.core.vector_store import RerankedBackend

    for dataset in args.datasets:
        docs, queries, qrels = _load_dataset(DATA_DIR / dataset)
        ds_out = output["datasets"].setdefault(
            dataset,
            {"n_docs": len(docs), "n_test_queries": len(queries), "arms": {}},
        )
        wanted = [
            name for name in args.arms if args.force or name not in ds_out["arms"]
        ]
        print(f"\n=== {dataset}: {len(docs)} docs, {len(queries)} queries ===")
        for embed_key, embedding_model in EMBEDDERS.items():
            arm_names = [n for n in wanted if ARMS[n][0] == embed_key]
            if not arm_names:
                continue
            store, ingest_seconds = _build_indexed_store(embedding_model, docs)
            for name in arm_names:
                rerank_key = ARMS[name][1]
                if rerank_key is None:
                    backend: Any = store.backend
                else:
                    backend = RerankedBackend(
                        base=store.backend,
                        reranker_model=RERANKERS[rerank_key],
                    )
                t0 = time.perf_counter()
                arm = {
                    "embedding_model": embedding_model,
                    "reranker_model": (RERANKERS[rerank_key] if rerank_key else None),
                    "ingest_seconds": round(ingest_seconds, 1),
                    **_run_queries(backend, queries, qrels),
                }
                arm["arm_seconds"] = round(time.perf_counter() - t0, 1)
                ds_out["arms"][name] = arm
                _write_artefact(output, path)
                print(
                    f"  {name:<20} nDCG@10={arm['ndcg_at_10']:.4f}  "
                    f"p50={arm['p50_ms']:.0f} ms  "
                    f"({arm['arm_seconds']:.0f}s)",
                )
                if rerank_key is not None:
                    del backend
                    gc.collect()
            del store
            gc.collect()

    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
