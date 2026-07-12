# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — grounded() ANN + rerank latency/recall benchmark

"""Latency/recall evidence for the ``grounded()`` ANN + rerank default (WCA-7).

Two measurement modes, both written into one JSON artefact:

* **engine** — deterministic seeded embeddings (no model downloads) isolate
  the search engine itself: the shipped ``SentenceTransformerBackend``
  linear scan (the pre-WCA-7 dense path) against ``FAISSBackend`` flat
  (the WCA-7 default) and IVF, at several corpus sizes. Recall@k is
  computed against exact brute-force cosine ground truth.
* **real** — locally cached sentence-transformer + cross-encoder models on
  the ``retrieval_bench`` evaluation set compare the full recipes
  end-to-end: pre-WCA-7 default (linear scan, no rerank), ANN only, and
  the new default (ANN + rerank), reporting hit@1/hit@3/precision@3 and
  per-query latency.

Usage::

    python -m benchmarks.grounded_ann_bench
    python -m benchmarks.grounded_ann_bench --sizes 200 2000 --skip-real
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
ENGINE_DIMENSIONS = 384
ENGINE_QUERY_NOISE = 0.35
REAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DeterministicEmbedder:
    """Deterministic text→vector lookup standing in for a sentence model."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self._vectors = vectors

    def encode(self, text: str, normalize_embeddings: bool = False) -> np.ndarray:
        """Return the precomputed unit vector registered for ``text``."""
        return self._vectors[text]

    def __call__(self, text: str) -> np.ndarray:
        """Allow use as a plain ``embed_fn`` callable."""
        return self._vectors[text]


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of ``matrix``."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _build_engine_corpus(
    n_docs: int,
    n_queries: int,
    seed: int,
) -> tuple[list[str], list[str], DeterministicEmbedder, list[list[str]]]:
    """Build seeded docs, queries, their embedder, and exact ground truth."""
    rng = np.random.default_rng(seed)
    doc_vectors = _unit_rows(
        rng.normal(size=(n_docs, ENGINE_DIMENSIONS)).astype(np.float32),
    )
    doc_texts = [f"synthetic document {i}" for i in range(n_docs)]

    query_targets = rng.integers(0, n_docs, size=n_queries)
    noise = rng.normal(size=(n_queries, ENGINE_DIMENSIONS)).astype(np.float32)
    query_vectors = _unit_rows(
        doc_vectors[query_targets] + ENGINE_QUERY_NOISE * noise,
    )
    query_texts = [f"synthetic query {j}" for j in range(n_queries)]

    vectors: dict[str, np.ndarray] = {}
    for text, vector in zip(doc_texts, doc_vectors, strict=True):
        vectors[text] = vector
    for text, vector in zip(query_texts, query_vectors, strict=True):
        vectors[text] = vector

    similarities = query_vectors @ doc_vectors.T
    ground_truth = [
        [doc_texts[i] for i in np.argsort(-row)[:_ENGINE_TOP_K]] for row in similarities
    ]
    return doc_texts, query_texts, DeterministicEmbedder(vectors), ground_truth


_ENGINE_TOP_K = 10


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    """Summarise per-query latencies in milliseconds."""
    ordered = sorted(samples_ms)
    p95_index = max(int(round(0.95 * len(ordered))) - 1, 0)
    return {
        "mean_ms": round(statistics.fmean(ordered), 4),
        "p50_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[p95_index], 4),
    }


def _measure_engine_backend(
    backend: Any,
    doc_texts: list[str],
    query_texts: list[str],
    ground_truth: list[list[str]],
) -> dict[str, Any]:
    """Ingest the corpus, run all queries, and report latency + recall."""
    t0 = time.perf_counter()
    for i, text in enumerate(doc_texts):
        backend.add(f"doc-{i}", text)
    ingest_seconds = time.perf_counter() - t0

    latencies: list[float] = []
    recalls: list[float] = []
    for query, expected in zip(query_texts, ground_truth, strict=True):
        t0 = time.perf_counter()
        rows = backend.query(query, n_results=_ENGINE_TOP_K)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved = {row["text"] for row in rows}
        recalls.append(len(retrieved & set(expected)) / len(expected))

    return {
        "ingest_seconds": round(ingest_seconds, 4),
        "recall_at_10": round(statistics.fmean(recalls), 4),
        **_latency_summary(latencies),
    }


def run_engine_mode(sizes: list[int], n_queries: int, seed: int) -> dict[str, Any]:
    """Compare the shipped dense engines on seeded synthetic corpora."""
    from director_ai.core.vector_store import (
        FAISSBackend,
        SentenceTransformerBackend,
    )

    runs: list[dict[str, Any]] = []
    for n_docs in sizes:
        doc_texts, query_texts, embedder, ground_truth = _build_engine_corpus(
            n_docs,
            n_queries,
            seed,
        )
        ivf_nlist = max(int(np.sqrt(n_docs)), 1)
        arms = {
            "linear_scan": SentenceTransformerBackend(model=embedder),
            "faiss_flat": FAISSBackend(
                embed_fn=embedder,
                vector_size=ENGINE_DIMENSIONS,
            ),
            "faiss_ivf": FAISSBackend(
                embed_fn=embedder,
                vector_size=ENGINE_DIMENSIONS,
                index_type="ivf",
                ivf_nlist=ivf_nlist,
            ),
        }
        measurements = {
            name: _measure_engine_backend(
                backend,
                doc_texts,
                query_texts,
                ground_truth,
            )
            for name, backend in arms.items()
        }
        runs.append(
            {
                "n_docs": n_docs,
                "n_queries": n_queries,
                "ivf_nlist": ivf_nlist,
                "backends": measurements,
            },
        )
        print(f"\n  Engine corpus n_docs={n_docs} (recall@10 vs exact cosine)")
        for name, row in measurements.items():
            print(
                f"    {name:<12} recall={row['recall_at_10']:.3f}  "
                f"p50={row['p50_ms']:.3f} ms  p95={row['p95_ms']:.3f} ms",
            )

    return {
        "dimensions": ENGINE_DIMENSIONS,
        "query_noise": ENGINE_QUERY_NOISE,
        "seed": seed,
        "top_k": _ENGINE_TOP_K,
        "note": (
            "Seeded Gaussian vectors isolate engine cost; flat FAISS is exact "
            "search, so its recall matches the linear scan by construction. "
            "Unclustered random vectors are the worst case for IVF recall."
        ),
        "runs": runs,
    }


def _measure_recipe(
    store: Any, eval_set: list[tuple[str, str, str, list[str]]]
) -> dict[str, Any]:
    """Run the retrieval_bench evaluation queries through one recipe store."""
    latencies: list[float] = []
    hits_1 = 0
    hits_3 = 0
    precision_sum = 0.0
    for _, _, query, relevant_keys in eval_set:
        t0 = time.perf_counter()
        rows = store.backend.query(query, n_results=3)
        latencies.append((time.perf_counter() - t0) * 1000)
        retrieved_keys = [
            row.get("metadata", {}).get("key", "")
            for row in rows
            if row.get("metadata", {}).get("key", "")
        ]
        hits_1 += bool(retrieved_keys and retrieved_keys[0] in relevant_keys)
        overlap = len(set(retrieved_keys) & set(relevant_keys))
        hits_3 += bool(overlap)
        precision_sum += overlap / max(len(retrieved_keys), 1)

    n = len(eval_set)
    return {
        "hit_at_1": round(hits_1 / n, 4),
        "hit_at_3": round(hits_3 / n, 4),
        "precision_at_3": round(precision_sum / n, 4),
        **_latency_summary(latencies),
    }


def run_real_mode() -> dict[str, Any]:
    """Compare the pre/post-WCA-7 recipes end-to-end with real local models."""
    from benchmarks.retrieval_bench import DISTRACTORS, EVAL_SET
    from director_ai.core.vector_store import VectorGroundTruthStore

    recipes = {
        "pre_wca7_default": {"use_ann": False, "use_reranker": False},
        "ann_only": {"use_reranker": False},
        "new_default_ann_rerank": {},
    }
    arms: dict[str, Any] = {}
    for name, overrides in recipes.items():
        t0 = time.perf_counter()
        store = VectorGroundTruthStore.grounded(
            embedding_model=REAL_EMBEDDING_MODEL,
            **overrides,
        )
        for key, value, _, _ in EVAL_SET:
            store.add_fact(key, value)
        for key, value in DISTRACTORS:
            store.add_fact(key, value)
        build_seconds = time.perf_counter() - t0
        arms[name] = {
            "backend_chain": _describe_backend_chain(store.backend),
            "build_seconds": round(build_seconds, 2),
            **_measure_recipe(store, EVAL_SET),
        }
        print(
            f"\n  Recipe {name}: chain={arms[name]['backend_chain']}\n"
            f"    hit@1={arms[name]['hit_at_1']:.3f}  "
            f"hit@3={arms[name]['hit_at_3']:.3f}  "
            f"p50={arms[name]['p50_ms']:.2f} ms  p95={arms[name]['p95_ms']:.2f} ms",
        )

    return {
        "embedding_model": REAL_EMBEDDING_MODEL,
        "n_facts": len(EVAL_SET) + len(DISTRACTORS),
        "n_queries": len(EVAL_SET),
        "note": (
            "retrieval_bench EVAL_SET; embedding model overridden to the "
            "small cached MiniLM so the benchmark runs offline. The reranker "
            "is the recipe default cross-encoder."
        ),
        "arms": arms,
    }


def _describe_backend_chain(backend: Any) -> str:
    """Render the decorator chain of a recipe backend for the report."""
    parts = []
    current = backend
    while current is not None:
        parts.append(type(current).__name__)
        current = getattr(current, "_base", None)
    return " -> ".join(parts)


def _environment() -> dict[str, Any]:
    """Record the library versions and host conditions."""
    import faiss

    from benchmarks.host_conditions import host_conditions

    env: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "host_conditions": host_conditions(),
    }
    try:
        import sentence_transformers

        env["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        env["sentence_transformers"] = None
    return env


def main() -> None:
    """Run the benchmark and write the JSON artefact."""
    parser = argparse.ArgumentParser(
        description="grounded() ANN + rerank latency/recall benchmark",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[200, 2000, 20000],
        help="engine-mode corpus sizes (default: 200 2000 20000)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="engine-mode queries per corpus (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260711,
        help="engine-mode RNG seed (default: 20260711)",
    )
    parser.add_argument(
        "--skip-real",
        action="store_true",
        help="skip the real-model recipe comparison",
    )
    args = parser.parse_args()

    output: dict[str, Any] = {
        "benchmark": "grounded_ann_bench",
        "environment": _environment(),
        "engine": run_engine_mode(args.sizes, args.queries, args.seed),
    }
    if args.skip_real:
        output["real"] = {"skipped": True, "reason": "--skip-real"}
    else:
        output["real"] = run_real_mode()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "grounded_ann_bench.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
