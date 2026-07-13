# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — internal curated-KB fusion A/B (WCA-9 default decision)

"""Fusion A/B on the internal curated-KB evaluation set.

The BEIR grid (``beir_fusion_bench``) showed ``convex`` s30/d70 beating
the shipped weighted-RRF default on both public datasets. ``grounded()``
however primarily targets curated fact KBs, and the shipped default
chain includes the ms-marco reranker — so the default-flip decision
needs this measurement, not only BEIR: fusion arms on the internal
``retrieval_bench`` EVAL_SET + DISTRACTORS corpus, each arm measured
both unreranked and under the default cross-encoder.

One bge-large index is built once; fusion arms are
``HybridBackend.with_fusion()`` views over it, and reranked arms wrap
the same view in ``RerankedBackend`` — identical corpus embeddings
across every arm by construction.

Usage::

    python -m benchmarks.retrieval_fusion_internal_ab
"""

from __future__ import annotations

import gc
import json
import time
from typing import Any

from benchmarks.grounded_ann_bench import RESULTS_DIR
from benchmarks.retrieval_model_refresh_ab import (
    BGE_LARGE,
    MSMARCO,
    _cached_revision,
    _environment,
    _measure_recipe,
)

# arm name -> (fusion method, sparse weight, dense weight, reranked)
ARMS: dict[str, tuple[str, float, float, bool]] = {
    "rrf_s50_d50__none": ("rrf", 1.0, 1.0, False),
    "convex_s30_d70__none": ("convex", 0.3, 0.7, False),
    "convex_s50_d50__none": ("convex", 0.5, 0.5, False),
    "rrf_s50_d50__msmarco": ("rrf", 1.0, 1.0, True),
    "convex_s30_d70__msmarco": ("convex", 0.3, 0.7, True),
    "convex_s50_d50__msmarco": ("convex", 0.5, 0.5, True),
}


def main() -> None:
    """Run the fusion arms over one shared index and write the artefact."""
    from benchmarks.retrieval_bench import DISTRACTORS, EVAL_SET
    from director_ai.core.vector_store import (
        RerankedBackend,
        VectorGroundTruthStore,
    )

    t0 = time.perf_counter()
    store = VectorGroundTruthStore.grounded(
        embedding_model=BGE_LARGE,
        use_reranker=False,
    )
    for key, value, _, _ in EVAL_SET:
        store.add_fact(key, value)
    for key, value in DISTRACTORS:
        store.add_fact(key, value)
    build_seconds = round(time.perf_counter() - t0, 2)
    hybrid = store.backend

    reranked_cache: dict[str, Any] = {}
    arms: dict[str, Any] = {}
    for name, (method, sparse_w, dense_w, reranked) in ARMS.items():
        view = hybrid.with_fusion(
            method,
            sparse_weight=sparse_w,
            dense_weight=dense_w,
        )
        backend: Any = view
        if reranked:
            backend = RerankedBackend(base=view, reranker_model=MSMARCO)
            reranked_cache[name] = backend
        probe = VectorGroundTruthStore(backend=backend)
        arms[name] = {
            "fusion_method": method,
            "sparse_weight": sparse_w,
            "dense_weight": dense_w,
            "reranker_model": MSMARCO if reranked else None,
            **_measure_recipe(probe, EVAL_SET),
        }
        result = arms[name]
        print(
            f"  {name:<26} hit@1={result['hit_at_1']:.3f}  "
            f"hit@3={result['hit_at_3']:.3f}  "
            f"p50={result['p50_ms']:.1f} ms",
        )
        reranked_cache.pop(name, None)
        gc.collect()

    output = {
        "benchmark": "retrieval_fusion_internal_ab",
        "note": (
            "Fusion A/B on the internal retrieval_bench EVAL_SET + "
            "DISTRACTORS corpus through grounded() (bge-large), one "
            "shared index, with_fusion() views, each arm measured "
            "unreranked and under the default ms-marco cross-encoder. "
            "Companion to results/beir_fusion_bench.json for the "
            "fusion-default decision."
        ),
        "environment": _environment(),
        "embedding_model": BGE_LARGE,
        "embedding_revision": _cached_revision(BGE_LARGE),
        "reranker_revision": _cached_revision(MSMARCO),
        "n_facts": len(EVAL_SET) + len(DISTRACTORS),
        "n_queries": len(EVAL_SET),
        "build_seconds": build_seconds,
        "arms": arms,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "retrieval_fusion_internal_ab.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
