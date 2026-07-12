# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — embedder/reranker refresh A/B benchmark (WCA-8)

"""A/B evidence for refreshing the ``grounded()`` default model pair.

Compares the shipped defaults against candidate replacements on the
``retrieval_bench`` evaluation set, all through the public
``VectorGroundTruthStore.grounded()`` recipe so every arm exercises the
exact retrieval path users get:

* embedders — ``all-MiniLM-L6-v2`` (small baseline), ``bge-large-en-v1.5``
  (current ``RECOMMENDED_EMBEDDING_MODEL``), ``bge-m3`` (candidate);
* rerankers — none, ``ms-marco-MiniLM-L-6-v2`` (current
  ``RECOMMENDED_RERANKER_MODEL``), ``bge-reranker-v2-m3`` (candidate);
* mixed arms attribute any gain to the embedder or the reranker alone.

Every arm reports hit@1/hit@3/precision@3 and per-query latency; the JSON
artefact records the exact locally cached model revisions and the host
environment. Models must already be in the local Hugging Face cache — the
benchmark runs with ``HF_HUB_OFFLINE=1`` semantics in mind and never
triggers a download by itself when the cache is warm.

Usage::

    python -m benchmarks.retrieval_model_refresh_ab
    python -m benchmarks.retrieval_model_refresh_ab --arms bge_m3__bge_v2_m3
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from benchmarks.grounded_ann_bench import (
    _describe_backend_chain,
    _measure_recipe,
)

RESULTS_DIR = Path(__file__).parent / "results"

MINILM = "sentence-transformers/all-MiniLM-L6-v2"
BGE_LARGE = "BAAI/bge-large-en-v1.5"
BGE_M3 = "BAAI/bge-m3"
MSMARCO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"

# arm name -> (embedding model, reranker model or None)
ARMS: dict[str, tuple[str, str | None]] = {
    "minilm__none": (MINILM, None),
    "bge_large__none": (BGE_LARGE, None),
    "bge_m3__none": (BGE_M3, None),
    "minilm__msmarco": (MINILM, MSMARCO),
    "bge_large__msmarco": (BGE_LARGE, MSMARCO),
    "bge_m3__msmarco": (BGE_M3, MSMARCO),
    "minilm__bge_v2_m3": (MINILM, BGE_RERANKER),
    "bge_m3__bge_v2_m3": (BGE_M3, BGE_RERANKER),
}


def _cached_revision(model_id: str) -> str | None:
    """Return the locally cached git revision of ``model_id``, if any."""
    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path.home() / ".cache" / "huggingface" / "hub",
        ),
    )
    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        return ref_main.read_text(encoding="utf-8").strip()
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        revisions = [p.name for p in snapshots.iterdir() if p.is_dir()]
        if len(revisions) == 1:
            return revisions[0]
    return None


def _cpu_model() -> str | None:
    """Read the CPU model name from ``/proc/cpuinfo`` when available."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def _environment() -> dict[str, Any]:
    """Record the hardware, library versions and host conditions."""
    import faiss
    import numpy as np
    import sentence_transformers
    import torch

    from benchmarks.host_conditions import host_conditions
    from director_ai.core._device import select_torch_device

    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "torch_device": select_torch_device(),
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "host_conditions": host_conditions(),
    }


def _run_arm(
    name: str,
    embedding_model: str,
    reranker_model: str | None,
    eval_set: list[tuple[str, str, str, list[str]]],
    distractors: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build one ``grounded()`` store, index the corpus, and measure it."""
    from director_ai.core.vector_store import VectorGroundTruthStore

    overrides: dict[str, Any] = {"use_reranker": reranker_model is not None}
    if reranker_model is not None:
        overrides["reranker_model"] = reranker_model
    t0 = time.perf_counter()
    store = VectorGroundTruthStore.grounded(
        embedding_model=embedding_model,
        **overrides,
    )
    for key, value, _, _ in eval_set:
        store.add_fact(key, value)
    for key, value in distractors:
        store.add_fact(key, value)
    build_seconds = time.perf_counter() - t0

    result = {
        "embedding_model": embedding_model,
        "embedding_revision": _cached_revision(embedding_model),
        "reranker_model": reranker_model,
        "reranker_revision": (
            _cached_revision(reranker_model) if reranker_model else None
        ),
        "backend_chain": _describe_backend_chain(store.backend),
        "build_seconds": round(build_seconds, 2),
        **_measure_recipe(store, eval_set),
    }
    print(
        f"\n  Arm {name}: chain={result['backend_chain']}\n"
        f"    hit@1={result['hit_at_1']:.3f}  hit@3={result['hit_at_3']:.3f}  "
        f"p@3={result['precision_at_3']:.3f}  "
        f"p50={result['p50_ms']:.1f} ms  p95={result['p95_ms']:.1f} ms",
    )
    del store
    gc.collect()
    return result


def main() -> None:
    """Run the requested arms and write the JSON artefact."""
    parser = argparse.ArgumentParser(
        description="grounded() embedder/reranker refresh A/B benchmark",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=sorted(ARMS),
        default=sorted(ARMS),
        help="subset of arms to run (default: all)",
    )
    args = parser.parse_args()

    from benchmarks.retrieval_bench import DISTRACTORS, EVAL_SET

    arms: dict[str, Any] = {}
    for name in args.arms:
        embedding_model, reranker_model = ARMS[name]
        arms[name] = _run_arm(
            name,
            embedding_model,
            reranker_model,
            EVAL_SET,
            DISTRACTORS,
        )

    output: dict[str, Any] = {
        "benchmark": "retrieval_model_refresh_ab",
        "environment": _environment(),
        "n_facts": len(EVAL_SET) + len(DISTRACTORS),
        "n_queries": len(EVAL_SET),
        "note": (
            "retrieval_bench EVAL_SET through the public grounded() recipe; "
            "arm name = <embedder>__<reranker>. Mixed arms attribute gains "
            "to one component. Internal evaluation set — these numbers are "
            "not comparable across systems and are not competitive claims."
        ),
        "arms": arms,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "retrieval_model_refresh_ab.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
