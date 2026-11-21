# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AtlasKV fallback diagnostic

"""Count why the pruned AtlasKV arm fell back to plain key search.

The pruned arm falls back when the best entity group's similarity is below
``tau`` or when its groups yield fewer than ``top_k`` items. The comparison
harness records only whether it fell back. This diagnostic answers the same
queries again with the same index and splits each fallback by condition. It
is a secondary diagnostic outside the frozen harness and decides nothing.

Each query's two conditions are recomputed from the index's own parts, and
the run refuses to report if their union differs from the fallback the
index itself returns for that query.

Run it on the host and split of the comparison::

    python -m benchmarks.atlaskv_fallback_diagnostic --split test --g 2 --tau 0.4
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from benchmarks.atlaskv_corpus import Query, build_corpus
from benchmarks.atlaskv_retrieval_bench import TOP_K, ArmIndex, build_arm, load_encoder

CONDITIONS = ("similarity_below_tau_only", "too_few_items_only", "both", "neither")


def conditions(
    index: ArmIndex, query: Query, *, g: int, tau: float
) -> tuple[bool, bool]:
    """Return whether the best group is below ``tau`` and whether too few items came.

    Parameters
    ----------
    index : ArmIndex
        A built ``pruned`` arm.
    query : Query
        The query to answer.
    g, tau : int, float
        The number of groups and the similarity threshold.

    Returns
    -------
    tuple[bool, bool]
        ``(best < tau, fewer than top_k items)``, as the arm computes them.

    Raises
    ------
    ValueError
        If the index is not a ``pruned`` arm.
    """
    if index.names is None:
        raise ValueError("only the pruned arm has entity groups")
    groups = [
        hit["id"]
        for hit in index.names.query(query.text, n_results=g, tenant_id=query.tenant_id)
    ]
    vector = index._encode(query.text)
    index.vectors -= 1
    best = max(
        (float(vector @ index.name_vectors[group]) for group in groups), default=-1.0
    )
    keys = sorted(
        (key for group in groups for key in index.entity_keys[group]),
        key=lambda key: -float(vector @ index.key_vectors[key]),
    )
    return best < tau, len(index._expand(keys, TOP_K)) < TOP_K


def count_conditions(
    index: ArmIndex, queries: Sequence[Query], *, g: int, tau: float
) -> dict[str, object]:
    """Split the pruned arm's fallbacks on ``queries`` by condition.

    Parameters
    ----------
    index : ArmIndex
        A built ``pruned`` arm.
    queries : Sequence[Query]
        The queries of one split.
    g, tau : int, float
        The parameters of the comparison.

    Returns
    -------
    dict[str, object]
        The query count, the fallback count, and the count of each condition.

    Raises
    ------
    RuntimeError
        If the recomputed conditions disagree with the arm's own fallback.
    """
    counts = dict.fromkeys(CONDITIONS, 0)
    for query in queries:
        low, few = conditions(index, query, g=g, tau=tau)
        if index.query(query, g=g, tau=tau).fallback != (low or few):
            raise RuntimeError(f"recomputed conditions disagree on {query.query_id}")
        name = (
            "both"
            if low and few
            else (
                "similarity_below_tau_only"
                if low
                else "too_few_items_only"
                if few
                else "neither"
            )
        )
        counts[name] += 1
    return {
        "queries": len(queries),
        "fallbacks": len(queries) - counts["neither"],
        "by_condition": counts,
    }


def main(
    argv: Sequence[str] | None = None,
    load: Callable[[str], tuple[object, str]] = load_encoder,
) -> dict[str, object]:
    """Build the pruned arm, split its fallbacks by condition and write the result.

    Parameters
    ----------
    argv : Sequence[str] or None, optional
        Command-line arguments.
    load : Callable[[str], tuple[object, str]], optional
        Returns the encoder and its revision.

    Returns
    -------
    dict[str, object]
        The diagnostic artefact.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--g", type=int, required=True)
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--entities", type=int, default=400)
    parser.add_argument("--similar-pairs", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    corpus = build_corpus(entities=args.entities, similar_pairs=args.similar_pairs)
    encoder, revision = load(args.encoder)
    index, _ = build_arm(corpus.snapshot, encoder, "pruned")
    queries = [query for query in corpus.queries if query.split == args.split]
    artefact: dict[str, object] = {
        "schema": "director-ai.atlaskv-fallback-diagnostic.v1",
        "split": args.split,
        "g": args.g,
        "tau": args.tau,
        "encoder": args.encoder,
        "encoder_revision": revision,
        **count_conditions(index, queries, g=args.g, tau=args.tau),
    }
    if args.output is not None:
        args.output.write_text(json.dumps(artefact, indent=2, sort_keys=True) + "\n")
    return artefact


if __name__ == "__main__":
    main()
