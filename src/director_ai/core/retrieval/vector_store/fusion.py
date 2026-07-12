# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Retrieval run fusion strategies

"""Fusion strategies for combining sparse and dense retrieval runs.

Each strategy takes two scored runs — sparse (BM25) and dense
(embedding similarity) — and returns one fused ranking. Runs are
lists of ``(row, score)`` pairs in descending native-score order,
where ``row`` is the backend result dict and ``score`` is the run's
own relevance scale (BM25 score, cosine similarity). Document ids
must be unique within a run.

Methods
-------
``rrf``
    Weighted Reciprocal Rank Fusion:
    ``score(d) = Σ_run w_run / (k + rank_run(d))`` with 1-based ranks
    (Cormack, Clarke & Büttcher, SIGIR 2009; ``k = 60`` was fixed in
    their pilot study). Rank-only — native scores are ignored.
``convex``
    Convex combination of min-max normalised scores:
    ``score(d) = ws'·minmax_sparse(d) + wd'·minmax_dense(d)`` where
    ``ws' + wd' = 1`` (CombSUM family, Fox & Shaw 1994, with the
    weights normalised to a convex combination).
``combmnz``
    CombMNZ: the ``convex`` score multiplied by the number of runs
    that retrieved the document (Fox & Shaw 1994) — rewards
    cross-run agreement.
``zscore``
    Convex-weighted sum of z-score standardised scores. A run with
    zero score variance contributes nothing (its z-scores are all
    zero) — presence in such a run carries no signal.

Normalisation conventions (documented behaviour, exercised in tests):
min-max over a constant run maps every score to 1.0 so presence still
counts; z-score over a constant run maps every score to 0.0.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "FUSION_METHODS",
    "ScoredRun",
    "fuse_results",
    "validate_fusion_method",
]

FUSION_METHODS: tuple[str, ...] = ("rrf", "convex", "combmnz", "zscore")

ScoredRun = list[tuple[dict[str, Any], float]]


def validate_fusion_method(method: object) -> str:
    """Return the canonical method name or raise ``ValueError``."""
    if not isinstance(method, str):
        raise ValueError("fusion_method must be a string")
    canonical = method.strip().lower()
    if canonical not in FUSION_METHODS:
        raise ValueError(
            f"fusion_method must be one of {FUSION_METHODS}, got {method!r}",
        )
    return canonical


def fuse_results(
    method: str,
    sparse: ScoredRun,
    dense: ScoredRun,
    *,
    rrf_k: int = 60,
    sparse_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[dict[str, Any]]:
    """Fuse two scored runs into one ranking of result rows.

    Rows for the same document id are merged last-write-wins (the
    dense row replaces the sparse row), matching the historical
    ``HybridBackend`` behaviour. The full fused ranking is returned;
    callers truncate to their own ``n_results``.
    """
    canonical = validate_fusion_method(method)
    if not isinstance(rrf_k, int) or isinstance(rrf_k, bool):
        raise ValueError("rrf_k must be an integer")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")
    sparse_weight = _validate_weight(sparse_weight, "sparse_weight")
    dense_weight = _validate_weight(dense_weight, "dense_weight")
    total_weight = sparse_weight + dense_weight
    if total_weight == 0.0:
        raise ValueError("at least one fusion weight must be positive")

    runs: tuple[tuple[ScoredRun, float], ...] = (
        (sparse, sparse_weight),
        (dense, dense_weight),
    )
    fused: dict[str, float] = {}
    hits: dict[str, int] = {}
    row_map: dict[str, dict[str, Any]] = {}

    for run, weight in runs:
        contributions = _run_contributions(canonical, run, weight, total_weight, rrf_k)
        for (row, _score), contribution in zip(run, contributions, strict=True):
            doc_id = str(row["id"])
            fused[doc_id] = fused.get(doc_id, 0.0) + contribution
            hits[doc_id] = hits.get(doc_id, 0) + 1
            row_map[doc_id] = row

    if canonical == "combmnz":
        fused = {doc_id: score * hits[doc_id] for doc_id, score in fused.items()}

    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [row_map[doc_id] for doc_id, _ in ranked]


def _run_contributions(
    method: str,
    run: ScoredRun,
    weight: float,
    total_weight: float,
    rrf_k: int,
) -> list[float]:
    """Per-document fusion contributions for one run, in run order."""
    if method == "rrf":
        return [weight / (rrf_k + rank + 1) for rank in range(len(run))]
    scores = [score for _, score in run]
    convex_weight = weight / total_weight
    # convex and combmnz share the min-max CombSUM core
    normalised = _zscores(scores) if method == "zscore" else _minmax(scores)
    return [convex_weight * value for value in normalised]


def _minmax(scores: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]; a constant run maps to all 1.0."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high == low:
        return [1.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def _zscores(scores: list[float]) -> list[float]:
    """Standardise scores; a zero-variance run maps to all 0.0."""
    if not scores:
        return []
    count = len(scores)
    mean = sum(scores) / count
    variance = sum((score - mean) ** 2 for score in scores) / count
    if variance == 0.0:
        return [0.0] * count
    deviation = variance**0.5
    return [(score - mean) / deviation for score in scores]


def _validate_weight(value: float, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{field_name} must be non-negative")
    return value
