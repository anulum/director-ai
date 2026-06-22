# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — keyword-fallback recall/precision benchmark

"""Recall/precision of the keyword fallback for API-ingested chunks.

API-ingested chunks are keyed by an opaque cid (``"doc:chunk:0"``) with no query
words, so the original key-only ``GroundTruthStore.retrieve_context`` could never
find them by content (recall 0). This benchmark quantifies the value-aware
fallback against a key-only baseline on a small labelled corpus, so the
precision/recall trade-off of the stopword-guarded value match is explicit.

Run: ``python benchmarks/keyword_fallback_recall.py``
"""

from __future__ import annotations

from director_ai.core.retrieval.knowledge import _STOPWORDS, GroundTruthStore

# Labelled corpus: cid-keyed ingested chunks (as the ingestion API stores them).
_CHUNKS: dict[str, str] = {
    "billing:chunk:0": "Refunds are issued within fourteen days of the request.",
    "billing:chunk:1": "Invoices are sent on the first business day of the month.",
    "security:chunk:0": "All passwords are hashed with a salted SHA-256 digest.",
    "security:chunk:1": "Two-factor authentication is mandatory for admin accounts.",
    "weather:chunk:0": "The forecast predicts clear skies and a light breeze.",
    "shipping:chunk:0": "Standard shipping takes three to five working days.",
    "shipping:chunk:1": "Express shipping delivers the parcel the next morning.",
    "support:chunk:0": "Support tickets are answered in the order they arrive.",
}

# Query -> the cids that genuinely answer it.
_QUERIES: list[tuple[str, set[str]]] = [
    ("how long do refunds take", {"billing:chunk:0"}),
    ("when are invoices sent", {"billing:chunk:1"}),
    ("how are passwords stored", {"security:chunk:0"}),
    ("is two-factor authentication required", {"security:chunk:1"}),
    ("how fast is express shipping", {"shipping:chunk:1"}),
    ("what is the weather forecast", {"weather:chunk:0"}),
]

_TOP_K = 3


def _key_only_hits(store: GroundTruthStore, query: str) -> set[str]:
    """Reproduce the original key-only matching (the baseline)."""
    query_lower = query.lower()
    hits: set[str] = set()
    for key, _value in store.facts.items():
        if any(word in query_lower for word in key.lower().split()):
            hits.add(key)
    return hits


def _value_aware_hits(store: GroundTruthStore, query: str, *, top_k: int) -> set[str]:
    """The shipped value-aware retrieval, mapped back to the cids it returned."""
    retrieved = store.retrieve_context(query, top_k=top_k)
    if not retrieved:
        return set()
    returned_values = set(retrieved.split("; "))
    return {key for key, value in store.facts.items() if value in returned_values}


def _score(hits_for_query, queries, *, top_k: int) -> tuple[float, float]:
    """Return ``(recall@k, precision@k)`` macro-averaged over the queries."""
    recalls: list[float] = []
    precisions: list[float] = []
    for query, relevant in queries:
        hits = hits_for_query(query)
        # Cap the predicted set at top_k for an honest precision@k.
        capped = set(list(hits)[:top_k])
        true_positives = len(capped & relevant)
        recalls.append(true_positives / len(relevant) if relevant else 0.0)
        precisions.append(true_positives / len(capped) if capped else 0.0)
    return (
        sum(recalls) / len(recalls),
        sum(precisions) / len(precisions),
    )


def main() -> None:
    """Print the recall/precision table for both retrieval strategies."""
    store = GroundTruthStore()
    store.facts.update(_CHUNKS)

    key_recall, key_precision = _score(
        lambda q: _key_only_hits(store, q), _QUERIES, top_k=_TOP_K
    )
    val_recall, val_precision = _score(
        lambda q: _value_aware_hits(store, q, top_k=_TOP_K), _QUERIES, top_k=_TOP_K
    )

    print(
        f"Keyword fallback over {len(_CHUNKS)} ingested chunks, "
        f"{len(_QUERIES)} queries, k={_TOP_K}, |stopwords|={len(_STOPWORDS)}"
    )
    print(f"{'strategy':<16}{'recall@k':>12}{'precision@k':>14}")
    print(f"{'key-only':<16}{key_recall:>12.3f}{key_precision:>14.3f}")
    print(f"{'value-aware':<16}{val_recall:>12.3f}{val_precision:>14.3f}")


if __name__ == "__main__":
    main()
