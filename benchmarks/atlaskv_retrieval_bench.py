# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — AtlasKV retrieval comparison

"""CPU comparison of text retrieval and entity-relation keys, without training.

Three arms run on one snapshot, one encoder, one ``top_k`` and one process:

* ``hybrid``: :class:`HybridBackend` over the evidence texts, the current
  default;
* ``keys``: flat ``entity · relation`` keys with the same hybrid fusion,
  each key returning its evidence, one paraphrase per fact first;
* ``pruned``: the keys grouped by entity. An entity-name index picks ``g``
  groups and only their keys are ranked, by dense cosine similarity. It
  falls back to ``keys`` when the best group's similarity is below ``tau``
  or the shortlist holds fewer than ``top_k`` documents.

Every arm queries inside the tenant's scope, drops retracted documents
before indexing, and checks the tenant, retraction and snapshot of each
result after reading it; any violation is counted, never hidden. The
acceptance rules are those of the preregistration, whose digest the
artefact records.

Usage::

    python -m benchmarks.atlaskv_retrieval_bench --split dev
    python -m benchmarks.atlaskv_retrieval_bench --split test --g 4 --tau 0.3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import resource
import statistics
import time
import tracemalloc
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import numpy as np

from benchmarks.atlaskv_corpus import (
    RELATIONS,
    Corpus,
    Evidence,
    Query,
    Snapshot,
    build_corpus,
)
from director_ai.core.retrieval.vector_store.composite import HybridBackend
from director_ai.core.retrieval.vector_store.embedding import SentenceTransformerBackend

TOP_K = 5


class _Encoder(Protocol):
    """The sentence encoder the arms share: one text in, one vector out."""

    def encode(self, text: str, normalize_embeddings: bool = ...) -> object:
        """Return the embedding vector for ``text``."""


GRID_G = (2, 4, 8)
GRID_TAU = (0.2, 0.3, 0.4)
TIMED_RUNS = 3
WARMUP_QUERIES = 20
ARMS = ("hybrid", "keys", "pruned")
RESULTS_DIR = Path(__file__).parent / "results"
Row = dict[str, float | bool | int]
"""One query's measurements in one arm."""


class HashingEmbedder:
    """Deterministic character-trigram embedder for tests and smoke runs."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def encode(self, text: str, normalize_embeddings: bool = True) -> np.ndarray:
        """Return a unit vector of hashed character trigrams of ``text``."""
        vector = np.zeros(self.dimensions, dtype=np.float32)
        padded = f"  {text.lower()}  "
        for index in range(len(padded) - 2):
            token = padded[index : index + 3].encode("utf-8")
            bucket = int.from_bytes(hashlib.sha256(token).digest()[:4], "little")
            vector[bucket % self.dimensions] += 1.0
        return (
            vector / float(np.linalg.norm(vector)) if normalize_embeddings else vector
        )


@dataclass
class Retrieved:
    """One arm's answer to one query, with its boundary checks."""

    docs: list[Evidence]
    seconds: float
    fallback: bool = False
    leaks: dict[str, int] = field(default_factory=dict)


def _noun(entity_type: str, relation: str) -> str:
    """Return the display noun of a relation."""
    return next(noun for name, noun, _, _ in RELATIONS[entity_type] if name == relation)


class ArmIndex:
    """One arm's index over one snapshot, queried inside a tenant's scope."""

    def __init__(self, snapshot: Snapshot, encoder: object, arm: str) -> None:
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.snapshot = snapshot
        self.encoder = encoder
        self.current = {item.doc_id: item for item in snapshot.evidence}
        self.valid = {
            doc_id: item for doc_id, item in self.current.items() if not item.retracted
        }
        self.backends: list[HybridBackend] = []
        self.vectors = 0
        meta = {"snapshot_id": snapshot.snapshot_id}
        if arm == "hybrid":
            self.text = self._backend()
            for item in self.valid.values():
                self.text.add(
                    item.doc_id, item.text, {**meta, "tenant_id": item.tenant_id}
                )
            return
        facts = {fact.fact_id: fact for fact in snapshot.facts}
        self.key_docs: dict[str, list[Evidence]] = {}
        for item in self.valid.values():
            self.key_docs.setdefault(
                f"{item.tenant_id}|{item.entity_id}|{item.relation}", []
            ).append(item)
        self.keys = self._backend()
        self.entity_keys: dict[str, list[str]] = {}
        self.key_vectors: dict[str, np.ndarray] = {}
        self.name_vectors: dict[str, np.ndarray] = {}
        self.names = self._backend() if arm == "pruned" else None
        for key, docs in self.key_docs.items():
            docs.sort(key=lambda d: (d.doc_id.rsplit(":p", 1)[1], d.source_id))
            tenant, entity, relation = key.split("|")
            fact = facts[docs[0].fact_id]
            text = f"{fact.entity_name} · {_noun(fact.entity_type, relation)}"
            self.keys.add(key, text, {**meta, "tenant_id": tenant})
            if self.names is None:
                continue
            self.key_vectors[key] = self._encode(text)
            group = f"{tenant}|{entity}"
            if group not in self.entity_keys:
                self.names.add(group, fact.entity_name, {**meta, "tenant_id": tenant})
                self.name_vectors[group] = self._encode(fact.entity_name)
            self.entity_keys.setdefault(group, []).append(key)

    def _backend(self) -> HybridBackend:
        """Return a new hybrid backend over the shared encoder."""
        backend = HybridBackend(
            SentenceTransformerBackend(model=cast(_Encoder, self.encoder))
        )
        self.backends.append(backend)
        return backend

    def _encode(self, text: str) -> np.ndarray:
        """Return the encoder's unit vector for ``text``."""
        self.vectors += 1
        vector = cast(_Encoder, self.encoder).encode(text, normalize_embeddings=True)
        return np.asarray(vector)

    def index_bytes(self, dimensions: int) -> int:
        """Return the arm's stored vectors and texts, in bytes."""
        docs = sum(backend.count() for backend in self.backends) + self.vectors
        texts = (
            sum(len(item.text.encode("utf-8")) for item in self.valid.values())
            if self.arm == "hybrid"
            else 0
        )
        keys = sum(len(key.encode("utf-8")) for key in getattr(self, "key_docs", {}))
        return docs * dimensions * 4 + texts + keys

    def _checked(
        self, hits: Sequence[tuple[str, str]], tenant: str
    ) -> tuple[list[Evidence], dict[str, int]]:
        """Read results back and count any tenant, retraction or snapshot leak."""
        leaks = {"tenant": 0, "retracted": 0, "snapshot": 0}
        kept: list[Evidence] = []
        for doc_id, snapshot_id in hits:
            item = self.current.get(doc_id)
            if item is None or snapshot_id != self.snapshot.snapshot_id:
                leaks["snapshot"] += 1
            elif item.retracted:
                leaks["retracted"] += 1
            elif item.tenant_id != tenant:
                leaks["tenant"] += 1
            else:
                kept.append(item)
        return kept, leaks

    def _expand(self, keys: Sequence[str], top_k: int) -> list[tuple[str, str]]:
        """Return the evidence of ranked keys, one paraphrase per fact first."""
        out: list[tuple[str, str]] = []
        for key in keys:
            for item in self.key_docs[key]:
                if len(out) < top_k:
                    out.append((item.doc_id, self.snapshot.snapshot_id))
        return out

    def query(
        self, query: Query, *, g: int, tau: float, top_k: int = TOP_K
    ) -> Retrieved:
        """Answer one query and check what the arm returns."""
        start = time.perf_counter()
        fallback = False
        if self.arm == "hybrid":
            found = self.text.query(
                query.text, n_results=top_k, tenant_id=query.tenant_id
            )
            hits = [
                (hit["id"], hit["metadata"].get("snapshot_id", "")) for hit in found
            ]
        else:
            hits = []
            if self.names is not None:
                groups = [
                    hit["id"]
                    for hit in self.names.query(
                        query.text, n_results=g, tenant_id=query.tenant_id
                    )
                ]
                vector = self._encode(query.text)
                self.vectors -= 1
                best = max(
                    (float(vector @ self.name_vectors[group]) for group in groups),
                    default=-1.0,
                )
                keys = sorted(
                    (key for group in groups for key in self.entity_keys[group]),
                    key=lambda key: -float(vector @ self.key_vectors[key]),
                )
                hits = self._expand(keys, top_k)
                fallback = best < tau or len(hits) < top_k
            if self.names is None or fallback:
                ranked = [
                    hit["id"]
                    for hit in self.keys.query(
                        query.text, n_results=top_k, tenant_id=query.tenant_id
                    )
                ]
                hits = self._expand(ranked, top_k)
        docs, leaks = self._checked(hits, query.tenant_id)
        return Retrieved(docs, time.perf_counter() - start, fallback, leaks)


def relevant_docs(query: Query, docs: Sequence[Evidence]) -> list[Evidence]:
    """Return the documents that name the queried entity and state the relation.

    An item is about the entity when the entity's name appears in its text
    as a whole phrase, as a text pipeline would decide. A similar-name
    partner's items therefore count, which is the confusion the false
    claims test.

    Parameters
    ----------
    query : Query
        The question.
    docs : Sequence[Evidence]
        The returned documents.

    Returns
    -------
    list[Evidence]
        The documents the value check reads.
    """
    pattern = re.compile(r"(?<!\w)" + re.escape(query.entity_name) + r"(?!\w)")
    return [d for d in docs if d.relation == query.relation and pattern.search(d.text)]


def value_check(query: Query, docs: Sequence[Evidence], value: str) -> bool:
    """Approve ``value`` only when a relevant document states it and none states another.

    Parameters
    ----------
    query : Query
        The question.
    docs : Sequence[Evidence]
        The returned documents.
    value : str
        The claimed value.

    Returns
    -------
    bool
        Whether the claim is approved.
    """
    relevant = relevant_docs(query, docs)
    return any(d.value == value for d in relevant) and all(
        d.value == value for d in relevant
    )


def score_query(query: Query, result: Retrieved) -> dict[str, float | bool | int]:
    """Return the per-query measurements of one arm.

    Parameters
    ----------
    query : Query
        The question with its gold evidence.
    result : Retrieved
        The arm's checked answer.

    Returns
    -------
    dict
        Hits, fractions, approvals, abstention, latency and leaks.
    """
    ids = {d.doc_id for d in result.docs}
    relevant = relevant_docs(query, result.docs)
    support = set(query.gold_support)
    contradict = set(query.gold_contradict)
    return {
        "support_hit": bool(ids & support),
        "support_fraction": len(ids & support) / len(support) if support else 0.0,
        "contradict_hit": bool(ids & contradict),
        "true_approved": value_check(query, result.docs, query.true_value),
        "false_approved": value_check(query, result.docs, query.false_value),
        "abstained": not relevant,
        "seconds": result.seconds,
        "fallback": result.fallback,
        "leaks": sum(result.leaks.values()),
    }


def _rate(rows: Sequence[dict[str, float | bool | int]], key: str) -> float:
    """Return the mean of ``key`` over ``rows`` as a percentage, 0 when empty."""
    return 100.0 * sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def summarise(
    queries: Sequence[Query], rows: Sequence[dict[str, float | bool | int]]
) -> dict[str, float]:
    """Aggregate one arm's per-query rows into the preregistered metrics.

    Parameters
    ----------
    queries : Sequence[Query]
        The queries, aligned with ``rows``.
    rows : Sequence[dict]
        The per-query measurements.

    Returns
    -------
    dict[str, float]
        Recall, approval, abstention, latency and leak figures.
    """
    answer = [
        r for q, r in zip(queries, rows, strict=True) if q.kind in {"answer", "swap"}
    ]
    contradicted = [
        r
        for q, r in zip(queries, rows, strict=True)
        if q.kind == "answer" and q.gold_contradict
    ]
    negative = [
        r
        for q, r in zip(queries, rows, strict=True)
        if q.kind in {"unknown_relation", "foreign_tenant"}
    ]
    seconds = sorted(float(r["seconds"]) for r in rows)
    return {
        "queries": float(len(rows)),
        "r_sup_at_k": _rate(answer, "support_hit"),
        "support_fraction": _rate(answer, "support_fraction"),
        "r_con_at_k": _rate(contradicted, "contradict_hit"),
        "false_approval": _rate(answer, "false_approved"),
        "true_approval": _rate(answer, "true_approved"),
        "negative_approval": _rate(negative, "true_approved"),
        "abstention": _rate(negative, "abstained"),
        "fallback": _rate(rows, "fallback"),
        "leaks": float(sum(int(r["leaks"]) for r in rows)),
        "p50_ms": 1000.0 * statistics.median(seconds) if seconds else 0.0,
        "p95_ms": 1000.0
        * seconds[min(len(seconds) - 1, math.ceil(0.95 * len(seconds)) - 1)]
        if seconds
        else 0.0,
    }


def paired_bootstrap(
    queries: Sequence[Query],
    left: Sequence[Row],
    right: Sequence[Row],
    key: str,
    *,
    groups: dict[str, str],
    reps: int = 2000,
    seed: int = 20261121,
) -> tuple[float, float]:
    """Return the 95 % interval of ``left − right`` for ``key``, resampling groups.

    The resampled unit is the split's unit: an entity together with its
    similar-name partner.

    Parameters
    ----------
    queries : Sequence[Query]
        The queries, aligned with both row lists.
    left, right : Sequence[Row]
        The two arms' per-query rows.
    key : str
        The measurement to compare, in percentage points.
    groups : dict[str, str]
        The group of each entity.
    reps : int
        Bootstrap replications.
    seed : int
        Seed of the resampling.

    Returns
    -------
    tuple[float, float]
        The 2.5 and 97.5 percentiles.
    """
    by_group: dict[str, list[int]] = {}
    for index, query in enumerate(queries):
        by_group.setdefault(groups[query.entity_id], []).append(index)
    names = sorted(by_group)
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(reps):
        picks = [i for _ in names for i in by_group[rng.choice(names)]]
        diff = sum(float(left[i][key]) - float(right[i][key]) for i in picks)
        diffs.append(100.0 * diff / len(picks))
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1]


def decide(
    summary: dict[str, dict[str, float]],
    interval: tuple[float, float],
    cost: dict[str, dict[str, float]],
    false_interval: tuple[float, float],
) -> dict[str, bool]:
    """Apply the preregistered acceptance rules to the pruned arm.

    Parameters
    ----------
    summary : dict
        Per-arm metrics from :func:`summarise`.
    interval : tuple[float, float]
        Bootstrap interval of the supporting-recall gain of ``pruned`` over ``hybrid``.
    cost : dict
        Per-arm ``ram_bytes`` and ``index_bytes``.
    false_interval : tuple[float, float]
        Bootstrap interval of the false-approval difference of ``pruned``
        over ``hybrid``.

    Returns
    -------
    dict[str, bool]
        Each rule's outcome and the overall ``accepted``. Abstention is
        reported but is not an acceptance rule.
    """
    c, a, b = summary["pruned"], summary["hybrid"], summary["keys"]
    rules = {
        "gain": c["r_sup_at_k"] - a["r_sup_at_k"] >= 5.0 and interval[0] > 0.0,
        "contradictions": c["r_con_at_k"] >= a["r_con_at_k"] - 1.0,
        "false_approvals": false_interval[1] <= 0.5,
        "boundaries": c["leaks"] == 0.0 and a["leaks"] == 0.0,
        "cost": c["p95_ms"] <= 1.5 * a["p95_ms"]
        and cost["pruned"]["ram_bytes"] <= 2 * cost["hybrid"]["ram_bytes"]
        and cost["pruned"]["index_bytes"] <= 2 * cost["hybrid"]["index_bytes"],
        "h2_faster": c["p95_ms"] < b["p95_ms"]
        and c["r_sup_at_k"] >= b["r_sup_at_k"] - 1.0
        and c["r_con_at_k"] >= b["r_con_at_k"] - 1.0,
    }
    rules["accepted"] = all(
        rules[name]
        for name in ("gain", "contradictions", "false_approvals", "boundaries", "cost")
    )
    return rules


def build_arm(
    snapshot: Snapshot, encoder: object, arm: str
) -> tuple[ArmIndex, dict[str, float]]:
    """Build one arm's index and measure its build time, peak memory and size.

    Memory is traced during the build only, so tracing never slows the
    timed queries.

    Parameters
    ----------
    snapshot : Snapshot
        The corpus state to index.
    encoder : object
        The shared encoder, with ``encode(text, normalize_embeddings=True)``.
    arm : str
        One of :data:`ARMS`.

    Returns
    -------
    tuple[ArmIndex, dict[str, float]]
        The index, and its build seconds, peak traced bytes and index bytes.
    """
    tracemalloc.start()
    started = time.perf_counter()
    index = ArmIndex(snapshot, encoder, arm)
    build_seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dimensions = int(
        np.asarray(
            cast(_Encoder, encoder).encode("x", normalize_embeddings=True)
        ).shape[0]
    )
    return index, {
        "build_seconds": build_seconds,
        "ram_bytes": float(peak),
        "index_bytes": float(index.index_bytes(dimensions)),
    }


def timed_rows(
    indices: dict[str, ArmIndex],
    queries: Sequence[Query],
    warmup: Sequence[Query],
    *,
    g: int,
    tau: float,
    runs: int = TIMED_RUNS,
    seed: int = 20261121,
) -> tuple[dict[str, list[Row]], dict[str, list[float]]]:
    """Answer every query with every arm, interleaved, and time each answer.

    Each arm first answers the warm-up queries untimed. Each timed run then
    visits every (query, arm) pair in a seeded random order, so no arm is
    always measured first or last. Rows come from the first run; the
    latencies of all runs are kept.

    Parameters
    ----------
    indices : dict[str, ArmIndex]
        The built index of each arm.
    queries : Sequence[Query]
        The timed questions.
    warmup : Sequence[Query]
        Untimed questions answered first by every arm.
    g, tau : int, float
        The pruned arm's parameters.
    runs : int
        Number of timed runs.
    seed : int
        Seed of the visiting order.

    Returns
    -------
    tuple[dict, dict]
        Per-arm rows aligned with ``queries``, and per-arm latencies.
    """
    for index in indices.values():
        for query in warmup:
            index.query(query, g=g, tau=tau)
    rows: dict[str, list[Row]] = {arm: [{} for _ in queries] for arm in indices}
    seconds: dict[str, list[float]] = {arm: [] for arm in indices}
    rng = random.Random(seed)
    pairs = [(i, arm) for i in range(len(queries)) for arm in indices]
    for run in range(runs):
        rng.shuffle(pairs)
        for i, arm in pairs:
            result = indices[arm].query(queries[i], g=g, tau=tau)
            seconds[arm].append(result.seconds)
            if run == 0:
                rows[arm][i] = score_query(queries[i], result)
    return rows, seconds


def run_arm(
    snapshot: Snapshot,
    encoder: object,
    arm: str,
    queries: Sequence[Query],
    *,
    g: int,
    tau: float,
) -> tuple[list[Row], dict[str, float]]:
    """Build one arm and answer every query once, untimed beyond the rows.

    Parameters
    ----------
    snapshot : Snapshot
        The corpus state to index.
    encoder : object
        The shared encoder.
    arm : str
        One of :data:`ARMS`.
    queries : Sequence[Query]
        The questions.
    g, tau : int, float
        The pruned arm's parameters.

    Returns
    -------
    tuple[list[Row], dict[str, float]]
        Per-query rows and the build cost.
    """
    index, cost = build_arm(snapshot, encoder, arm)
    return [
        score_query(query, index.query(query, g=g, tau=tau)) for query in queries
    ], cost


def key_overlap_share(queries: Sequence[Query], corpus: Corpus) -> float:
    """Return the share of questions sharing a word with their key's relation noun.

    The entity name is removed first, since every arm sees it.

    Parameters
    ----------
    queries : Sequence[Query]
        The questions.
    corpus : Corpus
        The corpus, for each entity's type and name.

    Returns
    -------
    float
        Percentage of questions with at least one shared word.
    """
    facts = {f.entity_id: f for f in corpus.snapshot.facts}
    hits = 0
    for query in queries:
        fact = facts[query.entity_id]
        words = set(
            query.text.lower()
            .replace(fact.entity_name.lower(), " ")
            .replace("?", " ")
            .split()
        )
        hits += bool(
            words & set(_noun(fact.entity_type, query.relation).lower().split())
        )
    return 100.0 * hits / len(queries) if queries else 0.0


def source_digests() -> dict[str, str]:
    """Return the SHA-256 of the generator, the harness and the template pools.

    Returns
    -------
    dict[str, str]
        Digests that the frozen preregistration names.
    """
    from benchmarks import atlaskv_corpus as corpus_module

    templates = {
        "relations": corpus_module.RELATIONS,
        "evidence": corpus_module.EVIDENCE_TEMPLATES,
        "questions": corpus_module.QUESTION_TEMPLATES,
        "claim": corpus_module.CLAIM_TEMPLATE,
    }
    return {
        "generator_sha256": hashlib.sha256(
            Path(corpus_module.__file__).read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "templates_sha256": hashlib.sha256(
            json.dumps(templates, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def _cached_revision(model_id: str) -> str:
    """Return the locally cached revision of a Hugging Face model, or ``""``."""
    ref = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
        / "refs"
        / "main"
    )
    return ref.read_text(encoding="utf-8").strip() if ref.is_file() else ""


def _sentence_transformer(
    name: str,
) -> object:  # pragma: no cover - needs the optional sentence-transformers package and a cached model; the lab-server run exercises it and records the revision
    """Load a sentence-transformers model from the local cache only, on CPU."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu", local_files_only=True)


def load_encoder(
    name: str, factory: Callable[[str], object] = _sentence_transformer
) -> tuple[object, str]:
    """Return the encoder and its revision; ``"hashing"`` needs no model.

    Parameters
    ----------
    name : str
        ``"hashing"`` or a Hugging Face model id already in the local cache.
    factory : callable
        Loader of a named model; the default never downloads.

    Returns
    -------
    tuple[object, str]
        The encoder and the cached revision (``"builtin"`` for hashing).
    """
    if name == "hashing":
        return HashingEmbedder(), "builtin"
    return factory(name), _cached_revision(name)


def _report_class(query: Query) -> str:
    """Return a query's false-value class, with the partner class split by direction.

    Whole-phrase relevance works one way: an item of ``X North`` names
    ``X``, but an item of ``X`` never names ``X North``. A partner value on
    the longer-named member can therefore never be approved, so the two
    directions are reported apart.
    """
    if query.false_value_class != "partner":
        return query.false_value_class
    return (
        "partner_on_longer_name"
        if query.entity_name.endswith(" North")
        else "partner_on_shorter_name"
    )


def _class_counts(queries: Sequence[Query]) -> dict[str, int]:
    """Return how many queries fall in each reported false-value class."""
    counts: dict[str, int] = {}
    for query in queries:
        name = _report_class(query)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _percentile_ms(seconds: Sequence[float], share: float) -> float:
    """Return a nearest-rank percentile of latencies in milliseconds."""
    ordered = sorted(seconds)
    if not ordered:
        return 0.0
    return 1000.0 * ordered[min(len(ordered) - 1, math.ceil(share * len(ordered)) - 1)]


def evaluate(
    corpus: Corpus, encoder: object, split: str, *, g: int, tau: float
) -> dict[str, object]:
    """Run all arms on one split and apply the acceptance rules.

    Parameters
    ----------
    corpus : Corpus
        The corpus.
    encoder : object
        The shared encoder.
    split : str
        ``"dev"`` or ``"test"``.
    g, tau : int, float
        The pruned arm's parameters.

    Returns
    -------
    dict[str, object]
        Per-arm metrics and costs, swap results, the intervals and the decision.
    """
    queries = [q for q in corpus.queries if q.split == split]
    warmup = [q for q in corpus.queries if q.split == "dev"][:WARMUP_QUERIES]
    swaps = [q for q in corpus.swap_queries if q.split == split]
    built = {arm: build_arm(corpus.snapshot, encoder, arm) for arm in ARMS}
    rows, seconds = timed_rows(
        {arm: built[arm][0] for arm in ARMS}, queries, warmup, g=g, tau=tau
    )
    summary: dict[str, dict[str, float]] = {}
    swap: dict[str, dict[str, float]] = {}
    swap_rows_by_arm: dict[str, list[Row]] = {}
    for arm in ARMS:
        summary[arm] = summarise(queries, rows[arm])
        summary[arm]["p50_ms"] = _percentile_ms(seconds[arm], 0.50)
        summary[arm]["p95_ms"] = _percentile_ms(seconds[arm], 0.95)
        swap_rows, swap_cost = run_arm(
            corpus.swapped, encoder, arm, swaps, g=g, tau=tau
        )
        swap_rows_by_arm[arm] = swap_rows
        swap[arm] = {
            **summarise(swaps, swap_rows),
            "rebuild_seconds": swap_cost["build_seconds"],
        }
        summary[arm]["leaks"] += swap[arm]["leaks"]
    answers = [i for i, q in enumerate(queries) if q.kind == "answer"]

    def interval(key: str) -> tuple[float, float]:
        """Return the paired group-bootstrap interval of pruned − hybrid."""
        return paired_bootstrap(
            [queries[i] for i in answers],
            [rows["pruned"][i] for i in answers],
            [rows["hybrid"][i] for i in answers],
            key,
            groups=corpus.entity_group,
        )

    gain, false = interval("support_hit"), interval("false_approved")
    cost = {arm: built[arm][1] for arm in ARMS}
    return {
        "split": split,
        "g": g,
        "tau": tau,
        "summary": summary,
        "cost": cost,
        "swap": swap,
        "gain_interval_pp": list(gain),
        "false_approval_interval_pp": list(false),
        "key_overlap_share": key_overlap_share(queries, corpus),
        "false_claims_by_class": _class_counts(
            [q for q in [*queries, *swaps] if q.kind in {"answer", "swap"}]
        ),
        "false_approvals_by_class": {
            arm: _class_counts(
                [
                    q
                    for q, row in [
                        *zip(queries, rows[arm], strict=True),
                        *zip(swaps, swap_rows_by_arm[arm], strict=True),
                    ]
                    if q.kind in {"answer", "swap"} and row["false_approved"]
                ]
            )
            for arm in ARMS
        },
        "support_discordance": 100.0
        * sum(
            rows["pruned"][i]["support_hit"] != rows["hybrid"][i]["support_hit"]
            for i in answers
        )
        / max(1, len(answers)),
        "timed_runs": TIMED_RUNS,
        "warmup_queries": len(warmup),
        "decision": decide(summary, gain, cost, false),
    }


def select_on_dev(corpus: Corpus, encoder: object) -> dict[str, object]:
    """Pick ``g`` and ``tau`` by supporting recall of the pruned arm on dev.

    Ties prefer the smaller ``g`` and then the larger ``tau``.

    Parameters
    ----------
    corpus : Corpus
        The corpus.
    encoder : object
        The shared encoder.

    Returns
    -------
    dict[str, object]
        The grid's recalls and the chosen parameters.
    """
    queries = [q for q in corpus.queries if q.split == "dev"]
    grid: list[dict[str, float]] = []
    for g in GRID_G:
        for tau in GRID_TAU:
            rows, _ = run_arm(corpus.snapshot, encoder, "pruned", queries, g=g, tau=tau)
            grid.append(
                {
                    "g": float(g),
                    "tau": tau,
                    "r_sup_at_k": summarise(queries, rows)["r_sup_at_k"],
                }
            )
    best = max(grid, key=lambda row: (row["r_sup_at_k"], -row["g"], row["tau"]))
    return {"grid": grid, "g": int(best["g"]), "tau": best["tau"]}


def _environment(cpuinfo: Path = Path("/proc/cpuinfo")) -> dict[str, str]:
    """Return the interpreter, numpy and CPU facts of this run."""
    import platform

    cpu = ""
    if cpuinfo.is_file():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            "",
        )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cpu": cpu,
        "machine": platform.machine(),
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[str], tuple[object, str]] = load_encoder,
) -> dict[str, object]:
    """Run the dev selection or the test comparison and write the artefact.

    Parameters
    ----------
    argv : Sequence[str], optional
        Command-line arguments.
    load : callable
        Encoder loader, replaceable in tests.

    Returns
    -------
    dict[str, object]
        The artefact written to ``--output``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--g", type=int, choices=GRID_G)
    parser.add_argument("--tau", type=float, choices=GRID_TAU)
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--entities", type=int, default=400)
    parser.add_argument("--similar-pairs", type=int, default=40)
    parser.add_argument("--preregistration-sha256", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.split == "test" and (args.g is None or args.tau is None):
        parser.error("--split test needs the --g and --tau chosen on dev")
    corpus = build_corpus(entities=args.entities, similar_pairs=args.similar_pairs)
    encoder, revision = load(args.encoder)
    started = time.time()
    if args.split == "dev":
        selection = select_on_dev(corpus, encoder)
        result = evaluate(
            corpus,
            encoder,
            "dev",
            g=int(cast(float, selection["g"])),
            tau=cast(float, selection["tau"]),
        )
        result["selection"] = selection
    else:
        result = evaluate(corpus, encoder, "test", g=args.g, tau=args.tau)
    artefact: dict[str, object] = {
        "schema": "director-ai.atlaskv-retrieval-comparison.v1",
        "preregistration_sha256": args.preregistration_sha256,
        "snapshot_id": corpus.snapshot.snapshot_id,
        "swapped_snapshot_id": corpus.swapped.snapshot_id,
        "encoder": args.encoder,
        "encoder_revision": revision,
        "top_k": TOP_K,
        "environment": _environment(),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "sources": source_digests(),
        "wall_seconds": time.time() - started,
        **result,
    }
    output = args.output or RESULTS_DIR / f"atlaskv_retrieval_{args.split}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artefact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return artefact


if __name__ == "__main__":
    main()
