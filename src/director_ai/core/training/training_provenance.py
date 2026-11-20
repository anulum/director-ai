# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Row-Level Training Provenance

"""Row-level training provenance for cluster-disjoint evaluation.

A fine-tuned model's anti-regression benchmark is only meaningful when its
evaluation rows are independent of the rows it was trained on. This module
gives every NLI row two identities:

* a content key, the SHA-256 of the case-folded, whitespace-normalised
  premise and hypothesis, which are the text the model sees; and
* an optional source cluster, read from the row's ``source_cluster`` field,
  which groups rows that come from one conversation, document or incident.

:class:`TrainingProvenance` records both identities for every training row,
including replay rows, and is written next to the saved model. Cluster
identifiers are stored only as SHA-256 digests. Evaluation code refuses rows
that share a content key or a source cluster with it.

The provenance file is training metadata. Its digests confirm whether a
guessed text or cluster identifier was part of training, so handle it like
the training data when a model directory is exported or shared.

:func:`split_rows_by_cluster` makes train and evaluation splits that keep
every cluster, and every repeated content, whole on one side.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

SOURCE_CLUSTER_FIELD = "source_cluster"
"""Row field naming the real source cluster an NLI row came from."""

PROVENANCE_FILENAME = "director_training_provenance.json"
"""File written next to a fine-tuned model to record its training rows."""

PROVENANCE_SCHEMA = "director-ai.training-provenance.v1"

_DIGEST = re.compile(r"[0-9a-f]{64}")

RowT = TypeVar("RowT", bound=Mapping[str, object])


def row_cluster(row: Mapping[str, object]) -> str | None:
    """Return a row's recorded source cluster, or ``None`` when absent.

    Parameters
    ----------
    row : Mapping[str, object]
        One decoded JSONL row.

    Returns
    -------
    str or None
        The cluster identifier, or ``None`` when the row records none.

    Raises
    ------
    ValueError
        If the field is present but empty, padded or not a string.
    """
    value = row.get(SOURCE_CLUSTER_FIELD)
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{SOURCE_CLUSTER_FIELD} must be a non-empty string without "
            f"surrounding whitespace; got {value!r}"
        )
    return value


def content_key(premise: str, hypothesis: str) -> str:
    """Return the SHA-256 of the normalised premise and hypothesis.

    Parameters
    ----------
    premise : str
        Row premise.
    hypothesis : str
        Row hypothesis.

    Returns
    -------
    str
        Hexadecimal digest that ignores case and whitespace differences.
    """
    normalised = "\x1f".join(
        " ".join(part.casefold().split()) for part in (premise, hypothesis)
    )
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _row_text(row: Mapping[str, object], key: str) -> str:
    """Return a required text field of a loaded NLI row."""
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"loaded NLI row lacks a text {key!r}")
    return value


def _row_key(row: Mapping[str, object]) -> str:
    """Return the content key of a loaded NLI row."""
    return content_key(_row_text(row, "premise"), _row_text(row, "hypothesis"))


def cluster_digest(cluster: str) -> str:
    """Return the SHA-256 digest under which a source cluster is recorded.

    Parameters
    ----------
    cluster : str
        Source cluster identifier.

    Returns
    -------
    str
        Hexadecimal digest of the domain-separated identifier.
    """
    return hashlib.sha256(f"cluster\x1f{cluster}".encode()).hexdigest()


def _row_cluster_digest(row: Mapping[str, object]) -> str | None:
    """Return the digest of a row's source cluster, or ``None``."""
    cluster = row_cluster(row)
    return None if cluster is None else cluster_digest(cluster)


@dataclass(frozen=True)
class TrainingProvenance:
    """Content keys and source clusters of every row a model trained on.

    Attributes
    ----------
    content_keys : frozenset[str]
        Normalised-content digests of the training and replay rows.
    cluster_digests : frozenset[str]
        Digests (:func:`cluster_digest`) of the source clusters recorded on
        those rows.
    """

    content_keys: frozenset[str]
    cluster_digests: frozenset[str]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, object]]) -> TrainingProvenance:
        """Record the identities of loaded training rows.

        Parameters
        ----------
        rows : Iterable[Mapping[str, object]]
            Loaded NLI rows with ``premise`` and ``hypothesis`` text.

        Returns
        -------
        TrainingProvenance
            Provenance covering every supplied row.
        """
        keys: set[str] = set()
        clusters: set[str] = set()
        for row in rows:
            keys.add(_row_key(row))
            digest = _row_cluster_digest(row)
            if digest is not None:
                clusters.add(digest)
        return cls(content_keys=frozenset(keys), cluster_digests=frozenset(clusters))

    def overlapping_rows(self, rows: Iterable[Mapping[str, object]]) -> int:
        """Count rows that share content or a source cluster with training.

        Parameters
        ----------
        rows : Iterable[Mapping[str, object]]
            Loaded evaluation rows.

        Returns
        -------
        int
            Number of rows that are not independent of the training rows.
        """
        return sum(
            1
            for row in rows
            if _row_key(row) in self.content_keys
            or _row_cluster_digest(row) in self.cluster_digests
        )

    def write(self, directory: str | Path) -> Path:
        """Write this provenance as JSON into ``directory``.

        Parameters
        ----------
        directory : str or Path
            Saved model directory, created when it does not exist yet.

        Returns
        -------
        Path
            Path of the written :data:`PROVENANCE_FILENAME`.
        """
        path = Path(directory) / PROVENANCE_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": PROVENANCE_SCHEMA,
            "content_keys": sorted(self.content_keys),
            "cluster_digests": sorted(self.cluster_digests),
        }
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> TrainingProvenance:
        """Read and validate a provenance file.

        Parameters
        ----------
        path : str or Path
            A file written by :meth:`write`.

        Returns
        -------
        TrainingProvenance
            The recorded identities.

        Raises
        ------
        ValueError
            If the file is not a well-formed provenance record.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != PROVENANCE_SCHEMA:
            raise ValueError("training provenance has an unknown schema")
        keys, clusters = payload.get("content_keys"), payload.get("cluster_digests")
        if not isinstance(keys, list) or not all(
            isinstance(key, str) and _DIGEST.fullmatch(key) for key in keys
        ):
            raise ValueError("training provenance content keys are malformed")
        if not keys:
            raise ValueError("training provenance records no training rows")
        if not isinstance(clusters, list) or not all(
            isinstance(digest, str) and _DIGEST.fullmatch(digest) for digest in clusters
        ):
            raise ValueError("training provenance cluster digests are malformed")
        return cls(content_keys=frozenset(keys), cluster_digests=frozenset(clusters))


def split_rows_by_cluster(
    rows: Sequence[RowT],
    *,
    eval_fraction: float,
    seed: int,
) -> tuple[list[RowT], list[RowT]]:
    """Split rows into train and evaluation sets with no shared group.

    Rows join one group when they share a source cluster or a content key.
    Groups are shuffled with ``seed`` and moved to evaluation until it holds
    at least ``eval_fraction`` of the rows. Both sides keep their input order.

    Parameters
    ----------
    rows : Sequence[Mapping[str, object]]
        Loaded NLI rows.
    eval_fraction : float
        Target evaluation share, in ``(0, 1)``.
    seed : int
        Seed for the group order.

    Returns
    -------
    tuple[list, list]
        Training rows and evaluation rows.

    Raises
    ------
    ValueError
        If the fraction is outside ``(0, 1)`` or the rows form fewer than two
        independent groups.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in (0, 1); got {eval_fraction!r}")
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        """Return the group root of ``node`` and compress its path."""
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    nodes: list[str] = []
    for index, row in enumerate(rows):
        cluster = row_cluster(row)
        node = f"row:{index}" if cluster is None else f"cluster:{cluster}"
        nodes.append(node)
        left, right = find(node), find(f"content:{_row_key(row)}")
        if left != right:
            low, high = sorted((left, right))
            parent[high] = low

    groups: dict[str, list[int]] = {}
    for index, node in enumerate(nodes):
        groups.setdefault(find(node), []).append(index)
    if len(groups) < 2:
        raise ValueError(
            "rows form fewer than two independent groups; add rows from "
            "other source clusters before splitting"
        )
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    target = math.ceil(eval_fraction * len(rows))
    held_out: set[int] = set()
    for root in order[:-1]:
        if len(held_out) >= target:
            break
        held_out.update(groups[root])
    train = [row for index, row in enumerate(rows) if index not in held_out]
    evaluation = [row for index, row in enumerate(rows) if index in held_out]
    return train, evaluation
