# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Model Lifecycle Proposer Lineage

"""Custody-derived proposer lineage and the sealed look budget.

The evaluator never trusts a proposer's clocks, context names or edge
completeness. Lineage is a union-find over the model artefacts in custody:

1. artefacts ingested under one submitting role form one class;
2. a declared edge whose endpoints are in custody may merge two classes,
   never split one, so omitting an edge never shrinks a lineage;
3. the artefacts of one look form one class;
4. after a ``released`` or ``candidate_failed`` look, every artefact
   ingested after the look's custody watermark whose class holds an
   artefact of the look's requester role joins the look's lineage.

A base checkpoint listed by the preregistration is left out of the union
when it was ingested before the preregistration and, at the
preregistration's custody sequence, was in no consuming look on the split
and in no class tainted on it. Edges from such a base merge nothing, so
independent proposers who fine-tune one clean base keep their own looks.

The lineage id is root-anchored: the SHA-256 of ``"{seq}:{digest}"`` for
the class member with the smallest ``(custody_ingest_seq, content_digest)``.
A look is refused when an earlier journal entry on the split shares a
member with the class, so a merge can never reopen a budget.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .custody_records import MODEL_CUSTODY_KINDS, CustodyIngestRecord
from .envelopes import LineageEdge
from .errors import (
    PreregistrationOrderError,
    UnverifiedCustodyArtefactError,
)
from .sealed_looks import (
    CONSUMING_OUTCOMES,
    SealedLookJournalEntry,
)


@dataclass(frozen=True, slots=True)
class LineageComponent:
    """The lineage class of a look's frozen artefacts.

    Attributes
    ----------
    derived_proposer_lineage_id : str
        Root-anchored lineage id.
    members : frozenset[str]
        Content digests in the class.
    exempt_base_digests : frozenset[str]
        Listed base checkpoints left out of the union.
    """

    derived_proposer_lineage_id: str
    members: frozenset[str]
    exempt_base_digests: frozenset[str]


class _UnionFind:
    """Union-find over content digests with path halving."""

    def __init__(self, nodes: Iterable[str]) -> None:
        self._parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        """Return the representative of ``node``'s class."""
        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]
        return node

    def union(self, left: str, right: str) -> bool:
        """Merge two classes and return whether they were separate."""
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return False
        self._parent[max(root_left, root_right)] = min(root_left, root_right)
        return True

    def members(self, node: str) -> frozenset[str]:
        """Return every node in ``node``'s class."""
        root = self.find(node)
        return frozenset(other for other in self._parent if self.find(other) == root)


def _base_classes(
    records: Mapping[str, CustodyIngestRecord], edges: Sequence[LineageEdge]
) -> _UnionFind:
    """Union role classes and the declared edges whose endpoints are in ``records``."""
    forest = _UnionFind(records)
    first_by_role: dict[str, str] = {}
    for digest in sorted(records):
        role = records[digest].submitting_role
        forest.union(first_by_role.setdefault(role, digest), digest)
    for edge in edges:
        if edge.parent_digest in records and edge.child_digest in records:
            forest.union(edge.parent_digest, edge.child_digest)
    return forest


def _role_member(records: Mapping[str, CustodyIngestRecord], role: str) -> str | None:
    """Return one artefact ingested under ``role``, or ``None``."""
    return next(
        (
            digest
            for digest in sorted(records)
            if records[digest].submitting_role == role
        ),
        None,
    )


def _touches(
    forest: _UnionFind,
    records: Mapping[str, CustodyIngestRecord],
    digest: str,
    entry: SealedLookJournalEntry,
) -> bool:
    """Return whether ``digest``'s class intersects a consuming look."""
    root = forest.find(digest)
    frozen = [item for item in entry.frozen_artefact_digests if item in records]
    if any(forest.find(item) == root for item in frozen):
        return True
    rep = _role_member(records, entry.requester_role)
    after = records[digest].custody_ingest_seq > entry.custody_seq_at_release
    return after and rep is not None and forest.find(rep) == root


def exempt_base_digests(
    listed_bases: Iterable[str],
    *,
    preregistration_seq: int,
    split_digest: str,
    records: Mapping[str, CustodyIngestRecord],
    edges: Sequence[LineageEdge],
    prior_journal: Sequence[SealedLookJournalEntry],
) -> frozenset[str]:
    """Return the listed bases that are left out of the lineage union.

    Parameters
    ----------
    listed_bases : Iterable[str]
        The preregistration's ``base_checkpoint_digests``.
    preregistration_seq : int
        The preregistration's custody ingest sequence.
    split_digest : str
        The requested split.
    records : Mapping[str, CustodyIngestRecord]
        Model artefact records by content digest.
    edges : Sequence[LineageEdge]
        The request's declared edges.
    prior_journal : Sequence[SealedLookJournalEntry]
        Every earlier journal entry.

    Returns
    -------
    frozenset[str]
        Clean bases: ingested before the preregistration and untouched by
        any consuming look on the split that finished before it.
    """
    listed = frozenset(digest for digest in listed_bases if digest in records)
    forest = _base_classes(
        records, [edge for edge in edges if edge.parent_digest not in listed]
    )
    earlier = [
        entry
        for entry in prior_journal
        if entry.status in CONSUMING_OUTCOMES
        and entry.split_digest == split_digest
        and entry.custody_seq_at_release < preregistration_seq
    ]
    return frozenset(
        base
        for base in listed
        if records[base].custody_ingest_seq < preregistration_seq
        and not any(_touches(forest, records, base, entry) for entry in earlier)
    )


def derive_proposer_lineage(
    frozen_artefact_digests: Sequence[str],
    *,
    exempt_bases: frozenset[str],
    records: Mapping[str, CustodyIngestRecord],
    edges: Sequence[LineageEdge],
    prior_journal: Sequence[SealedLookJournalEntry],
) -> LineageComponent:
    """Derive the root-anchored lineage of a look's frozen artefacts.

    Parameters
    ----------
    frozen_artefact_digests : Sequence[str]
        The look's artefacts; each must have a model custody record.
    exempt_bases : frozenset[str]
        Clean bases from :func:`exempt_base_digests`.
    records : Mapping[str, CustodyIngestRecord]
        Custody records by content digest; only model kinds take part.
    edges : Sequence[LineageEdge]
        The request's declared edges.
    prior_journal : Sequence[SealedLookJournalEntry]
        Every earlier journal entry.

    Returns
    -------
    LineageComponent
        The class, its root-anchored id and the exempt bases.

    Raises
    ------
    PreregistrationOrderError
        If every frozen artefact is an exempt base, so the look names no
        candidate.
    UnverifiedCustodyArtefactError
        If a frozen artefact has no model custody record.
    """
    models = {
        d: r
        for d, r in records.items()
        if r.kind in MODEL_CUSTODY_KINDS and d not in exempt_bases
    }
    candidates = [
        digest for digest in frozen_artefact_digests if digest not in exempt_bases
    ]
    if not candidates:
        raise PreregistrationOrderError(
            "a look needs a candidate that is not an exempt base"
        )
    if not all(digest in models for digest in candidates):
        raise UnverifiedCustodyArtefactError(
            "every frozen artefact needs a model custody record"
        )
    forest = _base_classes(models, edges)
    consuming = [entry for entry in prior_journal if entry.status in CONSUMING_OUTCOMES]
    for group in [candidates, *[entry.frozen_artefact_digests for entry in consuming]]:
        present = [digest for digest in group if digest in models]
        for digest in present[1:]:
            forest.union(present[0], digest)
    changed = True
    while changed:
        changed = False
        for entry in consuming:
            anchors = [
                digest for digest in entry.frozen_artefact_digests if digest in models
            ]
            rep = _role_member(models, entry.requester_role)
            if not anchors or rep is None:
                continue
            for digest, record in models.items():
                later = record.custody_ingest_seq > entry.custody_seq_at_release
                if later and forest.find(digest) == forest.find(rep):
                    changed |= forest.union(anchors[0], digest)
    members = forest.members(candidates[0])
    root_seq, root_digest = min(
        (models[digest].custody_ingest_seq, digest) for digest in members
    )
    lineage_id = hashlib.sha256(f"{root_seq}:{root_digest}".encode()).hexdigest()
    return LineageComponent(lineage_id, members, exempt_bases)
