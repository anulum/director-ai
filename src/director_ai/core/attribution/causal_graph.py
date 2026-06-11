# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Causal Attribution Graph

"""DAG attribution layer over existing scoring and halt evidence.

The graph is explanatory, not a learned causal model. It encodes the
operational causal path Director-AI actually used: retrieved evidence,
claim support or contradiction, score contribution, halt trace, and
counterfactual score deltas. Raw tenant text is stored separately from
labels and is excluded from ``to_dict()`` unless explicitly requested.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..types import (
    ClaimAttribution,
    CoherenceScore,
    CounterfactualFactChange,
    EvidenceChunk,
    HaltEvidence,
    ScoringEvidence,
)

AttributionNodeKind = Literal[
    "claim",
    "evidence",
    "score",
    "halt",
    "counterfactual",
    "trace",
]
AttributionRelation = Literal[
    "supports",
    "contradicts",
    "contributes_to",
    "triggered_halt",
    "counterfactual_delta",
]

SCHEMA_VERSION = "director.causal_attribution.v1"


def _finite_or_none(value: float | None, *, field_name: str) -> float | None:
    """Return a finite float or None for optional numeric fields."""
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _clamp_weight(value: float) -> float:
    """Return a finite edge weight clamped to the unit interval."""
    if not math.isfinite(value):
        raise ValueError("edge weight must be finite")
    return max(0.0, min(1.0, float(value)))


def _metadata_without_none(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return metadata with None-valued entries removed."""
    return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class AttributionNode:
    """Single vertex in an attribution DAG.

    ``label`` is safe for operator-facing summaries. Raw claim, source,
    or fact text belongs in ``text`` and is redacted by default during
    serialisation.
    """

    node_id: str
    kind: AttributionNodeKind
    label: str
    score: float | None = None
    text: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required node fields and optional score."""
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if not self.label:
            raise ValueError("label must be non-empty")
        _finite_or_none(self.score, field_name="node score")

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialise the node with raw text redacted by default."""
        return {
            "id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "score": self.score,
            "text": self.text if include_text else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AttributionEdge:
    """Directed causal influence edge between attribution nodes."""

    source: str
    target: str
    relation: AttributionRelation
    weight: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate edge endpoints and normalise edge weight."""
        if not self.source or not self.target:
            raise ValueError("edge endpoints must be non-empty")
        if self.source == self.target:
            raise ValueError("self-loop attribution edges are not allowed")
        object.__setattr__(self, "weight", _clamp_weight(self.weight))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the attribution edge."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CausalAttributionGraph:
    """Validated explanatory DAG for score and halt decisions."""

    nodes: Iterable[AttributionNode]
    edges: Iterable[AttributionEdge]
    root_id: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Freeze graph collections and validate DAG integrity."""
        nodes = tuple(self.nodes)
        edges = tuple(self.edges)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        if not self.root_id:
            raise ValueError("root_id must be non-empty")
        seen: set[str] = set()
        for node in nodes:
            if node.node_id in seen:
                raise ValueError(f"duplicate node {node.node_id!r}")
            seen.add(node.node_id)
        if self.root_id not in seen:
            raise ValueError(f"root_id {self.root_id!r} is not a graph node")
        for edge in edges:
            if edge.source not in seen or edge.target not in seen:
                raise ValueError(
                    f"unknown edge endpoint {edge.source!r}->{edge.target!r}"
                )
        self._validate_acyclic(nodes, edges)

    @staticmethod
    def _validate_acyclic(
        nodes: tuple[AttributionNode, ...],
        edges: tuple[AttributionEdge, ...],
    ) -> None:
        """Raise when edges introduce a cycle."""
        indegree = {node.node_id: 0 for node in nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for edge in edges:
            children[edge.source].append(edge.target)
            indegree[edge.target] += 1
        ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        visited = 0
        while ready:
            node_id = ready.popleft()
            visited += 1
            for child in children[node_id]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        if visited != len(nodes):
            raise ValueError("cycle detected in causal attribution graph")

    def node(self, node_id: str) -> AttributionNode:
        """Return one graph node by id."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

    def top_contributors(self, *, limit: int = 5) -> tuple[AttributionEdge, ...]:
        """Return strongest direct contributors to the graph root."""
        if limit < 1:
            raise ValueError("limit must be positive")
        root_edges = [edge for edge in self.edges if edge.target == self.root_id]
        return tuple(
            sorted(root_edges, key=lambda edge: edge.weight, reverse=True)[:limit]
        )

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialise the graph with raw text redacted by default."""
        return {
            "schema_version": self.schema_version,
            "root_id": self.root_id,
            "nodes": [node.to_dict(include_text=include_text) for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }


def build_causal_attribution_graph(
    evidence: CoherenceScore | ScoringEvidence | HaltEvidence,
) -> CausalAttributionGraph:
    """Build a causal attribution graph from Director-AI evidence objects."""

    if isinstance(evidence, CoherenceScore):
        return _build_score_graph(evidence)
    if isinstance(evidence, ScoringEvidence):
        score = CoherenceScore(
            score=evidence.nli_score,
            approved=False,
            h_logical=1.0 - evidence.nli_score,
            h_factual=0.0,
            evidence=evidence,
        )
        return _build_score_graph(score)
    if isinstance(evidence, HaltEvidence):
        return _build_halt_graph(evidence)
    raise TypeError(f"unsupported attribution evidence type: {type(evidence)!r}")


def _build_score_graph(score: CoherenceScore) -> CausalAttributionGraph:
    """Build an attribution graph for a coherence score."""
    nodes: list[AttributionNode] = []
    edges: list[AttributionEdge] = []
    evidence = score.evidence
    attributions = tuple(evidence.attributions or ()) if evidence is not None else ()
    if attributions:
        _append_claim_attribution_nodes(nodes, edges, attributions)
    elif evidence is not None:
        _append_chunk_nodes(nodes, edges, tuple(evidence.chunks))

    root_id = "score:coherence"
    nodes.append(
        AttributionNode(
            node_id=root_id,
            kind="score",
            label="Coherence score",
            score=score.score,
            metadata=_metadata_without_none(
                {
                    "approved": score.approved,
                    "h_logical": score.h_logical,
                    "h_factual": score.h_factual,
                    "claim_coverage": evidence.claim_coverage
                    if evidence is not None
                    else None,
                    "nli_score": evidence.nli_score if evidence is not None else None,
                }
            ),
        )
    )
    for node in nodes:
        if node.node_id != root_id and node.kind in {"claim", "evidence"}:
            edges.append(
                AttributionEdge(
                    source=node.node_id,
                    target=root_id,
                    relation="contributes_to",
                    weight=_node_contribution_weight(node),
                )
            )
    return CausalAttributionGraph(nodes=nodes, edges=edges, root_id=root_id)


def _append_claim_attribution_nodes(
    nodes: list[AttributionNode],
    edges: list[AttributionEdge],
    attributions: tuple[ClaimAttribution, ...],
) -> None:
    """Append claim/evidence nodes and support edges from attributions."""
    for attribution in attributions:
        evidence_id = f"evidence:{attribution.source_index}"
        claim_id = f"claim:{attribution.claim_index}"
        if not any(node.node_id == evidence_id for node in nodes):
            nodes.append(
                AttributionNode(
                    node_id=evidence_id,
                    kind="evidence",
                    label=f"Evidence sentence {attribution.source_index}",
                    text=attribution.source_sentence,
                    metadata={"source_index": attribution.source_index},
                )
            )
        nodes.append(
            AttributionNode(
                node_id=claim_id,
                kind="claim",
                label=f"Claim {attribution.claim_index}",
                text=attribution.claim,
                score=1.0 - attribution.divergence
                if attribution.supported
                else attribution.divergence,
                metadata={
                    "claim_index": attribution.claim_index,
                    "source_index": attribution.source_index,
                    "supported": attribution.supported,
                    "divergence": attribution.divergence,
                },
            )
        )
        edges.append(
            AttributionEdge(
                source=evidence_id,
                target=claim_id,
                relation="supports" if attribution.supported else "contradicts",
                weight=attribution.divergence
                if not attribution.supported
                else 1.0 - attribution.divergence,
            )
        )


def _append_chunk_nodes(
    nodes: list[AttributionNode],
    edges: list[AttributionEdge],
    chunks: tuple[EvidenceChunk, ...],
) -> None:
    """Append retrieval evidence chunk nodes."""
    for index, chunk in enumerate(chunks):
        nodes.append(
            AttributionNode(
                node_id=f"evidence:{index}",
                kind="evidence",
                label=f"Evidence chunk {index}",
                text=chunk.text,
                score=max(0.0, min(1.0, 1.0 - chunk.distance)),
                metadata={
                    "source": chunk.source,
                    "distance": chunk.distance,
                    "source_index": index,
                },
            )
        )


def _node_contribution_weight(node: AttributionNode) -> float:
    """Return the direct contribution weight for a graph node."""
    if node.score is not None:
        return _clamp_weight(node.score)
    divergence = node.metadata.get("divergence")
    if isinstance(divergence, int | float):
        return _clamp_weight(float(divergence))
    return 0.0


def _build_halt_graph(evidence: HaltEvidence) -> CausalAttributionGraph:
    """Build an attribution graph for a halt decision."""
    nodes: list[AttributionNode] = []
    edges: list[AttributionEdge] = []
    for index, chunk in enumerate(evidence.evidence_chunks):
        nodes.append(
            AttributionNode(
                node_id=f"evidence:{index}",
                kind="evidence",
                label=f"Halt evidence chunk {index}",
                text=chunk.text,
                score=max(0.0, min(1.0, 1.0 - chunk.distance)),
                metadata={
                    "source": chunk.source,
                    "distance": chunk.distance,
                    "source_index": index,
                },
            )
        )

    root_id = "halt:decision"
    diagnostic = evidence.counterfactual_diagnostic
    if diagnostic is not None and diagnostic.best_change is not None:
        change = diagnostic.best_change
        nodes.append(_counterfactual_node("counterfactual:best", change))
        edges.append(
            AttributionEdge(
                source="counterfactual:best",
                target=root_id,
                relation="counterfactual_delta",
                weight=change.required_score_delta,
                metadata={"prevented_halt": change.prevented_halt},
            )
        )

    trace = evidence.trace_attribution
    if trace is not None:
        nodes.append(
            AttributionNode(
                node_id="trace:halt",
                kind="trace",
                label="Halt trace",
                score=trace.causal_contribution,
                metadata=_metadata_without_none(
                    {
                        "fact_source": trace.fact_source,
                        "retrieval_path": trace.retrieval_path,
                        "scorer_path": trace.scorer_path,
                        "token_offset": trace.token_offset,
                        "threshold": trace.threshold,
                        "causal_contribution": trace.causal_contribution,
                    }
                ),
            )
        )
        edges.append(
            AttributionEdge(
                source="trace:halt",
                target=root_id,
                relation="triggered_halt",
                weight=trace.causal_contribution,
            )
        )

    nodes.append(
        AttributionNode(
            node_id=root_id,
            kind="halt",
            label="Halt decision",
            score=evidence.last_score,
            metadata={
                "reason": evidence.reason,
                "suggested_action": evidence.suggested_action,
                "nli_scores": evidence.nli_scores,
            },
        )
    )
    for node in nodes:
        if node.kind == "evidence":
            edges.append(
                AttributionEdge(
                    source=node.node_id,
                    target=root_id,
                    relation="contributes_to",
                    weight=float(node.metadata.get("distance", 0.0)),
                )
            )
    return CausalAttributionGraph(nodes=nodes, edges=edges, root_id=root_id)


def _counterfactual_node(
    node_id: str, change: CounterfactualFactChange
) -> AttributionNode:
    """Build a counterfactual change node."""
    return AttributionNode(
        node_id=node_id,
        kind="counterfactual",
        label="Best counterfactual fact change",
        text=f"{change.original_fact}\n{change.proposed_fact}",
        score=change.required_score_delta,
        metadata={
            "fact_source": change.fact_source,
            "required_score_delta": change.required_score_delta,
            "prevented_halt": change.prevented_halt,
        },
    )
