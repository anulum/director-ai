# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Causal Attribution Graph Tests

import pytest

from director_ai.core import (
    ClaimAttribution,
    CoherenceScore,
    CounterfactualFactChange,
    CounterfactualHaltDiagnostic,
    EvidenceChunk,
    HaltEvidence,
    HaltTraceAttribution,
    ScoringEvidence,
)
from director_ai.core.attribution import (
    AttributionEdge,
    AttributionNode,
    CausalAttributionGraph,
    build_causal_attribution_graph,
)


def _score_with_attributions() -> CoherenceScore:
    evidence = ScoringEvidence(
        chunks=[
            EvidenceChunk(
                text="The medicine is contraindicated for severe renal impairment.",
                distance=0.1,
                source="kb://drug/renal",
            )
        ],
        nli_premise="renal safety facts",
        nli_hypothesis="candidate answer",
        nli_score=0.71,
        claim_coverage=0.5,
        per_claim_divergences=[0.05, 0.72],
        claims=[
            "Use normal dosing in severe renal impairment.",
            "Monitor renal function.",
        ],
        attributions=[
            ClaimAttribution(
                claim="Use normal dosing in severe renal impairment.",
                claim_index=0,
                source_sentence=(
                    "The medicine is contraindicated for severe renal impairment."
                ),
                source_index=0,
                divergence=0.72,
                supported=False,
            ),
            ClaimAttribution(
                claim="Monitor renal function.",
                claim_index=1,
                source_sentence="Monitor renal function during treatment.",
                source_index=1,
                divergence=0.05,
                supported=True,
            ),
        ],
    )
    return CoherenceScore(
        score=0.44,
        approved=False,
        h_logical=0.56,
        h_factual=0.72,
        evidence=evidence,
    )


def _halt_evidence() -> HaltEvidence:
    return HaltEvidence(
        reason="hard_limit (0.31 < 0.6)",
        last_score=0.31,
        evidence_chunks=[
            EvidenceChunk(
                text="The source says the bridge load limit is 10 tons.",
                distance=0.2,
                source="kb://bridge/spec",
            )
        ],
        suggested_action="Regenerate with grounded load-limit facts.",
        trace_attribution=HaltTraceAttribution(
            fact_source="kb://bridge/spec",
            retrieval_path="vector.kb",
            scorer_path="CoherenceScorer.review",
            token_offset=18,
            threshold=0.6,
            causal_contribution=0.29,
        ),
        counterfactual_diagnostic=CounterfactualHaltDiagnostic(
            question="what single fact change would have prevented this halt?",
            observed_score=0.31,
            threshold=0.6,
            best_change=CounterfactualFactChange(
                fact_source="kb://bridge/spec",
                original_fact="The source says the bridge load limit is 10 tons.",
                proposed_fact="The bridge load limit is 20 tons.",
                required_score_delta=0.29,
                prevented_halt=True,
            ),
            candidates=[
                CounterfactualFactChange(
                    fact_source="kb://bridge/spec",
                    original_fact="The source says the bridge load limit is 10 tons.",
                    proposed_fact="The bridge load limit is 20 tons.",
                    required_score_delta=0.29,
                    prevented_halt=True,
                )
            ],
        ),
    )


def test_builds_claim_evidence_score_dag_without_raw_text_by_default():
    graph = build_causal_attribution_graph(_score_with_attributions())

    assert graph.root_id == "score:coherence"
    assert graph.schema_version == "director.causal_attribution.v1"
    assert graph.node("claim:0").kind == "claim"
    assert graph.node("evidence:0").kind == "evidence"
    assert graph.node("score:coherence").score == pytest.approx(0.44)
    assert any(
        edge.source == "evidence:0"
        and edge.target == "claim:0"
        and edge.relation == "contradicts"
        and edge.weight == pytest.approx(0.72)
        for edge in graph.edges
    )

    payload = graph.to_dict()
    assert "Use normal dosing" not in str(payload)
    assert "contraindicated" not in str(payload)
    assert payload["nodes"][0]["text"] is None

    payload_with_text = graph.to_dict(include_text=True)
    assert "Use normal dosing" in str(payload_with_text)
    assert "contraindicated" in str(payload_with_text)


def test_builds_chunk_only_score_dag_from_scoring_evidence():
    evidence = ScoringEvidence(
        chunks=[
            EvidenceChunk(text="Paris is in France.", distance=0.25, source="kb://geo"),
            EvidenceChunk(
                text="Berlin is in Germany.", distance=0.5, source="kb://geo"
            ),
        ],
        nli_premise="geography facts",
        nli_hypothesis="candidate answer",
        nli_score=0.8,
        claim_coverage=None,
    )

    graph = build_causal_attribution_graph(evidence)

    assert graph.root_id == "score:coherence"
    assert graph.node("score:coherence").score == pytest.approx(0.8)
    assert "claim_coverage" not in graph.node("score:coherence").metadata
    assert graph.node("evidence:0").score == pytest.approx(0.75)
    assert graph.node("evidence:1").score == pytest.approx(0.5)
    assert [edge.source for edge in graph.top_contributors(limit=2)] == [
        "evidence:0",
        "evidence:1",
    ]


def test_builds_score_graph_without_evidence_metadata_none_values_removed():
    graph = build_causal_attribution_graph(
        CoherenceScore(
            score=0.91,
            approved=True,
            h_logical=0.05,
            h_factual=0.04,
            evidence=None,
        )
    )

    root = graph.node("score:coherence")
    assert root.metadata == {
        "approved": True,
        "h_logical": 0.05,
        "h_factual": 0.04,
    }
    assert graph.edges == ()


def test_builds_halt_trace_and_counterfactual_nodes():
    graph = build_causal_attribution_graph(_halt_evidence())

    assert graph.root_id == "halt:decision"
    assert graph.node("halt:decision").metadata["reason"] == "hard_limit (0.31 < 0.6)"
    assert graph.node("trace:halt").metadata["token_offset"] == 18
    assert graph.node("counterfactual:best").score == pytest.approx(0.29)
    assert any(
        edge.source == "trace:halt"
        and edge.target == "halt:decision"
        and edge.relation == "triggered_halt"
        and edge.weight == pytest.approx(0.29)
        for edge in graph.edges
    )

    top = graph.top_contributors(limit=2)
    assert top[0].source == "counterfactual:best"
    assert top[0].weight == pytest.approx(0.29)


def test_builds_minimal_halt_graph_without_optional_trace_or_counterfactual():
    evidence = HaltEvidence(
        reason="manual_review",
        last_score=0.4,
        evidence_chunks=[
            EvidenceChunk(text="A cited fact.", distance=0.3, source="kb://fact")
        ],
        suggested_action="Review cited fact.",
        trace_attribution=None,
        counterfactual_diagnostic=None,
    )

    graph = build_causal_attribution_graph(evidence)

    assert graph.node("halt:decision").metadata["reason"] == "manual_review"
    assert graph.node("evidence:0").metadata["distance"] == 0.3
    assert len(graph.edges) == 1
    assert graph.edges[0].relation == "contributes_to"
    assert graph.edges[0].weight == pytest.approx(0.3)


def test_graph_rejects_missing_edge_reference_and_cycles():
    nodes = [
        AttributionNode(node_id="a", kind="claim", label="A"),
        AttributionNode(node_id="b", kind="score", label="B"),
    ]

    with pytest.raises(ValueError, match="unknown edge endpoint"):
        CausalAttributionGraph(
            nodes=nodes,
            edges=[AttributionEdge("missing", "b", "contributes_to", 1.0)],
            root_id="b",
        )

    with pytest.raises(ValueError, match="cycle"):
        CausalAttributionGraph(
            nodes=nodes,
            edges=[
                AttributionEdge("a", "b", "contributes_to", 0.5),
                AttributionEdge("b", "a", "contributes_to", 0.5),
            ],
            root_id="b",
        )


def test_graph_rejects_invalid_weights_and_duplicate_nodes():
    with pytest.raises(ValueError, match="duplicate node"):
        CausalAttributionGraph(
            nodes=[
                AttributionNode(node_id="x", kind="claim", label="X"),
                AttributionNode(node_id="x", kind="evidence", label="X2"),
            ],
            edges=[],
            root_id="x",
        )

    with pytest.raises(ValueError, match="finite"):
        AttributionEdge("a", "b", "contributes_to", float("nan"))


def test_graph_rejects_invalid_nodes_root_and_self_loops():
    with pytest.raises(ValueError, match="node_id"):
        AttributionNode(node_id="", kind="claim", label="claim")
    with pytest.raises(ValueError, match="label"):
        AttributionNode(node_id="claim:1", kind="claim", label="")
    with pytest.raises(ValueError, match="node score"):
        AttributionNode(
            node_id="claim:1", kind="claim", label="claim", score=float("inf")
        )
    with pytest.raises(ValueError, match="self-loop"):
        AttributionEdge("a", "a", "contributes_to", 0.5)
    with pytest.raises(ValueError, match="root_id"):
        CausalAttributionGraph(
            nodes=[AttributionNode(node_id="a", kind="claim", label="A")],
            edges=[],
            root_id="missing",
        )


def test_graph_lookup_limit_and_unsupported_evidence_errors():
    graph = CausalAttributionGraph(
        nodes=[
            AttributionNode(node_id="a", kind="claim", label="A", score=0.1),
            AttributionNode(node_id="root", kind="score", label="Root"),
        ],
        edges=[AttributionEdge("a", "root", "contributes_to", 2.0)],
        root_id="root",
    )

    assert graph.edges[0].weight == 1.0
    with pytest.raises(KeyError):
        graph.node("missing")
    with pytest.raises(ValueError, match="limit"):
        graph.top_contributors(limit=0)
    with pytest.raises(TypeError, match="unsupported attribution evidence"):
        build_causal_attribution_graph(object())
