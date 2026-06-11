# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — server helper tests

from __future__ import annotations

from types import SimpleNamespace

from director_ai._server_helpers import evidence_to_dict, halt_evidence_to_dict


def _chunk(text: str, distance: float, source: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, distance=distance, source=source)


def _fact_change(
    *,
    fact_source: str = "policy.md",
    original_fact: str = "robots may continue",
    proposed_fact: str = "robots must halt",
    required_score_delta: float = 0.17,
    prevented_halt: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        fact_source=fact_source,
        original_fact=original_fact,
        proposed_fact=proposed_fact,
        required_score_delta=required_score_delta,
        prevented_halt=prevented_halt,
    )


def test_halt_evidence_none_returns_none() -> None:
    assert halt_evidence_to_dict(None) is None


def test_halt_evidence_serializes_optional_diagnostics() -> None:
    halt_evidence = SimpleNamespace(
        reason="score_below_threshold",
        last_score=0.41,
        evidence_chunks=[_chunk("source sentence", 0.08, "kb://fact/1")],
        nli_scores={"entailment": 0.22, "contradiction": 0.61},
        suggested_action="halt",
        trace_attribution=SimpleNamespace(
            fact_source="safety.md",
            retrieval_path=["query", "rerank", "chunk"],
            scorer_path=["lite", "nli"],
            token_offset=42,
            threshold=0.75,
            causal_contribution=0.33,
        ),
        counterfactual_diagnostic=SimpleNamespace(
            question="What fact would prevent the halt?",
            observed_score=0.41,
            threshold=0.75,
            best_change=_fact_change(),
            candidates=[
                _fact_change(
                    fact_source="ops.md",
                    original_fact="continue",
                    proposed_fact="stop",
                    required_score_delta=0.21,
                    prevented_halt=False,
                )
            ],
        ),
    )

    result = halt_evidence_to_dict(halt_evidence)

    assert result == {
        "reason": "score_below_threshold",
        "last_score": 0.41,
        "evidence_chunks": [
            {"text": "source sentence", "distance": 0.08, "source": "kb://fact/1"}
        ],
        "nli_scores": {"entailment": 0.22, "contradiction": 0.61},
        "suggested_action": "halt",
        "trace_attribution": {
            "fact_source": "safety.md",
            "retrieval_path": ["query", "rerank", "chunk"],
            "scorer_path": ["lite", "nli"],
            "token_offset": 42,
            "threshold": 0.75,
            "causal_contribution": 0.33,
        },
        "counterfactual_diagnostic": {
            "question": "What fact would prevent the halt?",
            "observed_score": 0.41,
            "threshold": 0.75,
            "best_change": {
                "fact_source": "policy.md",
                "original_fact": "robots may continue",
                "proposed_fact": "robots must halt",
                "required_score_delta": 0.17,
                "prevented_halt": True,
            },
            "candidates": [
                {
                    "fact_source": "ops.md",
                    "original_fact": "continue",
                    "proposed_fact": "stop",
                    "required_score_delta": 0.21,
                    "prevented_halt": False,
                }
            ],
        },
    }


def test_halt_evidence_keeps_absent_optional_diagnostics_null() -> None:
    halt_evidence = SimpleNamespace(
        reason="manual_review",
        last_score=0.5,
        evidence_chunks=[],
        nli_scores={},
        suggested_action="review",
    )

    result = halt_evidence_to_dict(halt_evidence)

    assert result["trace_attribution"] is None
    assert result["counterfactual_diagnostic"] is None


def test_halt_evidence_serializes_missing_best_counterfactual_as_null() -> None:
    halt_evidence = SimpleNamespace(
        reason="counterfactual_probe",
        last_score=0.49,
        evidence_chunks=[],
        nli_scores={},
        suggested_action="halt",
        counterfactual_diagnostic=SimpleNamespace(
            question="Which change matters?",
            observed_score=0.49,
            threshold=0.7,
            best_change=None,
            candidates=[],
        ),
    )

    result = halt_evidence_to_dict(halt_evidence)

    assert result["counterfactual_diagnostic"]["best_change"] is None
    assert result["counterfactual_diagnostic"]["candidates"] == []


def test_scoring_evidence_none_returns_none() -> None:
    assert evidence_to_dict(None) is None


def test_scoring_evidence_serializes_required_fields_only() -> None:
    evidence = SimpleNamespace(
        chunks=[_chunk("premise", 0.12, "doc://a")],
        nli_premise="premise text",
        nli_hypothesis="hypothesis text",
        nli_score=0.64,
        premise_chunk_count=2,
        hypothesis_chunk_count=1,
        claim_coverage=None,
        attributions=None,
        token_count=None,
    )

    assert evidence_to_dict(evidence) == {
        "chunks": [{"text": "premise", "distance": 0.12, "source": "doc://a"}],
        "nli_premise": "premise text",
        "nli_hypothesis": "hypothesis text",
        "nli_score": 0.64,
        "premise_chunk_count": 2,
        "hypothesis_chunk_count": 1,
    }


def test_scoring_evidence_serializes_claims_attributions_and_cost() -> None:
    evidence = SimpleNamespace(
        chunks=[],
        nli_premise="",
        nli_hypothesis="",
        nli_score=0.91,
        premise_chunk_count=0,
        hypothesis_chunk_count=0,
        claim_coverage=0.8,
        per_claim_divergences=[0.1, 0.4],
        claims=["claim one", "claim two"],
        attributions=[
            SimpleNamespace(
                claim="claim one",
                claim_index=0,
                source_sentence="supporting sentence",
                source_index=3,
                divergence=0.1,
                supported=True,
            )
        ],
        token_count=128,
        estimated_cost_usd=0.00032,
    )

    result = evidence_to_dict(evidence)

    assert result["claim_coverage"] == 0.8
    assert result["per_claim_divergences"] == [0.1, 0.4]
    assert result["claims"] == ["claim one", "claim two"]
    assert result["attributions"] == [
        {
            "claim": "claim one",
            "claim_index": 0,
            "source_sentence": "supporting sentence",
            "source_index": 3,
            "divergence": 0.1,
            "supported": True,
        }
    ]
    assert result["token_count"] == 128
    assert result["estimated_cost_usd"] == 0.00032
