# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real public-surface coverage for cost and attribution scoring."""

from __future__ import annotations

from typing import cast

import pytest

pytest.importorskip("fastapi", reason="fastapi required for server route tests")

from fastapi.testclient import TestClient

import director_ai
from director_ai.compliance.cost_analyser import CostAnalyser
from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer
from director_ai.core.types import ClaimAttribution
from director_ai.server import create_app
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _policy_ground_truth() -> GroundTruthStore:
    """Build the public keyword store used by the attribution scorer path."""
    store = GroundTruthStore()
    store.add(
        "refund-policy",
        (
            "Refund approvals require signed manager evidence before shipment. "
            "Invoice reminders are separate."
        ),
        tenant_id="tenant-alpha",
    )
    return store


def _cost_attribution_scorer(store: GroundTruthStore) -> CoherenceScorer:
    """Build the public scorer with dependency-free NLI and retrieval wiring."""
    return DirectorConfig(
        mode="general",
        scorer_backend="lite",
        use_nli=True,
        coherence_threshold=0.0,
        hard_limit=0.0,
        soft_limit=0.0,
        adaptive_threshold_enabled=False,
        w_logic=0.0,
        w_fact=1.0,
    ).build_scorer(store)


def _cost_attribution_client() -> TestClient:
    """Build a real FastAPI app with a dependency-free vector store."""
    config = DirectorConfig(
        mode="grounded",
        scorer_backend="lite",
        use_nli=True,
        coherence_threshold=0.0,
        hard_limit=0.0,
        soft_limit=0.0,
        adaptive_threshold_enabled=False,
        hybrid_retrieval=False,
        reranker_enabled=False,
        retrieval_abstention_threshold=0.0,
        w_logic=0.0,
        w_fact=1.0,
    )
    return TestClient(create_app(config))


def test_cost_and_attribution_unit_guard_declares_this_companion() -> None:
    """The legacy cost/attribution unit guard should declare this companion."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_cost_and_attribution.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_cost_and_attribution_real_surface.py" in reason


def test_public_review_attaches_claim_attribution_evidence() -> None:
    """Public review should emit claim-level attribution through real scoring."""
    scorer = _cost_attribution_scorer(_policy_ground_truth())

    approved, score = scorer.review(
        "Which controls refund approval?",
        (
            "Refund approvals require signed manager evidence before shipment. "
            "This approval rule applies before goods leave the warehouse. "
            "Invoice reminders are separate from refund approvals, and the "
            "billing process does not replace manager signoff."
        ),
        tenant_id="tenant-alpha",
    )

    evidence = score.evidence
    assert approved is True
    assert score.detected_task_type == "qa"
    assert evidence is not None
    assert evidence.claim_coverage == pytest.approx(1.0)
    assert evidence.claims == [
        "Refund approvals require signed manager evidence before shipment.",
        "This approval rule applies before goods leave the warehouse.",
        (
            "Invoice reminders are separate from refund approvals, and the "
            "billing process does not replace manager signoff."
        ),
    ]
    assert evidence.attributions is not None
    assert [item.source_index for item in evidence.attributions] == [0, 0, 1]
    assert all(isinstance(item, ClaimAttribution) for item in evidence.attributions)
    assert evidence.attributions[0].supported is True
    assert evidence.attributions[0].divergence == pytest.approx(0.0)


def test_public_server_evidence_payload_serialises_attributions() -> None:
    """The review route should expose public attribution evidence fields."""
    with _cost_attribution_client() as client:
        ingest_response = client.post(
            "/v1/knowledge/ingest",
            headers={"X-Tenant-ID": "tenant-alpha"},
            json={
                "doc_id": "refund-policy",
                "source": "refund-policy.md",
                "text": (
                    "Refund approvals require signed manager evidence before "
                    "shipment. Invoice reminders are separate."
                ),
                "chunk_size": 128,
                "overlap": 16,
            },
        )
        review_response = client.post(
            "/v1/review",
            headers={"X-Tenant-ID": "tenant-alpha"},
            json={
                "prompt": "Which controls refund approval?",
                "response": (
                    "Refund approvals require signed manager evidence before "
                    "shipment. Invoice reminders are separate from refund "
                    "approvals."
                ),
            },
        )

    assert ingest_response.status_code == 201, ingest_response.text
    assert review_response.status_code == 200, review_response.text
    payload = cast(dict[str, object], review_response.json())
    evidence = cast(dict[str, object], payload["evidence"])
    attributions = cast(list[dict[str, object]], evidence["attributions"])

    assert evidence["claim_coverage"] == pytest.approx(1.0)
    assert evidence["claims"] == [
        "Refund approvals require signed manager evidence before shipment.",
        "Invoice reminders are separate from refund approvals.",
    ]
    assert attributions[0] == {
        "claim": "Refund approvals require signed manager evidence before shipment.",
        "claim_index": 0,
        "source_sentence": (
            "Refund approvals require signed manager evidence before shipment."
        ),
        "source_index": 0,
        "divergence": pytest.approx(0.0),
        "supported": True,
    }
    assert attributions[1]["source_sentence"] == ("Invoice reminders are separate.")


def test_public_cost_analyser_reports_usage_without_private_state() -> None:
    """CostAnalyser should expose token/cost totals through its public report."""
    analyser = CostAnalyser(currency="CHF")
    analyser.add_pricing("local-judge", input_per_1k=0.012, output_per_1k=0.024)

    analyser.record("local-judge", input_tokens=800, output_tokens=100)
    analyser.record(
        "local-judge",
        input_tokens=200,
        output_tokens=50,
        agent_id="reviewer",
    )

    report = analyser.report()

    assert report["currency"] == "CHF"
    assert report["total_tokens"] == 1150
    assert report["total_cost"] == pytest.approx(0.0156)
    assert report["models"]["local-judge"]["estimated_cost"] == pytest.approx(0.012)
    assert report["models"]["local-judge::reviewer"]["estimated_cost"] == (
        pytest.approx(0.0036)
    )


def test_public_package_exports_claim_attribution_type() -> None:
    """The package root should expose the public attribution dataclass."""
    assert director_ai.ClaimAttribution is ClaimAttribution
