# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — DP-RAG real-surface tests
"""Real public-surface coverage for differentially private RAG budgets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.dp_rag import (
    DifferentiallyPrivateRetrieval,
    DPBudgetExceededError,
    DPRagPipeline,
    PrivateRanking,
    ScoredItem,
)
from director_ai.guard import ProductionGuard
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def _items() -> list[ScoredItem]:
    """Return deterministic retrieval candidates for public DP ranking."""
    return [
        ScoredItem("doc-a", 0.91),
        ScoredItem("doc-b", 0.54),
        ScoredItem("doc-c", 0.13),
    ]


def test_dp_rag_unit_guard_has_real_surface_companion() -> None:
    """The DP-RAG helper guard needs public retrieval and pipeline coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_dp_rag.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_dp_rag_real_surface.py" in category


def test_public_guard_dp_retrieval_tracks_tenant_budget() -> None:
    """ProductionGuard should expose a persistent public DP retrieval surface."""
    guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
    retrieval = guard.dp_retrieval

    ranking = retrieval.rank(_items(), tenant_id="tenant-a", epsilon=0.5)
    payload = ranking.to_dict()

    assert isinstance(retrieval, DifferentiallyPrivateRetrieval)
    assert guard.dp_retrieval is retrieval
    assert isinstance(ranking, PrivateRanking)
    assert ranking.tenant_id == "tenant-a"
    assert ranking.epsilon_spent == pytest.approx(0.5)
    assert retrieval.spent("tenant-a") == pytest.approx(0.5)
    assert retrieval.remaining("tenant-a") == pytest.approx(9.5)
    assert set(payload) == {
        "tenant_id",
        "epsilon_spent",
        "epsilon_remaining",
        "items",
    }
    assert payload["tenant_id"] == "tenant-a"
    items_payload = payload["items"]
    assert isinstance(items_payload, list)
    typed_items = cast(list[Mapping[str, object]], items_payload)
    assert all(set(item) == {"item_id", "score"} for item in typed_items)


def test_public_pipeline_shares_budget_and_refuses_overrun() -> None:
    """The public DP-RAG pipeline should charge all stages to one budget."""
    guard = ProductionGuard(DirectorConfig(use_nli=False, llm_provider="mock"))
    pipeline = guard.dp_rag_pipeline(max_epsilon=1.0, seed=7)

    ranking = pipeline.rank(_items(), tenant_id="tenant-a", epsilon=0.3)
    choice = pipeline.decode([0.2, 0.7, 0.1], tenant_id="tenant-a", epsilon=0.2)
    released = pipeline.release_score(0.6, tenant_id="tenant-a", epsilon=0.2)
    before_spent = pipeline.spent("tenant-a")
    before_log = pipeline.stage_log("tenant-a")

    with pytest.raises(DPBudgetExceededError, match="decode epsilon"):
        pipeline.decode([0.2, 0.7], tenant_id="tenant-a", epsilon=0.4)

    assert isinstance(pipeline, DPRagPipeline)
    assert ranking.tenant_id == "tenant-a"
    assert 0 <= choice.index < 3
    assert isinstance(released, float)
    assert before_spent == pytest.approx(0.7)
    assert pipeline.remaining("tenant-a") == pytest.approx(0.3)
    assert [charge.stage for charge in before_log] == [
        "retrieve",
        "decode",
        "release",
    ]
    assert [charge.epsilon for charge in before_log] == [0.3, 0.2, 0.2]
    assert pipeline.spent("tenant-a") == pytest.approx(before_spent)
    assert pipeline.stage_log("tenant-a") == before_log
