# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real public-surface coverage for task-specific scorer routing."""

from __future__ import annotations

import pytest

import director_ai.core.scoring._task_accel as task_accel
import director_ai.core.scoring.lite_scorer as lite_scorer
from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.scoring.scorer import CoherenceScorer
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


@pytest.fixture
def deterministic_python_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run optional accelerator paths through deterministic Python fallbacks."""
    monkeypatch.setattr(task_accel, "_RUST_TASK", False)
    monkeypatch.setattr(lite_scorer, "_RUST_LITE", False)


def _build_lite_scorer(
    *,
    store: GroundTruthStore | None = None,
    threshold: float = 0.2,
) -> CoherenceScorer:
    """Build the public scorer surface with the dependency-free lite backend."""
    scorer = DirectorConfig(
        mode="general",
        coherence_threshold=threshold,
        hard_limit=threshold,
        soft_limit=threshold,
        scorer_backend="lite",
        cache_size=0,
        adaptive_threshold_enabled=False,
        w_logic=0.0,
        w_fact=1.0,
    ).build_scorer(store)
    scorer._minicheck_nli = None
    return scorer


def test_task_scoring_paths_unit_guard_declares_this_real_surface_companion() -> None:
    """The task-scoring unit guard should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_task_scoring_paths.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_task_scoring_paths_real_surface.py" in category


def test_review_routes_dialogue_through_public_bidirectional_profile(
    deterministic_python_scoring: None,
) -> None:
    """Public review should expose dialogue routing without logical scoring."""
    _ = deterministic_python_scoring
    scorer = _build_lite_scorer()

    approved, score = scorer.review(
        "User: What color is the sky?\nAssistant: It is blue.\nUser: Are you sure?",
        "Yes, the sky is typically blue on a clear day.",
        tenant_id="tenant-a",
    )

    assert approved is True
    assert score.detected_task_type == "dialogue"
    assert score.h_logical == pytest.approx(0.0)
    assert score.h_factual == pytest.approx(0.0)
    assert score.score == pytest.approx(1.0)


def test_review_routes_summarization_to_prompt_evidence(
    deterministic_python_scoring: None,
) -> None:
    """Public review should score summaries against the source prompt."""
    _ = deterministic_python_scoring
    scorer = _build_lite_scorer()
    prompt = (
        "Summarize the deployment note.\n\n"
        "Source document: Director AI stores session logs in coordination paths. "
        "Release evidence stays linked to the scorer manifest."
    )
    response = (
        "Director AI stores session logs in coordination paths. "
        "Release evidence stays linked to the scorer manifest."
    )

    approved, score = scorer.review(prompt, response, tenant_id="tenant-a")

    evidence = score.evidence
    assert approved is True
    assert score.detected_task_type == "summarization"
    assert score.h_logical == pytest.approx(0.0)
    assert evidence is not None
    assert evidence.nli_premise == prompt
    assert evidence.nli_hypothesis == response
    assert evidence.chunks[0].source == "prompt"
    assert evidence.claim_coverage is not None


def test_review_routes_qa_through_real_keyword_retrieval(
    deterministic_python_scoring: None,
) -> None:
    """Public review should attach evidence when QA routing retrieves facts."""
    _ = deterministic_python_scoring
    store = GroundTruthStore()
    store.add(
        "deployment status",
        "Director AI release gates are green for the scoring lane.",
        tenant_id="tenant-a",
    )
    scorer = _build_lite_scorer(store=store)

    approved, score = scorer.review(
        "What is the deployment status?",
        "Director AI release gates are green for the scoring lane.",
        tenant_id="tenant-a",
    )

    evidence = score.evidence
    assert approved is True
    assert score.detected_task_type == "qa"
    assert evidence is not None
    assert evidence.chunks[0].source == "keyword"
    assert evidence.nli_premise == (
        "Director AI release gates are green for the scoring lane."
    )
    assert score.retrieval_confidence == pytest.approx(1.0)
