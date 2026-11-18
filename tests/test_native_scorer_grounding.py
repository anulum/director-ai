# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — native scorer tenant and model-policy integration
"""Exercise real native scoring through scoped, concurrent and refused calls."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from backfire_kernel import BackfireConfig, RustCoherenceScorer

from director_ai.core.config import DirectorConfig
from director_ai.core.retrieval.knowledge import GroundTruthStore
from director_ai.core.runtime.session import ConversationSession
from director_ai.core.scoring.scorer import CoherenceScorer

BACKENDS = ["rust", "backfire"]
PROMPT = "reference"
ACTION = "alpha reference"


@pytest.mark.parametrize("entry", ["sync", "async"])
def test_repeated_review_preserves_configured_claim_verification(entry: str) -> None:
    """A base score above threshold must still receive its claim verdict."""
    store = GroundTruthStore()
    tenant = "team"
    supported = "Notifications support Slack and Microsoft Teams."
    store.add("notifications", supported, tenant_id=tenant)
    scorer = DirectorConfig(
        mode="grounded",
        scorer_backend="deberta",
        use_nli=False,
        hybrid_retrieval=False,
        reranker_enabled=False,
        coherence_threshold=0.1,
        adaptive_threshold_enabled=False,
        verified_scorer_enabled=True,
        verified_scorer_low_confidence_margin=1.0,
        cache_size=8,
    ).build_scorer(store)
    prompt = (
        "Based on the following source document, which notifications are supported?"
    )
    action = "Notifications support WhatsApp Business approvals."
    for _ in range(2):
        if entry == "async":
            approved, score = asyncio.run(
                scorer.areview(prompt, action, tenant_id=tenant)
            )
        else:
            approved, score = scorer.review(prompt, action, tenant_id=tenant)
        assert score.degraded_mode
        assert score.score >= scorer.threshold
        assert not approved and not score.approved
        assert score.verified_approved is False
        assert score.verified_result is not None
        assert score.evidence is not None
    assert scorer.history == []
    assert scorer.cache is not None and scorer.cache.size == 0
    approved, _ = scorer.review(prompt, supported, tenant_id=tenant)
    assert approved
    assert scorer.history == [supported]


def test_review_records_each_session_with_the_same_context() -> None:
    """Equivalent contexts must still record the review in both sessions."""
    scorer = CoherenceScorer(use_nli=False, cache_size=8)
    for _ in range(2):
        session = ConversationSession(track_intent_drift=True)
        session.add_turn("Earlier question", "Reference context.", 0.9)
        _, score = scorer.review(PROMPT, ACTION, session=session, tenant_id="team")
        assert [turn.response for turn in session.turns] == [
            "Reference context.",
            ACTION,
        ]
        assert session.turns[-1].score == score.score
        assert score.intent_drift_risk is not None
    assert scorer.cache is not None and scorer.cache.hits == 1


def test_enabled_output_detection_runs_after_stateless_cache_is_warm() -> None:
    """Enabling output detection must apply to an already reviewed request."""
    scorer = CoherenceScorer(use_nli=False, cache_size=8)
    _, initial = scorer.review(PROMPT, ACTION)
    assert initial.injection_risk is None
    assert scorer.cache is not None and scorer.cache.size == 1
    scorer.enable_injection_detection()
    _, score = scorer.review(PROMPT, ACTION)
    assert score.score == initial.score
    assert score.injection_risk is not None
    assert scorer.cache.hits == 1


def _reference(store: GroundTruthStore, tenant: str) -> tuple[float, float]:
    """Use the actual native API with an explicitly bound real-store callback."""

    def retrieve(query: str) -> str | None:
        """Retrieve only the independently specified reference tenant."""
        return store.retrieve_context(query, tenant_id=tenant)

    native = RustCoherenceScorer(
        config=BackfireConfig(coherence_threshold=0.5), knowledge_callback=retrieve
    )
    _, score = native.review(PROMPT, ACTION)
    return float(score.h_logical), float(score.h_factual)


def _factual(scorer: CoherenceScorer, entry: str, tenant: str) -> float:
    """Read the factual signal through an actual supported public entry point."""
    if entry == "factual":
        return scorer.calculate_factual_divergence(PROMPT, ACTION, tenant_id=tenant)
    if entry == "batch":
        return scorer.review_batch([(PROMPT, ACTION)] * 2, tenant_id=tenant)[0][
            1
        ].h_factual
    if entry == "async":
        return asyncio.run(scorer.areview(PROMPT, ACTION, tenant_id=tenant))[
            1
        ].h_factual
    return scorer.review(PROMPT, ACTION, tenant_id=tenant)[1].h_factual


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("entry", ["review", "factual", "batch", "async"])
def test_every_tenant_entry_matches_explicitly_scoped_native_reference(
    backend: str, entry: str
) -> None:
    """A foreign fact cannot affect any supported tenant-bearing scoring API."""
    store = GroundTruthStore()
    store.add(PROMPT, ACTION, tenant_id="owner")
    scorer = CoherenceScorer(
        use_nli=False, scorer_backend=backend, ground_truth_store=store
    )
    for tenant in ["owner", "empty"]:
        expected = _reference(store, tenant)[1]
        assert _factual(scorer, entry, tenant) == pytest.approx(expected)
    store.add(PROMPT, ACTION, tenant_id="foreign")
    assert _factual(scorer, entry, "empty") == pytest.approx(0.5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_direct_logical_divergence_retains_the_native_signal(backend: str) -> None:
    """The public logical calculator agrees with the actual native review path."""
    store = GroundTruthStore()
    store.add(PROMPT, ACTION)
    expected = _reference(store, "")[0]
    scorer = CoherenceScorer(
        use_nli=False, scorer_backend=backend, ground_truth_store=store
    )
    assert scorer.calculate_logical_divergence(PROMPT, ACTION) == pytest.approx(
        expected
    )
    assert scorer.review(PROMPT, ACTION)[1].h_logical == pytest.approx(expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_review_keeps_native_logical_signal_and_separates_threads(
    backend: str,
) -> None:
    """One shared scorer must match real native references across worker threads."""
    store = GroundTruthStore()
    store.add(PROMPT, ACTION, tenant_id="owner")
    store.add(PROMPT, "beta reference", tenant_id="other")
    tenants = ["owner", "other", "empty"]
    expected = {tenant: _reference(store, tenant) for tenant in tenants}
    scorer = CoherenceScorer(
        use_nli=False, scorer_backend=backend, ground_truth_store=store
    )

    def review(tenant: str) -> tuple[float, float]:
        """Call the real shared scorer from this worker's execution context."""
        _, score = scorer.review(PROMPT, ACTION, tenant_id=tenant)
        return score.h_logical, score.h_factual

    with ThreadPoolExecutor(max_workers=2) as pool:
        identities = tenants * 8
        results = list(pool.map(review, identities))
    for actual, tenant in zip(results, identities, strict=True):
        assert actual == pytest.approx(expected[tenant])


@pytest.mark.parametrize("backend", BACKENDS)
def test_invalid_native_request_restores_the_next_unscoped_grounding_context(
    backend: str,
) -> None:
    """A genuine FFI argument error must not leave tenant context on later calls."""
    store = GroundTruthStore()
    store.add(PROMPT, ACTION)
    scorer = CoherenceScorer(
        use_nli=False, scorer_backend=backend, ground_truth_store=store
    )
    expected = _reference(store, "")[1]
    invalid_prompt = json.loads("null")
    with pytest.raises(TypeError):
        scorer.review(invalid_prompt, ACTION, tenant_id="empty")
    assert scorer.calculate_factual_divergence(PROMPT, ACTION) == pytest.approx(
        expected
    )
    assert scorer.review(PROMPT, ACTION)[1].h_factual == pytest.approx(expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_backend_without_a_store_does_not_invent_demo_grounding(
    backend: str,
) -> None:
    """The Python no-KB contract stays neutral even for a native demo fact query."""
    scorer = CoherenceScorer(use_nli=False, scorer_backend=backend)
    assert scorer.review("sky color", "sky color is blue")[1].h_factual == 0.5
    assert scorer.calculate_factual_divergence("sky color", "sky color is blue") == 0.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_strict_native_profile_rejects_missing_models_consistently(
    backend: str,
) -> None:
    """Strict mode must use the common rejection path rather than native heuristics."""
    store = GroundTruthStore()
    store.add(PROMPT, ACTION)
    scorer = CoherenceScorer(
        use_nli=False,
        scorer_backend=backend,
        ground_truth_store=store,
        strict_mode=True,
    )
    approved, score = scorer.review(PROMPT, ACTION)
    assert not approved
    assert score.strict_mode_rejected
    assert score.degraded_mode
    assert score.h_logical == pytest.approx(0.9)
    assert score.h_factual == pytest.approx(0.9)
    assert scorer.calculate_logical_divergence(PROMPT, ACTION) == pytest.approx(0.9)
    assert scorer.calculate_factual_divergence(PROMPT, ACTION) == pytest.approx(0.9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("entry", ["review", "batch", "async"])
def test_required_model_policy_refuses_all_review_entrypoints(
    backend: str, entry: str
) -> None:
    """The batch and executor APIs retain the same required-model refusal."""
    scorer = CoherenceScorer(
        use_nli=False, scorer_backend=backend, require_model_backed_nli=True
    )
    with pytest.raises(RuntimeError, match="requires model-backed NLI"):
        _factual(scorer, entry, "owner")
