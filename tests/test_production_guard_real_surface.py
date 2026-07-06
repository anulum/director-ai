# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# Copyright 2020-2026 Miroslav Sotek
"""Real-surface coverage for the public ProductionGuard facade."""

from __future__ import annotations

import pytest

from director_ai.core.risk_threshold import RiskFactors
from director_ai.guard import ProductionGuard
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


def test_fast_profile_guard_scores_loaded_facts_through_real_facade() -> None:
    """The fast profile uses real config, storage, and scorer wiring."""
    guard = ProductionGuard.from_profile("fast")
    guard.load_facts(
        {"ibuprofen.max_single_dose": "The maximum single dose of ibuprofen is 400mg."}
    )

    supported = guard.check(
        "What is the maximum single dose of ibuprofen?",
        "The maximum single dose of ibuprofen is 400mg.",
    )
    unsupported = guard.check(
        "What is the maximum single dose of ibuprofen?",
        "Take any amount you want.",
    )

    assert guard.config.profile == "fast"
    assert guard.config.use_nli is False
    assert guard.scorer.threshold == pytest.approx(0.5)
    assert supported.approved is True
    assert supported.score > guard.config.coherence_threshold
    assert supported.coherence.evidence is not None
    assert supported.coherence.evidence.chunks
    assert unsupported.approved is False
    assert unsupported.score < guard.config.coherence_threshold


def test_verified_scoring_runs_real_claim_traceability() -> None:
    """The guard delegates claim support checks to the real verified scorer."""
    guard = ProductionGuard.from_profile("fast")

    result = guard.check_verified(
        "The maximum single dose is 400mg.",
        "Clinical policy says the maximum single dose is 400mg.",
        atomic=True,
    )

    assert result.approved is True
    assert result.supported_count == 1
    assert result.contradicted_count == 0
    assert result.fabricated_count == 0
    assert result.coverage == pytest.approx(1.0)
    assert result.claims[0].verdict == "supported"


def test_verified_scoring_exposes_real_signal_verdicts() -> None:
    """The guard should expose numeric, negation, and entity signal outcomes."""
    guard = ProductionGuard.from_profile("fast")

    numeric = guard.check_verified(
        "Data is retained for 30 days.",
        "Policy says data is retained for 90 days.",
        atomic=True,
    )
    negation = guard.check_verified(
        "Phone support is available on weekends.",
        "Policy says phone support is not available on weekends.",
        atomic=True,
    )
    entity = guard.check_verified(
        "Alice works at Google.",
        "Bob works at Microsoft.",
        atomic=True,
    )

    assert numeric.claims[0].numerical_match is False
    assert numeric.claims[0].traceability > 0.0
    assert negation.claims[0].negation_flip is True
    assert negation.claims[0].traceability == pytest.approx(1.0)
    assert entity.claims[0].entity_match == pytest.approx(0.0)
    assert entity.claims[0].verdict == "contradicted"


def test_rust_signals_unit_guard_declares_real_surface_companions() -> None:
    """The Rust signal unit guard is backed by public workflow tests."""
    classification, reason = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_rust_signals.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_production_guard_real_surface.py" in reason
    assert "tests/test_streaming_runtime_real_surface.py" in reason
    assert "tests/test_vector_store_real_surface.py" in reason


def test_stateful_facades_share_real_guard_context() -> None:
    """Canary, preflight, and risk threshold facades keep real guard state."""
    guard = ProductionGuard.from_profile("fast")

    canary = guard.plant_canary("tenant-a", token="CANARY-STATIC")
    signals = guard.scan_canaries(
        f"The draft leaked {canary.token}.",
        "tenant-a",
        evidence=(),
    )
    preflight = guard.preflight
    threshold = guard.risk_threshold(
        RiskFactors(
            domain="medical",
            retrieval_confidence=0.25,
            action_reversibility=0.2,
            external_exposure=True,
            pii_present=True,
        )
    )

    assert canary.tenant_id == "tenant-a"
    assert signals
    assert signals[0].canary_id == canary.canary_id
    assert signals[0].signal == "leakage"
    assert guard.preflight is preflight
    assert threshold.threshold > threshold.base_threshold
    assert {"domain", "external_exposure", "pii_present"}.issubset(
        threshold.contributions
    )
