# ─────────────────────────────────────────────────────────────────────
# Director-Class AI — Shared Test Fixtures
# (C) 1998-2026 Miroslav Sotek. All rights reserved.
# License: GNU AGPL v3 | Commercial licensing available
# ─────────────────────────────────────────────────────────────────────

import pytest

from director_ai.core import (
    CoherenceAgent,
    CoherenceScorer,
    GroundTruthStore,
    MockGenerator,
    SafetyKernel,
)


@pytest.fixture
def agent():
    """Pre-configured CoherenceAgent (mock mode)."""
    return CoherenceAgent()


@pytest.fixture
def scorer():
    """CoherenceScorer with default threshold and ground truth store."""
    store = GroundTruthStore()
    return CoherenceScorer(threshold=0.6, ground_truth_store=store)


@pytest.fixture
def strict_scorer():
    """CoherenceScorer with a strict threshold (0.7)."""
    store = GroundTruthStore()
    return CoherenceScorer(threshold=0.7, ground_truth_store=store)


@pytest.fixture
def kernel():
    """SafetyKernel instance."""
    return SafetyKernel()


@pytest.fixture
def store():
    """GroundTruthStore with mock facts."""
    return GroundTruthStore()


@pytest.fixture
def generator():
    """MockGenerator instance."""
    return MockGenerator()
