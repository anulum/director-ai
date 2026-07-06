# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - multimodal guard real-surface tests
"""Public production-surface coverage for multimodal guard primitives."""

from __future__ import annotations

import math

import pytest

from director_ai.core.multimodal_guard import (
    HashBagCrossModalVerifier,
    HashBagImageEncoder,
    MultimodalClaim,
    MultimodalGuard,
    TemporalConsistencyGuard,
)
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS


class _DeterministicEncoder:
    """Public protocol encoder used to exercise verifier contract handling."""

    dim = 2

    def encode(self, image_bytes: bytes) -> tuple[float, ...]:
        """Return a deterministic unit vector after payload validation."""
        if not image_bytes:
            raise ValueError("image_bytes must be non-empty")
        return (1.0, 0.0)


class _ContractVerifier:
    """Public protocol verifier returning a caller-controlled score."""

    dim = 2

    def __init__(self, score: float) -> None:
        """Store the score emitted through the verifier contract."""
        self._score = score

    def verify(self, image_embedding: tuple[float, ...], text: str) -> float:
        """Return the configured score for a valid public contract call."""
        assert image_embedding == (1.0, 0.0)
        assert text.strip()
        return self._score


def _claim() -> MultimodalClaim:
    """Return a minimal public multimodal claim."""
    return MultimodalClaim(
        image_bytes=b"package label bytes",
        text_claim="package label",
    )


def test_multimodal_guard_unit_guard_has_real_surface_companion() -> None:
    """The multimodal guard unit file should declare this companion."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_multimodal_guard.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_multimodal_guard_real_surface.py" in category


@pytest.mark.parametrize("score", (math.nan, math.inf, -math.inf, -0.1, 1.1))
def test_public_guard_rejects_invalid_verifier_similarity(score: float) -> None:
    """Verifier contract breaches should fail before any verdict is emitted."""
    guard = MultimodalGuard(
        encoder=_DeterministicEncoder(),
        verifier=_ContractVerifier(score),
    )

    with pytest.raises(ValueError, match="similarity must be finite and in \\[0, 1\\]"):
        guard.check(_claim())


def test_hashbag_guard_emits_public_tenant_safe_verdict() -> None:
    """The real hash-bag guard should emit only verdict metadata."""
    guard = MultimodalGuard(
        encoder=HashBagImageEncoder(dim=64),
        verifier=HashBagCrossModalVerifier(dim=64),
    )

    verdict = guard.check(_claim())

    assert verdict.label in {"consistent", "uncertain", "hallucinated"}
    assert 0.0 <= verdict.similarity <= 1.0
    assert "package label bytes" not in verdict.reason
    assert "package label" not in verdict.reason


@pytest.mark.parametrize("score", (math.nan, math.inf, -math.inf))
def test_temporal_guard_rejects_non_finite_similarity(score: float) -> None:
    """Temporal similarity updates should also reject non-finite values."""
    guard = TemporalConsistencyGuard()

    with pytest.raises(ValueError, match="similarity"):
        guard.update(score)
