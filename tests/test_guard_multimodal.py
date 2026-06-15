# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — ProductionGuard in-process multimodal live-path tests

from __future__ import annotations

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.multimodal_guard import (
    MultimodalCheckRequest,
    MultimodalCheckResult,
)
from director_ai.guard import ProductionGuard


def _guard(*, enabled=("image",), benchmarked=("image",)) -> ProductionGuard:
    return ProductionGuard(
        config=DirectorConfig(
            use_nli=False,
            multimodal_enabled_modalities=enabled,
            multimodal_benchmarked_modalities=benchmarked,
        )
    )


def _request(modality="image") -> MultimodalCheckRequest:
    return MultimodalCheckRequest(
        modality=modality,
        claim_text="a photograph of a tabby cat on a sofa",
        media_ref="img://catalogue/1",
        image_bytes=b"\x89PNG fake image payload bytes for hashing",
    )


def test_check_multimodal_returns_result():
    guard = _guard()
    result = guard.check_multimodal(_request())
    assert isinstance(result, MultimodalCheckResult)
    assert result.signal.modality == "image"
    assert result.guard_decision is not None
    # tenant-safe serialisation does not echo the claim text or media bytes
    payload = result.to_dict()
    assert "tabby cat" not in str(payload)


def test_adapter_is_cached():
    guard = _guard()
    first = guard.multimodal_adapter
    assert guard.multimodal_adapter is first


def test_disabled_multimodal_raises():
    guard = ProductionGuard(config=DirectorConfig(use_nli=False))  # no modalities
    with pytest.raises(RuntimeError, match="multimodal guard is disabled"):
        guard.check_multimodal(_request())


def test_unbenchmarked_modality_warns():
    # enabled but not benchmarked -> never silently passes
    guard = _guard(enabled=("image",), benchmarked=())
    result = guard.check_multimodal(_request())
    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "multimodal_unbenchmarked"


def test_injected_adapter_is_used():
    guard = _guard()
    sentinel = object()

    class _StubAdapter:
        def check(self, request, *, risk_envelope, policy_id):
            return sentinel

    assert guard.check_multimodal(_request(), adapter=_StubAdapter()) is sentinel


def test_disabled_modality_in_request_rejected():
    guard = _guard(enabled=("image",), benchmarked=("image",))
    with pytest.raises(ValueError):
        guard.check_multimodal(_request(modality="audio"))  # not enabled
