# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI - multimodal factory real-surface tests
"""Real public-surface coverage for multimodal adapter factory configuration."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from director_ai.core.config import DirectorConfig
from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.multimodal_guard import (
    MultimodalCheckRequest,
    MultimodalVerifierAdapter,
    build_hashbag_adapter,
)
from director_ai.guard import ProductionGuard
from tools.test_surface_policy_manifest import KNOWN_TEST_SURFACE_CLASSIFICATIONS

_ENVELOPE = RiskEnvelope(
    action_category="multimodal",
    reversibility="reversible",
    domain="general",
    calibrated_threshold=0.5,
    no_go_threshold=0.9,
)


def _video_request() -> MultimodalCheckRequest:
    """Return a public video-check request with private claim text."""
    return MultimodalCheckRequest(
        modality="video",
        claim_text="Private inspection claim about a labelled package.",
        media_ref="media://video-42",
        frame_similarities=(0.9, 0.0),
    )


def _audio_request(
    *, transcript_text: str = "policy refund window", caption_text: str = ""
) -> MultimodalCheckRequest:
    """Return a public audio-check request for adapter decision tests."""
    return MultimodalCheckRequest(
        modality="audio",
        claim_text="policy refund window",
        media_ref="media://audio-42",
        transcript_text=transcript_text,
        caption_text=caption_text,
    )


def _exact_text_score(_reference: str, _claim: str) -> float:
    """Return a supported text-grounding score."""
    return 1.0


def _low_text_score(_reference: str, _claim: str) -> float:
    """Return a floor-breaching text-grounding score."""
    return 0.1


def _middle_text_score(_reference: str, _claim: str) -> float:
    """Return an uncertain text-grounding score."""
    return 0.5


def _invalid_text_score(_reference: str, _claim: str) -> float:
    """Return an invalid score to exercise fail-closed validation."""
    return 1.1


def test_multimodal_factory_unit_guard_has_real_surface_companion() -> None:
    """The factory unit guard should be backed by public adapter coverage."""
    classification, category = KNOWN_TEST_SURFACE_CLASSIFICATIONS[
        "tests/test_multimodal_factory.py"
    ]

    assert classification == "unit-guard-with-companion"
    assert "tests/test_multimodal_factory_real_surface.py" in category


@pytest.mark.parametrize(
    ("modality", "claim_text", "media_ref", "match"),
    (
        ("document", "claim", "media://x", "unsupported modality"),
        ("audio", " ", "media://x", "claim_text"),
        ("audio", "claim", " ", "media_ref"),
    ),
)
def test_request_validation_rejects_unsafe_payload_shape(
    modality: object,
    claim_text: str,
    media_ref: str,
    match: str,
) -> None:
    """Request construction should reject unsupported or empty critical fields."""
    with pytest.raises(ValueError, match=match):
        MultimodalCheckRequest(
            modality=cast(Any, modality),
            claim_text=claim_text,
            media_ref=media_ref,
        )


@pytest.mark.parametrize(
    ("enabled", "benchmarked", "match"),
    (
        (("document",), (), "unsupported enabled modalities"),
        (("image",), ("document",), "unsupported benchmarked modalities"),
        (("image",), ("video",), "benchmarked modalities must be enabled"),
    ),
)
def test_adapter_configuration_rejects_invalid_modality_sets(
    enabled: tuple[str, ...],
    benchmarked: tuple[str, ...],
    match: str,
) -> None:
    """Adapter construction should fail closed for invalid modality policy."""
    with pytest.raises(ValueError, match=match):
        build_hashbag_adapter(
            enabled_modalities=enabled,
            benchmarked_modalities=benchmarked,
        )


@pytest.mark.parametrize(
    ("temporal_alpha", "temporal_floor", "match"),
    (
        (0.0, 0.2, "temporal_alpha"),
        (1.1, 0.2, "temporal_alpha"),
        (0.5, -0.1, "temporal_floor"),
        (0.5, 1.1, "temporal_floor"),
    ),
)
def test_hashbag_adapter_rejects_invalid_temporal_bounds_at_construction(
    temporal_alpha: float,
    temporal_floor: float,
    match: str,
) -> None:
    """Invalid video-temporal policy should fail before a check is accepted."""
    with pytest.raises(ValueError, match=match):
        build_hashbag_adapter(
            enabled_modalities=("video",),
            benchmarked_modalities=("video",),
            temporal_alpha=temporal_alpha,
            temporal_floor=temporal_floor,
        )


def test_benchmarked_audio_supported_path_allows_and_emits_safety_event() -> None:
    """A benchmarked supported audio check should become an allow event."""
    adapter = build_hashbag_adapter(
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )

    result = adapter.check(_audio_request(), risk_envelope=_ENVELOPE, policy_id="p")
    event = result.to_safety_event(
        hook_id="multimodal",
        request_id="request-1",
        tenant_id="tenant-1",
        latency_ms=3.5,
    )

    assert result.guard_decision.decision == "allow"
    assert result.guard_decision.reason == "multimodal_supported"
    assert event.policy_decision == "allow"
    assert event.hook_id == "multimodal"
    assert event.request_id == "request-1"
    assert event.tenant_id == "tenant-1"


def test_disabled_request_modality_is_rejected() -> None:
    """Enabled modality policy should reject mismatched request modalities."""
    adapter = build_hashbag_adapter(
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    with pytest.raises(ValueError, match="is not enabled"):
        adapter.check(_audio_request(), risk_envelope=_ENVELOPE, policy_id="p")


def test_image_modality_requires_image_guard() -> None:
    """Direct adapter construction should not silently pass image checks."""
    adapter = MultimodalVerifierAdapter(
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    with pytest.raises(ValueError, match="image_guard"):
        adapter.check(
            MultimodalCheckRequest(
                modality="image",
                claim_text="image claim",
                media_ref="media://image-1",
                image_bytes=b"image",
            ),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )


def test_audio_modality_requires_score_function() -> None:
    """Direct adapter construction should not silently pass audio checks."""
    adapter = MultimodalVerifierAdapter(
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )

    with pytest.raises(ValueError, match="audio_score_fn"):
        adapter.check(_audio_request(), risk_envelope=_ENVELOPE, policy_id="p")


def test_audio_modality_requires_transcript_text() -> None:
    """Audio checks should reject empty transcript payloads before scoring."""
    adapter = build_hashbag_adapter(
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )

    with pytest.raises(ValueError, match="transcript_text"):
        adapter.check(
            _audio_request(transcript_text=" "),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )


def test_audio_score_must_be_unit_interval() -> None:
    """Invalid score callbacks should fail closed instead of emitting decisions."""
    adapter = MultimodalVerifierAdapter(
        audio_score_fn=_invalid_text_score,
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )

    with pytest.raises(ValueError, match="score must be finite"):
        adapter.check(_audio_request(), risk_envelope=_ENVELOPE, policy_id="p")


def test_caption_grounding_floor_halts_supported_audio() -> None:
    """Caption grounding below the floor should halt an otherwise supported claim."""
    adapter = MultimodalVerifierAdapter(
        audio_score_fn=_exact_text_score,
        caption_score_fn=_low_text_score,
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )

    result = adapter.check(
        _audio_request(caption_text="conflicting caption"),
        risk_envelope=_ENVELOPE,
        policy_id="p",
    )

    assert result.guard_decision.decision == "halt"
    assert result.guard_decision.reason == "multimodal_hallucinated"
    assert "media://audio-42#caption" in result.signal.evidence_refs


def test_caption_grounding_allow_threshold_warns_supported_audio() -> None:
    """Caption grounding below allow threshold should warn but not halt."""
    adapter = MultimodalVerifierAdapter(
        audio_score_fn=_exact_text_score,
        caption_score_fn=_middle_text_score,
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )

    result = adapter.check(
        _audio_request(caption_text="partially grounded caption"),
        risk_envelope=_ENVELOPE,
        policy_id="p",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "multimodal_uncertain"
    assert result.signal.verdict == "uncertain"


def test_video_modality_requires_frame_similarities() -> None:
    """Video checks should reject missing temporal evidence before scoring."""
    adapter = build_hashbag_adapter(
        enabled_modalities=("video",),
        benchmarked_modalities=("video",),
    )

    with pytest.raises(ValueError, match="frame_similarities"):
        adapter.check(
            MultimodalCheckRequest(
                modality="video",
                claim_text="steady scene",
                media_ref="media://video-empty",
            ),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )


def test_video_consistent_temporal_path_allows() -> None:
    """Stable benchmarked video frame similarities should support the claim."""
    adapter = build_hashbag_adapter(
        enabled_modalities=("video",),
        benchmarked_modalities=("video",),
        temporal_alpha=0.5,
        temporal_floor=0.5,
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="video",
            claim_text="steady scene",
            media_ref="media://video-stable",
            frame_similarities=(0.95, 0.9),
        ),
        risk_envelope=_ENVELOPE,
        policy_id="p",
    )

    assert result.guard_decision.decision == "allow"
    assert result.signal.verdict == "consistent"


def test_production_guard_rejects_invalid_temporal_config_before_check() -> None:
    """Configured guard paths should fail before the first video request."""
    guard = ProductionGuard(
        config=DirectorConfig(
            multimodal_enabled_modalities=("video",),
            multimodal_benchmarked_modalities=("video",),
            multimodal_temporal_alpha=0.0,
        )
    )

    with pytest.raises(ValueError, match="temporal_alpha"):
        _ = guard.multimodal_adapter


def test_hashbag_video_adapter_emits_tenant_safe_halt() -> None:
    """The public hash-bag adapter should halt temporal drift without leaks."""
    adapter = build_hashbag_adapter(
        enabled_modalities=("video",),
        benchmarked_modalities=("video",),
        temporal_alpha=0.5,
        temporal_floor=0.5,
    )

    result = adapter.check(_video_request(), risk_envelope=_ENVELOPE, policy_id="p")
    payload = result.to_dict()
    guard_decision = cast(dict[str, object], payload["guard_decision"])
    encoded = json.dumps(payload, sort_keys=True)

    assert guard_decision["decision"] == "halt"
    assert guard_decision["reason"] == "multimodal_hallucinated"
    assert "media://video-42#frame:1" in cast(
        tuple[str, ...], result.signal.evidence_refs
    )
    assert "Private inspection claim" not in encoded
    assert "frame_similarities" not in encoded
