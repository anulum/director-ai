# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Multimodal verifier adapter tests."""

from __future__ import annotations

import pytest

from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.multimodal_guard import (
    MultimodalCheckRequest,
    MultimodalVerifierAdapter,
)


class _ConstantGuard:
    def __init__(self, label: str, similarity: float) -> None:
        self.label = label
        self.similarity = similarity

    def check(self, claim):
        return type(
            "Verdict",
            (),
            {
                "label": self.label,
                "similarity": self.similarity,
                "reason": f"{self.label}:{self.similarity}",
            },
        )()


def _envelope() -> RiskEnvelope:
    return RiskEnvelope(
        action_category="multimodal",
        reversibility="reversible",
        domain="regulated",
        calibrated_threshold=0.5,
        no_go_threshold=0.85,
    )


def test_request_validates_required_fields_and_normalises_metadata():
    request = MultimodalCheckRequest(
        modality="video",
        claim_text="The frame sequence is stable.",
        media_ref="media://video-normalised",
        frame_similarities=("0.1", 0.2, 0.3),
        metadata={"frame_count": 3, "source": "sensor"},
    )

    assert request.frame_similarities == (0.1, 0.2, 0.3)
    assert request.metadata == {"frame_count": "3", "source": "sensor"}

    with pytest.raises(ValueError, match="claim_text is required"):
        MultimodalCheckRequest(
            modality="image",
            claim_text=" ",
            media_ref="media://image",
        )
    with pytest.raises(ValueError, match="media_ref is required"):
        MultimodalCheckRequest(
            modality="image",
            claim_text="A claim.",
            media_ref=" ",
        )


def test_adapter_requires_enabled_modalities_and_unit_thresholds():
    with pytest.raises(ValueError, match="at least one enabled modality"):
        MultimodalVerifierAdapter(enabled_modalities=())

    with pytest.raises(ValueError, match=r"score must be finite and in \[0, 1\]"):
        MultimodalVerifierAdapter(
            enabled_modalities=("image",),
            grounding_floor=1.1,
        )


def test_image_hallucination_maps_to_halt_without_raw_media_in_audit():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("hallucinated", 0.05),
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )
    request = MultimodalCheckRequest(
        modality="image",
        claim_text="The image shows a medical device.",
        media_ref="media://image-1",
        image_bytes=b"raw image bytes",
    )

    result = adapter.check(
        request,
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "halt"
    assert result.signal.verdict == "hallucinated"
    assert "raw image bytes" not in str(result.to_dict())
    assert "medical device" not in str(result.to_dict())
    assert result.to_safety_event(hook_id="multimodal.check").hook_scope == "agent"


def test_image_modality_requires_guard_and_valid_similarity():
    no_guard = MultimodalVerifierAdapter(enabled_modalities=("image",))
    request = MultimodalCheckRequest(
        modality="image",
        claim_text="The image shows a badge.",
        media_ref="media://image-no-guard",
        image_bytes=b"payload",
    )

    with pytest.raises(ValueError, match="image modality requires image_guard"):
        no_guard.check(
            request,
            risk_envelope=_envelope(),
            policy_id="policy.multimodal.regulated",
        )

    invalid_score = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("consistent", 1.2),
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )
    with pytest.raises(ValueError, match=r"score must be finite and in \[0, 1\]"):
        invalid_score.check(
            request,
            risk_envelope=_envelope(),
            policy_id="policy.multimodal.regulated",
        )


def test_uncertain_image_evidence_warns_instead_of_allowing():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("uncertain", 0.3),
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="image",
            claim_text="The image shows a badge.",
            media_ref="media://image-2",
            image_bytes=b"payload",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "multimodal_uncertain"


def test_supported_image_evidence_allows_when_benchmarked():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("consistent", 0.95),
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="image",
            claim_text="The image shows a labelled component.",
            media_ref="media://image-allow",
            image_bytes=b"payload",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "allow"
    assert result.guard_decision.reason == "multimodal_supported"
    assert result.guard_decision.attributes["benchmarked"] == "true"


def test_unbenchmarked_consistent_modality_warns():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("consistent", 0.9),
        enabled_modalities=("image",),
        benchmarked_modalities=(),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="image",
            claim_text="The image shows a badge.",
            media_ref="media://image-3",
            image_bytes=b"payload",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "warn"
    assert result.guard_decision.reason == "multimodal_unbenchmarked"


def test_caption_grounding_can_halt_supported_image_claim():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("consistent", 0.94),
        caption_score_fn=lambda caption, claim: 0.12,
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="image",
            claim_text="The image shows an approved label.",
            media_ref="media://image-4",
            image_bytes=b"payload",
            caption_text="Caption says the label is missing.",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "halt"
    assert result.signal.verdict == "hallucinated"
    assert result.guard_decision.evidence_refs == (
        "media://image-4",
        "media://image-4#caption",
    )
    assert result.guard_decision.attributes["caption_grounded"] == "true"
    assert "Caption says" not in str(result.to_dict())


def test_caption_grounding_can_warn_supported_image_claim():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("consistent", 0.94),
        caption_score_fn=lambda caption, claim: 0.55,
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="image",
            claim_text="The image shows an approved label.",
            media_ref="media://image-caption-warn",
            image_bytes=b"payload",
            caption_text="Caption partly supports the label.",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "warn"
    assert result.signal.verdict == "uncertain"
    assert result.guard_decision.evidence_refs == (
        "media://image-caption-warn",
        "media://image-caption-warn#caption",
    )


def test_metadata_grounding_refs_are_audit_safe_on_allow():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("consistent", 0.93),
        metadata_score_fn=lambda metadata, claim: 0.91,
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="image",
            claim_text="The product was captured in 2026.",
            media_ref="media://image-5",
            image_bytes=b"payload",
            metadata={"captured_at": "2026-05-13", "device": "private-camera"},
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "allow"
    assert result.guard_decision.evidence_refs == (
        "media://image-5",
        "media://image-5#metadata:captured_at",
        "media://image-5#metadata:device",
    )
    assert result.guard_decision.attributes["metadata_grounded"] == "true"
    assert "private-camera" not in str(result.to_dict())
    assert "2026-05-13" not in str(result.to_dict())


def test_metadata_grounding_preserves_existing_hallucination_verdict():
    adapter = MultimodalVerifierAdapter(
        image_guard=_ConstantGuard("hallucinated", 0.2),
        metadata_score_fn=lambda metadata, claim: 0.95,
        enabled_modalities=("image",),
        benchmarked_modalities=("image",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="image",
            claim_text="The image shows a verified product.",
            media_ref="media://image-metadata-halt",
            image_bytes=b"payload",
            metadata={"capture": "verified"},
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "halt"
    assert result.signal.verdict == "hallucinated"
    assert result.guard_decision.evidence_refs == (
        "media://image-metadata-halt",
        "media://image-metadata-halt#metadata:capture",
    )


def test_audio_transcript_adapter_can_allow_benchmarked_supported_claim():
    adapter = MultimodalVerifierAdapter(
        audio_score_fn=lambda transcript, claim: 0.92,
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="audio",
            claim_text="The speaker says approved.",
            media_ref="media://audio-1",
            transcript_text="approved approved approved",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "allow"
    assert result.signal.modality == "audio"
    assert "approved approved" not in str(result.to_dict())


def test_audio_modality_requires_score_function_and_transcript():
    no_scorer = MultimodalVerifierAdapter(enabled_modalities=("audio",))
    request = MultimodalCheckRequest(
        modality="audio",
        claim_text="The speaker says approved.",
        media_ref="media://audio-missing",
        transcript_text="approved",
    )

    with pytest.raises(ValueError, match="audio modality requires audio_score_fn"):
        no_scorer.check(
            request,
            risk_envelope=_envelope(),
            policy_id="policy.multimodal.regulated",
        )

    no_transcript = MultimodalVerifierAdapter(
        audio_score_fn=lambda transcript, claim: 0.9,
        enabled_modalities=("audio",),
    )
    with pytest.raises(ValueError, match="audio modality requires transcript_text"):
        no_transcript.check(
            MultimodalCheckRequest(
                modality="audio",
                claim_text="The speaker says approved.",
                media_ref="media://audio-no-transcript",
            ),
            risk_envelope=_envelope(),
            policy_id="policy.multimodal.regulated",
        )


def test_audio_score_bands_map_to_warn_and_halt():
    adapter = MultimodalVerifierAdapter(
        audio_score_fn=lambda transcript, claim: 0.5,
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )
    uncertain = adapter.check(
        MultimodalCheckRequest(
            modality="audio",
            claim_text="The speaker says approved.",
            media_ref="media://audio-uncertain",
            transcript_text="partially approved",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    halt_adapter = MultimodalVerifierAdapter(
        audio_score_fn=lambda transcript, claim: 0.1,
        enabled_modalities=("audio",),
        benchmarked_modalities=("audio",),
    )
    halted = halt_adapter.check(
        MultimodalCheckRequest(
            modality="audio",
            claim_text="The speaker says approved.",
            media_ref="media://audio-halt",
            transcript_text="denied",
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert uncertain.guard_decision.decision == "warn"
    assert uncertain.signal.verdict == "uncertain"
    assert halted.guard_decision.decision == "halt"
    assert halted.signal.verdict == "hallucinated"


def test_video_temporal_drop_halts_with_frame_evidence_refs():
    adapter = MultimodalVerifierAdapter(
        enabled_modalities=("video",),
        benchmarked_modalities=("video",),
    )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="video",
            claim_text="The object remains in view.",
            media_ref="media://video-1",
            frame_similarities=(0.9, 0.1, 0.0, 0.0),
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "halt"
    assert result.signal.verdict == "temporal_inconsistent"
    assert "media://video-1#frame:3" in result.guard_decision.evidence_refs


def test_video_requires_frames_and_warns_on_low_consistent_ema():
    adapter = MultimodalVerifierAdapter(
        enabled_modalities=("video",),
        benchmarked_modalities=("video",),
        temporal_floor=0.0,
    )

    with pytest.raises(ValueError, match="video modality requires frame_similarities"):
        adapter.check(
            MultimodalCheckRequest(
                modality="video",
                claim_text="The object remains in view.",
                media_ref="media://video-empty",
            ),
            risk_envelope=_envelope(),
            policy_id="policy.multimodal.regulated",
        )

    result = adapter.check(
        MultimodalCheckRequest(
            modality="video",
            claim_text="The object remains in view.",
            media_ref="media://video-uncertain",
            frame_similarities=(0.5, 0.55, 0.6),
        ),
        risk_envelope=_envelope(),
        policy_id="policy.multimodal.regulated",
    )

    assert result.guard_decision.decision == "warn"
    assert result.signal.verdict == "uncertain"


def test_unsupported_or_disabled_modality_raises_instead_of_silent_pass():
    adapter = MultimodalVerifierAdapter(enabled_modalities=("image",))

    with pytest.raises(ValueError, match="not enabled"):
        adapter.check(
            MultimodalCheckRequest(
                modality="audio",
                claim_text="A spoken claim.",
                media_ref="media://audio-2",
                transcript_text="spoken claim",
            ),
            risk_envelope=_envelope(),
            policy_id="policy.multimodal.regulated",
        )

    with pytest.raises(ValueError, match="unsupported modality"):
        MultimodalCheckRequest(
            modality="lidar",  # type: ignore[arg-type]
            claim_text="A point cloud claim.",
            media_ref="media://lidar-1",
        )


def test_configured_modalities_are_validated_before_runtime_checks():
    with pytest.raises(ValueError, match="unsupported enabled modalities"):
        MultimodalVerifierAdapter(enabled_modalities=("lidar",))

    with pytest.raises(ValueError, match="benchmarked modalities must be enabled"):
        MultimodalVerifierAdapter(
            enabled_modalities=("image",),
            benchmarked_modalities=("audio",),
        )
