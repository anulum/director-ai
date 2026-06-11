# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multimodal factory + text-bag similarity tests

"""Tests for the dependency-free multimodal adapter factory and the
text-to-text hash-bag similarity it grounds audio/caption/metadata with:
identity/disjoint/empty similarity behaviour, case-insensitivity, dim
validation, and that the built adapter scores image, audio, and video."""

from __future__ import annotations

import pytest

from director_ai.core.guard_control import RiskEnvelope
from director_ai.core.multimodal_guard import (
    MultimodalCheckRequest,
    build_hashbag_adapter,
    text_bag_similarity,
)

_ENVELOPE = RiskEnvelope(
    action_category="multimodal",
    reversibility="reversible",
    domain="general",
    calibrated_threshold=0.5,
    no_go_threshold=0.9,
)


class TestTextBagSimilarity:
    def test_identical_text_is_one(self):
        assert text_bag_similarity("refund policy terms", "refund policy terms") == (
            pytest.approx(1.0)
        )

    def test_disjoint_text_is_zero(self):
        assert text_bag_similarity("alpha beta", "gamma delta") == pytest.approx(0.0)

    def test_empty_input_is_zero(self):
        assert text_bag_similarity("", "something") == 0.0
        assert text_bag_similarity("something", "   ") == 0.0

    def test_partial_overlap_between(self):
        score = text_bag_similarity("refund within 30 days", "refund within 14 days")
        assert 0.0 < score < 1.0

    def test_case_insensitive_by_default(self):
        assert text_bag_similarity("Refund Policy", "refund policy") == pytest.approx(
            1.0
        )

    def test_invalid_dim_rejected(self):
        with pytest.raises(ValueError, match="dim must be positive"):
            text_bag_similarity("a", "b", dim=0)


class TestBuildHashbagAdapter:
    def _adapter(self, **kwargs):
        kwargs.setdefault("enabled_modalities", ("image", "audio", "video"))
        kwargs.setdefault("benchmarked_modalities", ("image", "audio", "video"))
        return build_hashbag_adapter(**kwargs)

    def test_scores_image(self):
        adapter = self._adapter()
        result = adapter.check(
            MultimodalCheckRequest(
                modality="image",
                claim_text="a lake",
                media_ref="m",
                image_bytes=b"some-image-bytes",
            ),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )
        assert result.guard_decision.decision in {"allow", "warn", "halt"}

    def test_scores_audio_via_text_similarity(self):
        adapter = self._adapter()
        result = adapter.check(
            MultimodalCheckRequest(
                modality="audio",
                claim_text="thanks everyone",
                media_ref="m",
                transcript_text="thanks everyone for joining",
            ),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )
        assert result.signal.modality == "audio"

    def test_scores_video_with_frame_drift(self):
        adapter = self._adapter()
        result = adapter.check(
            MultimodalCheckRequest(
                modality="video",
                claim_text="steady",
                media_ref="m",
                frame_similarities=[0.1, 0.05, 0.02],
            ),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )
        assert result.guard_decision.decision == "halt"

    def test_metadata_grounding(self):
        adapter = self._adapter()
        result = adapter.check(
            MultimodalCheckRequest(
                modality="image",
                claim_text="a red car",
                media_ref="m",
                image_bytes=b"bytes",
                metadata={"alt": "a blue bicycle"},
            ),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )
        assert "m#metadata:alt" in result.guard_decision.evidence_refs

    def test_empty_enabled_rejected(self):
        with pytest.raises(ValueError, match="at least one enabled modality"):
            build_hashbag_adapter(enabled_modalities=())
