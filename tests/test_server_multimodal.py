# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — /v1/multimodal/check endpoint tests

"""Tests for the opt-in multi-modal hallucination endpoint.

Covers the default-off posture (refused without experimental hooks or
configured modalities), image/audio/video decisions over the dependency-free
hash-bag backend, the unbenchmarked-warn band, video frame-drift halt,
modality and base64 validation, the tenant-safe response shape, and the
inert config defaults."""

from __future__ import annotations

import base64

import pytest

from director_ai.core.config import DirectorConfig

try:
    from fastapi.testclient import TestClient

    from director_ai.server import create_app

    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

from director_ai.experimental import (
    disable_experimental_hooks,
    enable_experimental_hooks,
)

pytestmark = pytest.mark.skipif(not _SERVER_AVAILABLE, reason="fastapi not installed")

_IMAGE_B64 = base64.b64encode(b"fake-image-payload-bytes-for-hashbag").decode()
_ALL = ("image", "audio", "video")


@pytest.fixture
def experimental_on():
    enable_experimental_hooks()
    try:
        yield
    finally:
        disable_experimental_hooks()


def _app(*, enabled=_ALL, benchmarked=_ALL):
    cfg = DirectorConfig(
        api_keys=[],
        llm_provider="mock",
        multimodal_enabled_modalities=tuple(enabled),
        multimodal_benchmarked_modalities=tuple(benchmarked),
    )
    return create_app(cfg)


class TestDisabledPosture:
    def test_refused_without_experimental_hooks(self):
        disable_experimental_hooks()
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "image",
                    "claim_text": "a cat",
                    "media_ref": "m1",
                    "image_base64": _IMAGE_B64,
                },
            )
        assert r.status_code == 404

    def test_refused_when_no_modalities_configured(self, experimental_on):
        with TestClient(_app(enabled=(), benchmarked=())) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "image",
                    "claim_text": "a cat",
                    "media_ref": "m1",
                    "image_base64": _IMAGE_B64,
                },
            )
        assert r.status_code == 404


class TestEnabledDecisions:
    def test_image_returns_guard_decision(self, experimental_on):
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "image",
                    "claim_text": "a photo of a mountain lake",
                    "media_ref": "img-1",
                    "image_base64": _IMAGE_B64,
                },
            )
        assert r.status_code == 200
        body = r.json()
        assert body["modality"] == "image"
        assert body["media_ref"] == "img-1"
        assert body["guard_decision"]["decision"] in {"allow", "warn", "halt"}

    def test_unbenchmarked_modality_warns(self, experimental_on):
        with TestClient(_app(enabled=("image",), benchmarked=())) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "image",
                    "claim_text": "a cat",
                    "media_ref": "img-2",
                    "image_base64": _IMAGE_B64,
                },
            )
        assert r.status_code == 200
        decision = r.json()["guard_decision"]
        assert decision["decision"] == "warn"
        assert decision["reason"] == "multimodal_unbenchmarked"

    def test_audio_returns_decision(self, experimental_on):
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "audio",
                    "claim_text": "the speaker thanks the audience",
                    "media_ref": "aud-1",
                    "transcript_text": "thank you all for coming today",
                },
            )
        assert r.status_code == 200
        assert r.json()["modality"] == "audio"

    def test_video_frame_drift_halts(self, experimental_on):
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "video",
                    "claim_text": "a steady scene",
                    "media_ref": "vid-1",
                    "frame_similarities": [0.1, 0.05, 0.02, 0.01],
                },
            )
        assert r.status_code == 200
        decision = r.json()["guard_decision"]
        assert decision["decision"] == "halt"
        assert decision["reason"] == "multimodal_hallucinated"


class TestValidation:
    def test_invalid_modality_rejected(self, experimental_on):
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={"modality": "text", "claim_text": "x", "media_ref": "m"},
            )
        assert r.status_code == 422

    def test_invalid_base64_rejected(self, experimental_on):
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "image",
                    "claim_text": "x",
                    "media_ref": "m",
                    "image_base64": "not!valid!base64!",
                },
            )
        assert r.status_code == 400

    def test_video_requires_frame_similarities(self, experimental_on):
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "video",
                    "claim_text": "x",
                    "media_ref": "m",
                },
            )
        assert r.status_code == 400


class TestTenantSafety:
    def test_response_omits_raw_claim_and_media(self, experimental_on):
        with TestClient(_app()) as client:
            r = client.post(
                "/v1/multimodal/check",
                json={
                    "modality": "audio",
                    "claim_text": "SECRET-CLAIM-TOKEN",
                    "media_ref": "aud-2",
                    "transcript_text": "SECRET-TRANSCRIPT-TOKEN",
                },
            )
        raw = r.text
        assert "SECRET-CLAIM-TOKEN" not in raw
        assert "SECRET-TRANSCRIPT-TOKEN" not in raw


class TestConfigDefaults:
    def test_multimodal_disabled_by_default(self):
        cfg = DirectorConfig()
        assert cfg.multimodal_enabled_modalities == ()
        assert cfg.multimodal_benchmarked_modalities == ()
