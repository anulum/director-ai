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

import sys
import types

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


# -- CLIP adapter wiring (injected loader; no open_clip/torch/model download) -


class TestBuildClipAdapter:
    """build_clip_adapter wires real CLIP image encoder/verifier through one
    shared model; the loader is injected so the wiring is verified offline."""

    @staticmethod
    def _stub_loader_factory():
        calls = []

        def loader(model_name, pretrained, device):
            calls.append((model_name, pretrained, device))
            # (model, preprocess, tokenizer, dim) — opaque stubs are enough for
            # construction; encode/verify (which need torch) are not exercised.
            return object(), object(), object(), 16

        return loader, calls

    def test_loader_called_with_configured_model_and_returns_adapter(self):
        from director_ai.core.multimodal_guard import (
            MultimodalVerifierAdapter,
            build_clip_adapter,
        )

        loader, calls = self._stub_loader_factory()
        adapter = build_clip_adapter(
            enabled_modalities=("image", "audio"),
            benchmarked_modalities=("image",),
            model_name="ViT-L-14",
            pretrained="laion2b_s32b_b82k",
            device="cpu",
            loader=loader,
        )
        assert isinstance(adapter, MultimodalVerifierAdapter)
        assert calls == [("ViT-L-14", "laion2b_s32b_b82k", "cpu")]

    def test_image_path_uses_clip_encoder_and_verifier(self):
        from director_ai.core.multimodal_guard import build_clip_adapter
        from director_ai.core.multimodal_guard.encoders import TorchCLIPImageEncoder
        from director_ai.core.multimodal_guard.verifier import (
            TorchCLIPCrossModalVerifier,
        )

        loader, _calls = self._stub_loader_factory()
        adapter = build_clip_adapter(enabled_modalities=("image",), loader=loader)
        guard = adapter._image_guard  # noqa: SLF001 - wiring assertion
        assert isinstance(guard._encoder, TorchCLIPImageEncoder)  # noqa: SLF001
        assert isinstance(guard._verifier, TorchCLIPCrossModalVerifier)  # noqa: SLF001

    def test_textual_modalities_use_hashbag_similarity(self):
        from director_ai.core.multimodal_guard import build_clip_adapter

        loader, _calls = self._stub_loader_factory()
        adapter = build_clip_adapter(
            enabled_modalities=("audio",),
            loader=loader,
            text_dim=32,
        )

        result = adapter.check(
            MultimodalCheckRequest(
                modality="audio",
                claim_text="policy refund window",
                media_ref="audio-1",
                transcript_text="policy refund window confirmed",
                caption_text="policy refund window visible",
                metadata={"title": "policy refund window"},
            ),
            risk_envelope=_ENVELOPE,
            policy_id="p",
        )

        assert result.signal.modality == "audio"
        assert result.signal.score < 1.0
        assert "audio-1#caption" in result.signal.evidence_refs
        assert "audio-1#metadata:title" in result.signal.evidence_refs

    def test_default_loader_without_open_clip_raises_install_hint(self):
        # open_clip is not a core dependency; the default loader must point the
        # operator at the [multimodal] extra rather than fail obscurely.
        import importlib.util

        from director_ai.core.multimodal_guard.factory import _default_clip_loader

        if importlib.util.find_spec("open_clip") is not None:
            pytest.skip("open_clip is installed; the ImportError path is not taken")
        with pytest.raises(ImportError, match=r"director-ai\[multimodal\]"):
            _default_clip_loader("ViT-B-32", "openai", "cpu")

    def test_default_loader_uses_open_clip_quickgelu_and_tokenizer(self, monkeypatch):
        from director_ai.core.multimodal_guard.factory import _default_clip_loader

        calls: dict[str, object] = {}

        class FakeModel:
            def __init__(self) -> None:
                self.visual = types.SimpleNamespace(output_dim=128)
                self.device = ""
                self.eval_called = False

            def to(self, device: str) -> FakeModel:
                self.device = device
                return self

            def eval(self) -> FakeModel:
                self.eval_called = True
                return self

        fake_model = FakeModel()

        def fake_create_model_and_transforms(
            model_name: str, *, pretrained: str, force_quick_gelu: bool
        ):
            calls["model_name"] = model_name
            calls["pretrained"] = pretrained
            calls["force_quick_gelu"] = force_quick_gelu
            return fake_model, object(), "preprocess"

        fake_open_clip = types.SimpleNamespace(
            create_model_and_transforms=fake_create_model_and_transforms,
            get_tokenizer=lambda model_name: f"tokenizer:{model_name}",
        )
        monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

        model, preprocess, tokenizer, dim = _default_clip_loader(
            "ViT-B-32", "openai", "cuda:0"
        )

        assert model is fake_model
        assert preprocess == "preprocess"
        assert tokenizer == "tokenizer:ViT-B-32"
        assert dim == 128
        assert fake_model.device == "cuda:0"
        assert fake_model.eval_called is True
        assert calls == {
            "model_name": "ViT-B-32",
            "pretrained": "openai",
            "force_quick_gelu": True,
        }
