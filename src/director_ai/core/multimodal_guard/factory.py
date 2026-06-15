# SPDX-License-Identifier: BUSL-1.1
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — dependency-free multimodal adapter factory

"""Build a fully dependency-free :class:`MultimodalVerifierAdapter`.

The hash-bag backends need no ML stack, so the server can stand up a live
multimodal guard from configuration alone. Image checks run through a
:class:`MultimodalGuard` on the FNV hash-bag encoder/verifier; audio,
caption, and metadata grounding reuse :func:`text_bag_similarity`; video
consistency uses the per-frame similarities supplied on each request. Swap
in CLIP/SigLIP backends by constructing :class:`MultimodalVerifierAdapter`
directly.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .adapter import MultimodalVerifierAdapter
from .encoders import HashBagImageEncoder, TorchCLIPImageEncoder
from .guard import MultimodalGuard
from .verifier import (
    HashBagCrossModalVerifier,
    TorchCLIPCrossModalVerifier,
    text_bag_similarity,
)

__all__ = ["build_clip_adapter", "build_hashbag_adapter"]

# A CLIP loader returns ``(model, preprocess, tokenizer, dim)`` — injected in
# tests so the wiring is verifiable without open_clip, torch, or a model download.
ClipLoader = Callable[[str, str, str], tuple[Any, Any, Any, int]]


def build_hashbag_adapter(
    *,
    enabled_modalities: Sequence[str],
    benchmarked_modalities: Sequence[str] = (),
    dim: int = 512,
    hallucination_threshold: float = 0.15,
    consistency_threshold: float = 0.45,
    temporal_alpha: float = 0.5,
    temporal_floor: float = 0.2,
    grounding_floor: float = 0.4,
    grounding_allow_threshold: float = 0.75,
) -> MultimodalVerifierAdapter:
    """Return a hash-bag-backed adapter for the configured modalities.

    Every enabled modality is functional without any optional dependency:
    image via the hash-bag guard, audio/caption/metadata via
    :func:`text_bag_similarity`, and video via per-request frame
    similarities.
    """
    image_guard = MultimodalGuard(
        encoder=HashBagImageEncoder(dim=dim),
        verifier=HashBagCrossModalVerifier(dim=dim),
        hallucination_threshold=hallucination_threshold,
        consistency_threshold=consistency_threshold,
    )

    def _text_score(reference: str, claim: str) -> float:
        return text_bag_similarity(reference, claim, dim=dim)

    def _metadata_score(metadata: Mapping[str, str], claim: str) -> float:
        joined = " ".join(str(value) for value in metadata.values())
        return text_bag_similarity(joined, claim, dim=dim)

    return MultimodalVerifierAdapter(
        image_guard=image_guard,
        audio_score_fn=_text_score,
        caption_score_fn=_text_score,
        metadata_score_fn=_metadata_score,
        enabled_modalities=enabled_modalities,
        benchmarked_modalities=benchmarked_modalities,
        temporal_alpha=temporal_alpha,
        temporal_floor=temporal_floor,
        grounding_floor=grounding_floor,
        grounding_allow_threshold=grounding_allow_threshold,
    )


def _default_clip_loader(model_name: str, pretrained: str, device: str):
    """Load an ``open_clip`` model once for both the encoder and the verifier."""
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        raise ImportError(
            "build_clip_adapter requires open_clip_torch + torch + Pillow. "
            "Install with: pip install director-ai[multimodal]",
        ) from exc
    # The original OpenAI CLIP weights were trained with QuickGELU; open_clip's
    # default activation differs, which it warns about and which subtly degrades
    # the embeddings. Force QuickGELU for the OpenAI tag so the model matches how
    # it was trained.
    force_quick_gelu = pretrained == "openai" and "quickgelu" not in model_name.lower()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, force_quick_gelu=force_quick_gelu
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    dim = int(model.visual.output_dim)
    return model, preprocess, tokenizer, dim


def build_clip_adapter(
    *,
    enabled_modalities: Sequence[str],
    benchmarked_modalities: Sequence[str] = (),
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: str = "cpu",
    loader: ClipLoader | None = None,
    text_dim: int = 512,
    hallucination_threshold: float = 0.15,
    consistency_threshold: float = 0.45,
    temporal_alpha: float = 0.5,
    temporal_floor: float = 0.2,
    grounding_floor: float = 0.4,
    grounding_allow_threshold: float = 0.75,
) -> MultimodalVerifierAdapter:
    """Return a CLIP-backed adapter: real semantic vision for the image modality.

    One ``open_clip`` model is loaded and shared between the image encoder and the
    cross-modal verifier, so image grounding becomes semantic (CLIP cosine of the
    image against the claim) instead of the FNV byte hash-bag baseline. Audio,
    caption, and metadata grounding stay lexical (``text_bag_similarity``) — CLIP
    is an image↔text model, so text↔text scoring is out of its scope and left to a
    dedicated text encoder. ``loader`` is injectable for testing the wiring without
    open_clip / torch / a model download.
    """
    load = loader or _default_clip_loader
    model, preprocess, tokenizer, dim = load(model_name, pretrained, device)

    image_guard = MultimodalGuard(
        encoder=TorchCLIPImageEncoder(
            model=model, preprocess=preprocess, dim=dim, device=device
        ),
        verifier=TorchCLIPCrossModalVerifier(
            model=model, tokenizer=tokenizer, dim=dim, device=device
        ),
        hallucination_threshold=hallucination_threshold,
        consistency_threshold=consistency_threshold,
    )

    def _text_score(reference: str, claim: str) -> float:
        return text_bag_similarity(reference, claim, dim=text_dim)

    def _metadata_score(metadata: Mapping[str, str], claim: str) -> float:
        joined = " ".join(str(value) for value in metadata.values())
        return text_bag_similarity(joined, claim, dim=text_dim)

    return MultimodalVerifierAdapter(
        image_guard=image_guard,
        audio_score_fn=_text_score,
        caption_score_fn=_text_score,
        metadata_score_fn=_metadata_score,
        enabled_modalities=enabled_modalities,
        benchmarked_modalities=benchmarked_modalities,
        temporal_alpha=temporal_alpha,
        temporal_floor=temporal_floor,
        grounding_floor=grounding_floor,
        grounding_allow_threshold=grounding_allow_threshold,
    )
