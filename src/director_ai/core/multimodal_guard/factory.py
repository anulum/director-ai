# SPDX-License-Identifier: AGPL-3.0-or-later
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

from collections.abc import Mapping, Sequence

from .adapter import MultimodalVerifierAdapter
from .encoders import HashBagImageEncoder
from .guard import MultimodalGuard
from .verifier import HashBagCrossModalVerifier, text_bag_similarity

__all__ = ["build_hashbag_adapter"]


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
