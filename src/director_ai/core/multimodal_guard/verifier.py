# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — cross-modal verifiers

"""Score an ``(image embedding, text)`` pair in ``[0, 1]``.

Two real backends:

* :class:`HashBagCrossModalVerifier` — dependency-free.
  Tokenises the text, runs the same FNV-1a hash-bag family as
  :class:`HashBagImageEncoder`, and returns cosine similarity.
* :class:`TorchCLIPCrossModalVerifier` — ``open_clip`` text
  encoder adapter. Shares the model with
  :class:`TorchCLIPImageEncoder` so loading cost amortises across
  encoder + verifier.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .encoders import _fnv1a_64, _normalise


@runtime_checkable
class CrossModalVerifier(Protocol):
    """Scores ``(image_embedding, text)`` pairs."""

    dim: int

    def verify(
        self,
        image_embedding: tuple[float, ...],
        text: str,
    ) -> float: ...


class HashBagCrossModalVerifier:
    """Cosine similarity between an image hash-bag and a text
    hash-bag produced by the same FNV-1a family.

    Parameters
    ----------
    dim :
        Embedding dimensionality. Must match the image encoder.
        Default 512.
    lowercase :
        Lowercase the text before tokenisation. Default ``True``
        so the verifier is case-insensitive by default.
    """

    def __init__(self, *, dim: int = 512, lowercase: bool = True) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim!r}")
        self.dim = dim
        self._lowercase = lowercase

    def verify(
        self,
        image_embedding: tuple[float, ...],
        text: str,
    ) -> float:
        if len(image_embedding) != self.dim:
            raise ValueError(
                f"image embedding dim {len(image_embedding)} != verifier dim {self.dim}"
            )
        if not text or not text.strip():
            return 0.0
        text_vec = self._embed_text(text)
        sim = _cosine(image_embedding, text_vec)
        # Cosine on non-negative hash-bag vectors is already in [0, 1];
        # keep the clamp so callers can swap in a signed encoder.
        return max(0.0, min(1.0, sim))

    def _embed_text(self, text: str) -> tuple[float, ...]:
        normalised = text.lower() if self._lowercase else text
        bag = [0.0] * self.dim
        for token in normalised.split():
            h = _fnv1a_64(token.encode("utf-8"))
            bag[h % self.dim] += 1.0
        return _normalise(tuple(bag))


class TorchCLIPCrossModalVerifier:
    """``open_clip`` text encoder + cosine against an image
    embedding.

    Construct via :meth:`from_pretrained` or pass a loaded
    ``(model, tokenizer)`` pair directly — the typical pattern is
    to share the model with :class:`TorchCLIPImageEncoder`.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        dim: int,
        device: str = "cpu",
    ) -> None:
        if model is None:
            raise ValueError("model is required")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim!r}")
        self._model = model
        self._tokenizer = tokenizer
        self.dim = dim
        self._device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        device: str = "cpu",
    ) -> TorchCLIPCrossModalVerifier:
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "TorchCLIPCrossModalVerifier.from_pretrained requires "
                "open_clip_torch. Install with: pip install director-ai[multimodal]",
            ) from exc
        model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        dim = int(model.visual.output_dim)
        return cls(model=model, tokenizer=tokenizer, dim=dim, device=device)

    def verify(
        self,
        image_embedding: tuple[float, ...],
        text: str,
    ) -> float:
        if len(image_embedding) != self.dim:
            raise ValueError(
                f"image embedding dim {len(image_embedding)} != verifier dim {self.dim}"
            )
        if not text or not text.strip():
            return 0.0
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "TorchCLIPCrossModalVerifier.verify requires torch. "
                "Install with: pip install director-ai[multimodal]",
            ) from exc
        tokens = self._tokenizer([text]).to(self._device)
        with torch.no_grad():
            text_embedding = self._model.encode_text(tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        text_vec = tuple(float(x) for x in text_embedding[0].cpu().tolist())
        sim = _cosine(image_embedding, text_vec)
        # CLIP cosine is in [-1, 1]; rescale to [0, 1] so the band
        # thresholds in :class:`MultimodalGuard` work uniformly
        # across backends.
        return (sim + 1.0) / 2.0


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return max(-1.0, min(1.0, dot))
