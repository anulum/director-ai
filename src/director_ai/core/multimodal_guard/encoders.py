# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — image encoders

"""Image encoders used by :class:`MultimodalGuard`.

Two real backends:

* :class:`HashBagImageEncoder` — dependency-free FNV-1a hash-bag
  encoder. Splits the image byte payload into chunks, hashes each
  chunk, accumulates into a fixed-dim bag, and unit-norms.
  Deterministic, reproducible across runs, zero setup.
* :class:`TorchCLIPImageEncoder` — ``open_clip`` adapter loaded
  via :meth:`from_pretrained`. Uses the standard ``open_clip``
  preprocess pipeline, encodes with ``model.encode_image``, and
  unit-norms the result. Optional dependency; :class:`ImportError`
  carries install instructions.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

# FNV-1a parameters for 64-bit hashing of byte chunks.
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_UINT64_MASK = 0xFFFFFFFFFFFFFFFF


@runtime_checkable
class ImageEncoder(Protocol):
    """Anything that returns a fixed-dim, unit-norm embedding for
    an image byte payload."""

    dim: int

    def encode(self, image_bytes: bytes) -> tuple[float, ...]: ...


class HashBagImageEncoder:
    """FNV-1a hash-bag encoder for image bytes.

    Parameters
    ----------
    dim :
        Output embedding dimensionality. Default 512 — matches the
        common CLIP projection dim so swapping backends requires
        no downstream change.
    chunk :
        Byte-chunk size fed to FNV-1a. Default 64 — balances
        hash-bag sparsity against collision rate.
    """

    def __init__(self, *, dim: int = 512, chunk: int = 64) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim!r}")
        if chunk <= 0:
            raise ValueError(f"chunk must be positive; got {chunk!r}")
        self.dim = dim
        self._chunk = chunk

    def encode(self, image_bytes: bytes) -> tuple[float, ...]:
        if not image_bytes:
            raise ValueError("image_bytes must be non-empty")
        bag = [0.0] * self.dim
        for start in range(0, len(image_bytes), self._chunk):
            chunk = image_bytes[start : start + self._chunk]
            h = _fnv1a_64(chunk)
            bag[h % self.dim] += 1.0
        return _normalise(tuple(bag))


class TorchCLIPImageEncoder:
    """``open_clip`` image encoder. Lazily loaded.

    Construct via :meth:`from_pretrained` so the optional
    dependency check + model load happens in one place. Operators
    who already hold a loaded ``(model, preprocess)`` pair can
    instantiate directly.
    """

    def __init__(
        self,
        *,
        model: Any,
        preprocess: Any,
        dim: int,
        device: str = "cpu",
    ) -> None:
        if model is None:
            raise ValueError("model is required")
        if preprocess is None:
            raise ValueError("preprocess is required")
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim!r}")
        self._model = model
        self._preprocess = preprocess
        self.dim = dim
        self._device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        *,
        device: str = "cpu",
    ) -> TorchCLIPImageEncoder:
        """Load an ``open_clip`` model.

        Raises :class:`ImportError` with install instructions when
        ``open_clip_torch`` is not available.
        """
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "TorchCLIPImageEncoder.from_pretrained requires "
                "open_clip_torch and torch. Install with: "
                "pip install director-ai[multimodal]",
            ) from exc
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        dim = int(model.visual.output_dim)
        # ``preprocess`` closes over ``torch`` in open_clip — keep the
        # explicit import alive so static analysers do not trim it.
        _ = torch
        return cls(model=model, preprocess=preprocess, dim=dim, device=device)

    def encode(self, image_bytes: bytes) -> tuple[float, ...]:
        if not image_bytes:
            raise ValueError("image_bytes must be non-empty")
        try:
            import io

            import torch
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "TorchCLIPImageEncoder.encode requires Pillow and torch. "
                "Install with: pip install director-ai[multimodal]",
            ) from exc
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._preprocess(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            embedding = self._model.encode_image(tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return tuple(float(x) for x in embedding[0].cpu().tolist())


def _fnv1a_64(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & _UINT64_MASK
    return h


def _normalise(vec: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    inv = 1.0 / norm
    return tuple(x * inv for x in vec)
