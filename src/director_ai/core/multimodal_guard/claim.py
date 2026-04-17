# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MultimodalClaim

"""One (image, text) pair to check for hallucination."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MultimodalClaim:
    """Image bytes + the text claim about them.

    ``image_bytes`` is the raw image payload. The
    :class:`ImageEncoder` Protocol owns the decoding —
    :class:`HashBagImageEncoder` treats the bytes as opaque for
    determinism; :class:`TorchCLIPImageEncoder` decodes JPEG/PNG
    via Pillow before running the CLIP pipeline.
    ``text_claim`` is the claim that needs to be consistent with
    the image.
    """

    image_bytes: bytes
    text_claim: str

    def __post_init__(self) -> None:
        if not self.image_bytes:
            raise ValueError("MultimodalClaim.image_bytes must be non-empty")
        if not self.text_claim or not self.text_claim.strip():
            raise ValueError("MultimodalClaim.text_claim must be non-empty")
