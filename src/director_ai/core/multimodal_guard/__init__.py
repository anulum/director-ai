# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — multimodal hallucination guard

"""Detect text claims that do not match a paired image.

Two Protocol boundaries separate the orchestrator from the
model backends:

* :class:`ImageEncoder` — turns image bytes into a unit-norm
  embedding.
* :class:`CrossModalVerifier` — scores an ``(image embedding,
  text)`` pair in ``[0, 1]``.

Two backends ship:

* :class:`HashBagImageEncoder` / :class:`HashBagCrossModalVerifier`
  — pure-Python, dependency-free. FNV-1a hash-bag encoders that
  produce deterministic unit-norm embeddings from image bytes and
  from tokenised text. Fast, reproducible, and zero-setup — they
  are the right choice for CI, air-gapped deployments, and
  bootstrap-before-a-model-is-available scenarios.
* :class:`TorchCLIPImageEncoder` / :class:`TorchCLIPCrossModalVerifier`
  — real CLIP / SigLIP adapters via ``open_clip``. Loaded lazily
  through :meth:`TorchCLIPImageEncoder.from_pretrained` so the
  hash-bag backends work without any ML stack installed. Raise
  :class:`ImportError` with install instructions when the
  optional dependency is missing.

The :class:`MultimodalGuard` orchestrator picks a verdict band
(``consistent`` / ``uncertain`` / ``hallucinated``) based on the
cross-modal similarity against two caller-configurable thresholds.
Temporal aggregation for video / audio streams ships as
:class:`TemporalConsistencyGuard`.
"""

from .claim import MultimodalClaim
from .encoders import HashBagImageEncoder, ImageEncoder, TorchCLIPImageEncoder
from .guard import MultimodalGuard, MultimodalVerdict, TemporalConsistencyGuard
from .verifier import (
    CrossModalVerifier,
    HashBagCrossModalVerifier,
    TorchCLIPCrossModalVerifier,
)

__all__ = [
    "CrossModalVerifier",
    "HashBagCrossModalVerifier",
    "HashBagImageEncoder",
    "ImageEncoder",
    "MultimodalClaim",
    "MultimodalGuard",
    "MultimodalVerdict",
    "TemporalConsistencyGuard",
    "TorchCLIPCrossModalVerifier",
    "TorchCLIPImageEncoder",
]
