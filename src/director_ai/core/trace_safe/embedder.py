# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TraceEmbedder abstractions

"""Text-to-vector embedders for the trace-safe oracle.

The oracle only cares that ``embed(text)`` returns a fixed-length
unit-norm vector. Two concrete backends ship here:

* :class:`HashBagEmbedder` — zero-dependency hashing-trick
  embedder. Stable across machines (seeded FNV-1a hash, no Python
  hash randomisation), produces a feature-hashed bag-of-tokens
  with optional trigram expansion. Good enough for the foundation
  and for deployments where adding sentence-transformers is not
  acceptable.
* A real sentence-transformer backend can be plugged in by
  satisfying the :class:`TraceEmbedder` protocol — deliberately
  kept out of this package to avoid pulling ``torch`` as a
  transitive default.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_WORD_RE = re.compile(r"[a-z0-9]+")


class TraceEmbedder(ABC):
    """Protocol — ``embed(text)`` returns a tuple of floats whose
    L2 norm is 1."""

    dim: int

    @abstractmethod
    def embed(self, text: str) -> tuple[float, ...]: ...  # pragma: no cover


class HashBagEmbedder(TraceEmbedder):
    """Hashing-trick bag-of-tokens embedder.

    Parameters
    ----------
    dim :
        Output vector dimension. Larger values reduce collision
        rate at the cost of memory.
    ngram :
        Include ``1..ngram``-grams in the bag. ``1`` is plain bag
        of tokens; ``2`` adds bigrams; etc.
    lowercase :
        When ``True`` (default) the text is lowercased before
        tokenisation.
    """

    def __init__(
        self,
        *,
        dim: int = 256,
        ngram: int = 2,
        lowercase: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive; got {dim}")
        if ngram <= 0:
            raise ValueError(f"ngram must be positive; got {ngram}")
        self.dim = dim
        self._ngram = ngram
        self._lowercase = lowercase

    def embed(self, text: str) -> tuple[float, ...]:
        tokens = self._tokenise(text)
        if not tokens:
            return tuple([0.0] * self.dim)
        counts = [0.0] * self.dim
        for t in tokens:
            counts[_bucket(t, self.dim)] += 1.0
        for n in range(2, self._ngram + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i : i + n])
                counts[_bucket(gram, self.dim)] += 1.0
        norm = math.sqrt(sum(x * x for x in counts))
        if norm == 0.0:
            return tuple(counts)
        inv = 1.0 / norm
        return tuple(x * inv for x in counts)

    def _tokenise(self, text: str) -> list[str]:
        source = text.lower() if self._lowercase else text
        return _WORD_RE.findall(source)


def _bucket(token: str, dim: int) -> int:
    """Stable FNV-1a bucket. Avoids ``hash()`` whose per-process
    salt makes embeddings non-reproducible across restarts."""
    h = _FNV_OFFSET
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h % dim
