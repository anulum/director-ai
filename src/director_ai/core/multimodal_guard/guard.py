# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — MultimodalGuard + TemporalConsistencyGuard

"""Orchestrators that bind an :class:`ImageEncoder` + a
:class:`CrossModalVerifier` into a verdict producer.

:class:`MultimodalGuard` classifies a single ``(image, text)``
claim; :class:`TemporalConsistencyGuard` folds a stream of
per-frame similarities through an EMA and raises when the
consistency drops below a caller-supplied floor — the behaviour
video / audio guardrails need to catch drift across frames.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .claim import MultimodalClaim
from .encoders import ImageEncoder
from .verifier import CrossModalVerifier

VerdictLabel = Literal["consistent", "uncertain", "hallucinated"]


@dataclass(frozen=True)
class MultimodalVerdict:
    """Result of one :meth:`MultimodalGuard.check` call."""

    label: VerdictLabel
    similarity: float
    reason: str


class MultimodalGuard:
    """Bind encoder + verifier + thresholds into a verdict producer.

    Parameters
    ----------
    encoder :
        Any :class:`ImageEncoder`.
    verifier :
        Any :class:`CrossModalVerifier`. Its ``dim`` must equal the
        encoder's ``dim`` — the constructor validates this.
    hallucination_threshold :
        Similarities strictly below this are labelled
        ``hallucinated``. Default 0.15.
    consistency_threshold :
        Similarities at or above this are labelled ``consistent``.
        Must be strictly greater than ``hallucination_threshold``;
        the band between is ``uncertain``. Default 0.45.
    """

    def __init__(
        self,
        *,
        encoder: ImageEncoder,
        verifier: CrossModalVerifier,
        hallucination_threshold: float = 0.15,
        consistency_threshold: float = 0.45,
    ) -> None:
        if encoder.dim != verifier.dim:
            raise ValueError(
                f"encoder.dim={encoder.dim} does not match verifier.dim={verifier.dim}"
            )
        if not 0.0 <= hallucination_threshold < consistency_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= hallucination < consistency <= 1; "
                f"got ({hallucination_threshold}, {consistency_threshold})"
            )
        self._encoder = encoder
        self._verifier = verifier
        self._hallucination = hallucination_threshold
        self._consistency = consistency_threshold

    def check(self, claim: MultimodalClaim) -> MultimodalVerdict:
        embedding = self._encoder.encode(claim.image_bytes)
        similarity = self._verifier.verify(embedding, claim.text_claim)
        return self._band(similarity)

    def check_many(
        self, claims: Iterable[MultimodalClaim]
    ) -> tuple[MultimodalVerdict, ...]:
        """Batch variant. Each claim is scored independently — the
        method is a convenience wrapper, not a performance
        optimisation. Torch-backed backends can shadow this
        method later to push batching into the tensor path."""
        return tuple(self.check(c) for c in claims)

    def _band(self, similarity: float) -> MultimodalVerdict:
        if similarity < self._hallucination:
            return MultimodalVerdict(
                label="hallucinated",
                similarity=similarity,
                reason=(
                    f"similarity {similarity:.3f} < hallucination_threshold "
                    f"({self._hallucination:.3f})"
                ),
            )
        if similarity >= self._consistency:
            return MultimodalVerdict(
                label="consistent",
                similarity=similarity,
                reason=(
                    f"similarity {similarity:.3f} >= consistency_threshold "
                    f"({self._consistency:.3f})"
                ),
            )
        return MultimodalVerdict(
            label="uncertain",
            similarity=similarity,
            reason=(
                f"{self._hallucination:.3f} <= similarity={similarity:.3f} "
                f"< {self._consistency:.3f}"
            ),
        )


@dataclass
class TemporalConsistencyGuard:
    """EMA-smoothed temporal consistency monitor for video / audio.

    Each frame's similarity is folded into an exponential moving
    average with decay ``alpha``. The guard raises as soon as the
    EMA crosses ``consistency_floor`` so a caller can halt a
    streaming generation mid-frame.

    Parameters
    ----------
    alpha :
        EMA decay, in ``(0, 1]``. Larger values weight recent
        frames more. Default 0.3.
    consistency_floor :
        EMA value that triggers the halt. Default 0.2.
    """

    alpha: float = 0.3
    consistency_floor: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {self.alpha!r}")
        if not 0.0 <= self.consistency_floor <= 1.0:
            raise ValueError(
                f"consistency_floor must be in [0, 1]; got {self.consistency_floor!r}"
            )
        self._ema: float | None = None

    @property
    def ema(self) -> float | None:
        return self._ema

    def update(self, similarity: float) -> bool:
        """Fold ``similarity`` into the EMA and return ``True`` if
        the caller should halt."""
        if not 0.0 <= similarity <= 1.0:
            raise ValueError(f"similarity must be in [0, 1]; got {similarity!r}")
        if self._ema is None:
            self._ema = similarity
        else:
            self._ema = self.alpha * similarity + (1.0 - self.alpha) * self._ema
        return self._ema < self.consistency_floor

    def reset(self) -> None:
        self._ema = None
