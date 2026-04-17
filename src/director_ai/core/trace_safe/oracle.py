# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — TraceSafe centroid oracle

"""Centroid-distance classifier over embedded traces.

The oracle keeps two centroids — safe and unsafe — computed as the
mean of the labelled exemplars in the corpus. A new trace is
embedded and compared with both centroids via cosine similarity;
the verdict falls in one of three bands:

* ``safe`` — cosine(trace, safe) − cosine(trace, unsafe) ≥
  ``decision_margin``.
* ``unsafe`` — cosine(trace, unsafe) − cosine(trace, safe) ≥
  ``decision_margin``.
* ``uncertain`` — otherwise. The caller should escalate (full NLI
  scoring, human review, or an LLM judge).

The margin is a tuning knob: low margin = more decisive verdicts at
the cost of false positives / negatives; high margin = more
``uncertain`` verdicts that defer to the caller. The default is
0.05 — small enough to be actionable, large enough to avoid
flipping on noise.

Foundation scope: centroid classifier only. Spectral clustering
for multi-modal unsafe classes (``attack`` vs ``drift`` vs
``policy-violation``) is a follow-up — the centroid approach
misclassifies multi-modal unsafe clusters when the centroid lands
between modes. The public API already returns the nearest exemplar
label so operators can begin annotating clusters manually.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .embedder import TraceEmbedder

TraceLabel = Literal["safe", "unsafe", "uncertain"]


@dataclass(frozen=True)
class TraceSample:
    """One labelled exemplar in the corpus."""

    text: str
    label: TraceLabel
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceVerdict:
    """Classification of one trace."""

    label: TraceLabel
    safe_similarity: float
    unsafe_similarity: float
    margin: float
    nearest_exemplar_label: TraceLabel
    nearest_exemplar_similarity: float
    reason: str


class TraceSafeOracle:
    """Centroid-distance safety classifier.

    Parameters
    ----------
    embedder :
        Any :class:`TraceEmbedder`.
    samples :
        Labelled corpus. The oracle needs at least one ``safe`` and
        one ``unsafe`` exemplar; otherwise :meth:`classify` returns
        ``uncertain`` for every input.
    decision_margin :
        Minimum cosine-similarity gap required for a decisive
        verdict.
    """

    def __init__(
        self,
        *,
        embedder: TraceEmbedder,
        samples: Iterable[TraceSample] = (),
        decision_margin: float = 0.05,
    ) -> None:
        if decision_margin < 0.0:
            raise ValueError(
                f"decision_margin must be non-negative; got {decision_margin}"
            )
        self._embedder = embedder
        self._decision_margin = decision_margin
        self._samples: list[TraceSample] = []
        self._sample_vectors: list[tuple[float, ...]] = []
        self._safe_centroid: tuple[float, ...] | None = None
        self._unsafe_centroid: tuple[float, ...] | None = None
        for s in samples:
            self.add_sample(s)

    def add_sample(self, sample: TraceSample) -> None:
        """Append one exemplar and recompute the centroids."""
        if sample.label not in ("safe", "unsafe"):
            raise ValueError(
                f"sample.label must be 'safe' or 'unsafe'; got {sample.label!r}"
            )
        self._samples.append(sample)
        self._sample_vectors.append(self._embedder.embed(sample.text))
        self._rebuild_centroids()

    def _rebuild_centroids(self) -> None:
        pairs = zip(self._samples, self._sample_vectors, strict=True)
        safe: list[tuple[float, ...]] = []
        unsafe: list[tuple[float, ...]] = []
        for s, v in pairs:
            if s.label == "safe":
                safe.append(v)
            elif s.label == "unsafe":
                unsafe.append(v)
        self._safe_centroid = _normalise(_mean(safe)) if safe else None
        self._unsafe_centroid = _normalise(_mean(unsafe)) if unsafe else None

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def classify(self, trace: str) -> TraceVerdict:
        """Classify one trace snapshot."""
        trace_vec = self._embedder.embed(trace)
        safe_sim = (
            _cosine(trace_vec, self._safe_centroid)
            if self._safe_centroid is not None
            else 0.0
        )
        unsafe_sim = (
            _cosine(trace_vec, self._unsafe_centroid)
            if self._unsafe_centroid is not None
            else 0.0
        )
        nearest_label: TraceLabel = "uncertain"
        nearest_sim = -1.0
        for s, v in zip(self._samples, self._sample_vectors, strict=True):
            sim = _cosine(trace_vec, v)
            if sim > nearest_sim:
                nearest_sim = sim
                nearest_label = s.label
        margin = safe_sim - unsafe_sim
        if self._safe_centroid is None or self._unsafe_centroid is None:
            return TraceVerdict(
                label="uncertain",
                safe_similarity=safe_sim,
                unsafe_similarity=unsafe_sim,
                margin=margin,
                nearest_exemplar_label=nearest_label,
                nearest_exemplar_similarity=nearest_sim,
                reason="corpus missing safe or unsafe exemplar",
            )
        if margin >= self._decision_margin:
            return TraceVerdict(
                label="safe",
                safe_similarity=safe_sim,
                unsafe_similarity=unsafe_sim,
                margin=margin,
                nearest_exemplar_label=nearest_label,
                nearest_exemplar_similarity=nearest_sim,
                reason=(
                    f"margin {margin:.3f} >= decision_margin "
                    f"({self._decision_margin:.3f})"
                ),
            )
        if -margin >= self._decision_margin:
            return TraceVerdict(
                label="unsafe",
                safe_similarity=safe_sim,
                unsafe_similarity=unsafe_sim,
                margin=margin,
                nearest_exemplar_label=nearest_label,
                nearest_exemplar_similarity=nearest_sim,
                reason=(
                    f"margin {margin:.3f} <= -decision_margin "
                    f"({self._decision_margin:.3f})"
                ),
            )
        return TraceVerdict(
            label="uncertain",
            safe_similarity=safe_sim,
            unsafe_similarity=unsafe_sim,
            margin=margin,
            nearest_exemplar_label=nearest_label,
            nearest_exemplar_similarity=nearest_sim,
            reason=(
                f"|margin|={abs(margin):.3f} < decision_margin "
                f"({self._decision_margin:.3f})"
            ),
        )


def _mean(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not vectors:
        return ()
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            acc[i] += v[i]
    n = float(len(vectors))
    return tuple(x / n for x in acc)


def _normalise(vec: tuple[float, ...]) -> tuple[float, ...]:
    if not vec:
        return ()
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    inv = 1.0 / norm
    return tuple(x * inv for x in vec)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))
