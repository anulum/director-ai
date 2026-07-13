# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Self-consistency / semantic-entropy signal

"""Sample-consistency uncertainty signal (WCA-3).

SelfCheckGPT-style intuition (Manakul et al. 2023): when a model
confabulates, independent samples for the same prompt disagree;
grounded answers are stable across samples. Semantic entropy (Farquhar
et al., Nature 2024) formalises the disagreement: cluster samples by
bidirectional entailment and measure entropy over the cluster
distribution — high entropy means the model has no stable answer.

Director-AI is a scorer, not the generator, so this module consumes
**pre-sampled** alternative responses supplied by the caller (a proxy
fanning out ``n>1`` completions, an agent re-querying its model, or an
offline dataset such as wiki_bio_gpt3_hallucination that ships
samples). It computes:

- ``semantic_entropy`` — discrete entropy over entailment clusters of
  ``[primary] + samples``, normalised to [0, 1] by ``log(n)`` (the
  maximum over ``n`` singleton clusters);
- ``primary_agreement`` — fraction of all texts landing in the
  primary response's cluster;
- ``consistency_score`` — ``1 − semantic_entropy`` blended with
  ``primary_agreement`` (equal weights): high when samples agree
  *and* the primary sits in the consensus.

Entailment backend: the shipped NLI scorer when supplied (production
path), otherwise a dependency-free lexical-overlap fallback — the same
graceful-floor discipline as the rest of the scoring stack. The
fallback is a weaker clustering signal and is labelled in the result
(``entailment_backend``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ConsistencyResult",
    "SelfConsistencyScorer",
]

#: Bidirectional NLI divergence at or below which two texts count as
#: mutually entailing (same semantic cluster). Matches the
#: VerifiedScorer support threshold philosophy: conservative, so
#: clusters split rather than merge on doubt.
_NLI_ENTAIL_DIVERGENCE = 0.35

#: Jaccard token overlap at or above which the lexical fallback treats
#: two texts as equivalent.
_LEXICAL_ENTAIL_OVERLAP = 0.6


@dataclass
class ConsistencyResult:
    """Semantic-entropy verdict over one primary response and its samples."""

    semantic_entropy: float
    primary_agreement: float
    consistency_score: float
    n_samples: int
    n_clusters: int
    entailment_backend: str  # "nli" | "lexical"
    clusters: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the signal as a JSON-ready mapping."""
        return {
            "semantic_entropy": round(self.semantic_entropy, 4),
            "primary_agreement": round(self.primary_agreement, 4),
            "consistency_score": round(self.consistency_score, 4),
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "entailment_backend": self.entailment_backend,
            "clusters": self.clusters,
        }


class SelfConsistencyScorer:
    """Cluster sampled responses and score the primary's consensus.

    Parameters
    ----------
    nli_scorer : object or None
        Shipped NLI scorer exposing ``score(premise, hypothesis) ->
        float`` divergence in [0, 1] (0 = entailment). None selects
        the lexical fallback.
    entail_divergence : float
        Bidirectional divergence threshold for the NLI backend.
    lexical_overlap : float
        Jaccard threshold for the fallback backend.
    """

    def __init__(
        self,
        nli_scorer: Any = None,
        entail_divergence: float = _NLI_ENTAIL_DIVERGENCE,
        lexical_overlap: float = _LEXICAL_ENTAIL_OVERLAP,
    ) -> None:
        if not 0.0 < entail_divergence < 1.0:
            raise ValueError("entail_divergence must be in (0, 1)")
        if not 0.0 < lexical_overlap < 1.0:
            raise ValueError("lexical_overlap must be in (0, 1)")
        if nli_scorer is not None and not callable(
            getattr(nli_scorer, "score", None),
        ):
            raise ValueError(
                "nli_scorer must provide a callable score method",
            )
        self._nli = nli_scorer
        self._entail_divergence = entail_divergence
        self._lexical_overlap = lexical_overlap

    @property
    def entailment_backend(self) -> str:
        """Active equivalence backend: ``nli`` or ``lexical``."""
        return "nli" if self._nli is not None else "lexical"

    def score(self, primary: str, samples: list[str]) -> ConsistencyResult:
        """Score ``primary`` against alternative ``samples``.

        Raises ``ValueError`` on empty/blank inputs — an empty sample
        set has no consistency evidence and callers must not mistake
        that for confidence.
        """
        if not isinstance(primary, str) or not primary.strip():
            raise ValueError("primary must be a non-empty string")
        if not isinstance(samples, list) or not samples:
            raise ValueError("samples must be a non-empty list of strings")
        if any(not isinstance(s, str) or not s.strip() for s in samples):
            raise ValueError("samples must contain only non-empty strings")

        texts = [primary, *samples]
        clusters = self._cluster(texts)
        n = len(texts)

        entropy = 0.0
        for members in clusters:
            p = len(members) / n
            entropy -= p * math.log(p)
        max_entropy = math.log(n)
        normalised = entropy / max_entropy if max_entropy > 0 else 0.0

        primary_cluster = next(m for m in clusters if 0 in m)
        agreement = len(primary_cluster) / n
        score = 0.5 * (1.0 - normalised) + 0.5 * agreement

        return ConsistencyResult(
            semantic_entropy=normalised,
            primary_agreement=agreement,
            consistency_score=score,
            n_samples=len(samples),
            n_clusters=len(clusters),
            entailment_backend=self.entailment_backend,
            clusters=clusters,
        )

    def _cluster(self, texts: list[str]) -> list[list[int]]:
        """Greedy bidirectional-entailment clustering (semantic entropy)."""
        clusters: list[list[int]] = []
        for index, text in enumerate(texts):
            for members in clusters:
                # Cluster identity is transitive-by-representative: the
                # first member anchors the cluster (standard greedy
                # semantic-entropy clustering).
                if self._equivalent(texts[members[0]], text):
                    members.append(index)
                    break
            else:
                clusters.append([index])
        return clusters

    def _equivalent(self, a: str, b: str) -> bool:
        """Mutual entailment under the active backend."""
        if self._nli is not None:
            forward = float(self._nli.score(a, b))
            backward = float(self._nli.score(b, a))
            return (
                forward <= self._entail_divergence
                and backward <= self._entail_divergence
            )
        return self._jaccard(a, b) >= self._lexical_overlap

    @staticmethod
    def _jaccard(a: str, b: str) -> float:
        tokens_a = set(re.findall(r"\w+", a.lower()))
        tokens_b = set(re.findall(r"\w+", b.lower()))
        if not tokens_a and not tokens_b:
            return 1.0
        union = tokens_a | tokens_b
        return len(tokens_a & tokens_b) / len(union)
