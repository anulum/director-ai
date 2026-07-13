# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Claim Coverage (decomposition + attribution)
"""Claim-coverage surface of the NLI scorer.

:class:`ClaimCoverageMixin` decomposes a hypothesis into individual
claims and scores each one against the source: per-claim divergences,
coverage ratios, and sentence-level attributions that map every claim to
its best-matching source sentence. It builds on
:class:`~._nli_chunking.ChunkingMixin` for sentence splitting and chunked
scoring; the reduction fast lanes dispatch through :mod:`._nli_accel`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..mandatory import mandatory_execution
from . import _nli_accel
from ._nli_chunking import ChunkingMixin
from ._nli_numeric import _count_below_threshold

if TYPE_CHECKING:
    from ..types import ClaimAttribution
    from .claim_decomposition import AtomicClaimDecomposer

__all__ = ["ClaimCoverageMixin"]


class ClaimCoverageMixin(ChunkingMixin):
    """Claim decomposition, coverage, and attribution scoring.

    All state is initialised by the composing scorer's ``__init__``;
    the pair-scoring services come from the composing scorer through
    the contracts declared on :class:`ChunkingMixin`.
    """

    # Optional FActScore-style LLM decomposer set by the composing
    # scorer's __init__; None keeps the regex sentence-split behaviour.
    _claim_decomposer: AtomicClaimDecomposer | None

    def decompose_claims(self, text: str) -> list[str]:
        """Split text into individual claims.

        With a configured :class:`AtomicClaimDecomposer` this is the
        FActScore-style LLM extraction (falling back to sentence
        splitting on provider failure, honestly labelled inside the
        decomposer result); without one it is the regex sentence split.
        """
        decomposer = getattr(self, "_claim_decomposer", None)
        if decomposer is not None:
            result = decomposer.decompose(
                text,
                sentence_splitter=self._split_sentences,
            )
            return list(result.claims)
        return self._split_sentences(text)

    def score_decomposed(
        self,
        premise: str,
        hypothesis: str,
    ) -> tuple[float, list[float]]:
        """Score each claim in hypothesis independently against premise.

        Returns (max_score, per_claim_scores).
        """
        claims = self.decompose_claims(hypothesis)
        if not claims:
            return self.score(premise, hypothesis), [self.score(premise, hypothesis)]

        if len(claims) == 1:
            s = self.score(premise, claims[0])
            return s, [s]

        pairs = [(premise, c) for c in claims]
        scores = self.score_batch(pairs)
        return max(scores), scores

    def score_claim_coverage(
        self,
        source: str,
        summary: str,
        support_threshold: float = 0.6,
    ) -> tuple[float, list[float], list[str]]:
        """Decompose summary into claims and compute coverage against source.

        A claim is "supported" when its NLI divergence < support_threshold.
        Coverage = supported_claims / total_claims.

        For long sources, each claim is scored with chunked NLI so that
        at least one source chunk can provide evidence.

        Returns (coverage, per_claim_divergences, claims).
        """
        claims = self.decompose_claims(summary)
        if not claims:
            s = self.score(source, summary)
            return float(s < support_threshold), [s], [summary]

        # Score each claim against the full source via chunked NLI.
        # inner_agg="min" picks the best-matching source chunk per claim.
        divs: list[float] = []
        for claim in claims:
            div, _ = self.score_chunked(
                source,
                claim,
                inner_agg="min",
                outer_agg="mean",
                premise_ratio=0.85,
            )
            divs.append(div)

        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                coverage, _supported = _nli_accel.rust_coverage_from_divergences(
                    [float(d) for d in divs],
                    float(support_threshold),
                )
                return float(coverage), divs, claims
        supported = _count_below_threshold(divs, support_threshold)
        coverage = supported / len(claims)
        return coverage, divs, claims

    def score_claim_coverage_with_attribution(
        self,
        source: str,
        summary: str,
        support_threshold: float = 0.6,
    ) -> tuple[float, list[float], list[str], list[ClaimAttribution]]:
        """Like score_claim_coverage but also returns sentence-level attributions.

        For each claim, finds the source sentence with lowest divergence
        (best evidence match). Returns list of ClaimAttribution objects.
        """
        from ..types import ClaimAttribution

        claims = self.decompose_claims(summary)
        source_sents = self._split_sentences(source)

        if not claims:
            s = self.score(source, summary)
            attr = [
                ClaimAttribution(
                    claim=summary,
                    claim_index=0,
                    source_sentence=source_sents[0] if source_sents else source,
                    source_index=0,
                    divergence=s,
                    supported=s < support_threshold,
                ),
            ]
            return float(s < support_threshold), [s], [summary], attr

        if not source_sents:
            source_sents = [source]

        max_attribution_pairs = 10_000
        n_pairs = len(claims) * len(source_sents)
        if n_pairs > max_attribution_pairs:
            raise ValueError(
                f"Attribution would create {n_pairs} pairs "
                f"({len(claims)} claims × {len(source_sents)} source sentences), "
                f"exceeding limit of {max_attribution_pairs}",
            )

        pairs = [(src_s, claim) for claim in claims for src_s in source_sents]
        all_divs = self.score_batch(pairs)

        n_src = len(source_sents)
        if _nli_accel._RUST_NLI:
            try:
                per_claim_divs, best_indices = _nli_accel.rust_reduce_claim_attribution(
                    [float(v) for v in all_divs],
                    len(claims),
                    n_src,
                )
            except Exception:
                per_claim_divs, best_indices = [], []
        else:
            per_claim_divs, best_indices = [], []

        if not per_claim_divs:
            per_claim_divs = []
            best_indices = []
            for c_idx in range(len(claims)):
                claim_scores = all_divs[c_idx * n_src : (c_idx + 1) * n_src]
                best_idx = int(np.argmin(claim_scores))
                per_claim_divs.append(claim_scores[best_idx])
                best_indices.append(best_idx)

        attributions: list[ClaimAttribution] = []
        for c_idx, claim in enumerate(claims):
            best_idx = int(best_indices[c_idx])
            best_div = float(per_claim_divs[c_idx])
            attributions.append(
                ClaimAttribution(
                    claim=claim,
                    claim_index=c_idx,
                    source_sentence=source_sents[best_idx],
                    source_index=best_idx,
                    divergence=best_div,
                    supported=best_div < support_threshold,
                ),
            )

        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                coverage, _supported = _nli_accel.rust_coverage_from_divergences(
                    [float(d) for d in per_claim_divs],
                    float(support_threshold),
                )
                return float(coverage), per_claim_divs, claims, attributions
        supported = _count_below_threshold(per_claim_divs, support_threshold)
        coverage = supported / len(claims)
        return coverage, per_claim_divs, claims, attributions
