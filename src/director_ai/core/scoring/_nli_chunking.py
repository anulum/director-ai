# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — NLI Chunked Scoring (sentence split + chunk aggregation)
"""Chunked-scoring surface of the NLI scorer.

:class:`ChunkingMixin` carries the long-input machinery of
:class:`~director_ai.core.scoring.nli.NLIScorer`: sentence splitting,
token-budgeted chunk building (with and without sliding-window overlap),
and the bidirectional chunked scoring paths, including the
confidence-weighted aggregation variant. Accelerated branches dispatch
through :mod:`._nli_accel`; the composing scorer provides the underlying
pair-scoring services declared as ``TYPE_CHECKING`` stubs below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..mandatory import mandatory_execution
from ..metrics import metrics
from ..text_segmentation import split_sentences as _py_split_sentences
from . import _nli_accel
from ._nli_numeric import _mean_float, _sum_float_list, _weighted_sum_float

__all__ = ["ChunkingMixin"]


class ChunkingMixin:
    """Chunked scoring for long premises and hypotheses.

    All state is initialised by the composing scorer's ``__init__``; the
    annotations below declare that shared contract for static analysis
    without creating attributes.
    """

    # Shared state initialised by the composing scorer.
    max_length: int

    if TYPE_CHECKING:
        # Services provided by the composing scorer.

        def score(self, premise: str, hypothesis: str) -> float: ...

        def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]: ...

        def score_batch_with_confidence(
            self, pairs: list[tuple[str, str]]
        ) -> list[tuple[float, float]]: ...

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split on sentence-ending punctuation + whitespace.

        Avoids splitting after abbreviations (e.g. U.S., Dr., etc.)
        and decimal numbers (e.g. 2.3%). The pure-Python floor delegates
        to the bit-exact port in :mod:`..text_segmentation`; the previous
        regex floor re-inserted a space into decimals (``2.5`` → ``2. 5``),
        diverging from the Rust splitter.
        """
        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                return [
                    s.strip()
                    for s in _nli_accel.rust_split_sentences(text)
                    if s.strip()
                ]
        return _py_split_sentences(text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count (~4 chars/token for English)."""
        return len(text) // 4 + 1

    def _build_chunks(
        self,
        sentences: list[str],
        budget: int,
        overlap_ratio: float = 0.0,
    ) -> list[str]:
        """Group sentences into chunks within *budget* tokens.

        When ``overlap_ratio > 0``, uses sliding-window overlap: after
        filling a chunk, the next chunk starts from the sentence at
        ``(1 - overlap_ratio) * chunk_length`` position. With the
        default ``overlap_ratio=0``, uses 1-sentence overlap (legacy).
        """
        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                return list(
                    _nli_accel.rust_build_chunks(sentences, budget, overlap_ratio)
                )
        if overlap_ratio > 0:
            return self._build_chunks_overlap(sentences, budget, overlap_ratio)

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current and current_tokens + sent_tokens > budget:
                chunks.append(" ".join(current))
                current = [current[-1]]
                current_tokens = self._estimate_tokens(current[0])
            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(" ".join(current))
        return chunks or [" ".join(sentences)]

    def _build_chunks_overlap(
        self,
        sentences: list[str],
        budget: int,
        overlap_ratio: float,
    ) -> list[str]:
        """Sliding-window chunking with configurable token overlap."""
        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                return list(
                    _nli_accel.rust_build_chunks(sentences, budget, overlap_ratio)
                )
        chunks: list[str] = []
        i = 0
        while i < len(sentences):
            current: list[str] = []
            current_tokens = 0
            j = i
            while j < len(sentences):
                st = self._estimate_tokens(sentences[j])
                if current and current_tokens + st > budget:
                    break
                current.append(sentences[j])
                current_tokens += st
                j += 1

            chunks.append(" ".join(current))
            # Stride: advance by (1 - overlap_ratio) * num sentences in chunk
            stride = max(1, int(len(current) * (1.0 - overlap_ratio)))
            i += stride

        return chunks or [" ".join(sentences)]

    def _score_chunked_with_counts(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
        inner_agg: str = "max",
        premise_ratio: float = 0.4,
        overlap_ratio: float = 0.0,
    ) -> tuple[float, list[float], int, int]:
        """Bidirectional chunked scoring with chunk counts.

        Returns (agg_score, per_hyp_scores, prem_count, hyp_count).

        inner_agg controls how premise-chunk scores are combined per
        hypothesis chunk: "max" (default, conservative — worst evidence
        wins), "min" (best evidence wins — use for summarization), or
        "mean".

        outer_agg controls how hypothesis-chunk scores are combined:
        "max" (default) or "mean".

        premise_ratio controls the token budget split between premise and
        hypothesis.  Default 0.4 (40% premise, 60% hypothesis) suits QA;
        use 0.85 for summarization where source documents are long and
        summary sentences are short.
        """
        hyp_budget = int(self.max_length * (1 - premise_ratio))
        prem_budget = int(self.max_length * premise_ratio)

        hyp_fits = self._estimate_tokens(hypothesis) <= hyp_budget
        prem_fits = self._estimate_tokens(premise) <= prem_budget

        if hyp_fits and prem_fits:
            s = self.score(premise, hypothesis)
            metrics.observe("nli_premise_chunks", 1)
            metrics.observe("nli_hypothesis_chunks", 1)
            return s, [s], 1, 1

        hyp_sents = self._split_sentences(hypothesis)
        hyp_chunks = (
            self._build_chunks(hyp_sents, hyp_budget, overlap_ratio)
            if not hyp_fits and len(hyp_sents) > 1
            else [hypothesis]
        )

        prem_sents = self._split_sentences(premise)
        prem_chunks = (
            self._build_chunks(prem_sents, prem_budget, overlap_ratio)
            if not prem_fits and len(prem_sents) > 1
            else [premise]
        )

        pairs = [(pc, hc) for pc in prem_chunks for hc in hyp_chunks]
        all_scores = self.score_batch(pairs)

        n_prem = len(prem_chunks)
        n_hyp = len(hyp_chunks)
        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                agg_rust, per_hyp_rust = _nli_accel.rust_aggregate_chunk_scores(
                    [float(v) for v in all_scores],
                    n_prem,
                    n_hyp,
                    inner_agg,
                    outer_agg,
                )
                metrics.observe("nli_premise_chunks", n_prem)
                metrics.observe("nli_hypothesis_chunks", n_hyp)
                return float(agg_rust), [float(v) for v in per_hyp_rust], n_prem, n_hyp

        per_hyp: list[float] = []
        for h_idx in range(n_hyp):
            scores_h = [all_scores[p * n_hyp + h_idx] for p in range(n_prem)]
            if inner_agg == "min":
                per_hyp.append(min(scores_h))
            elif inner_agg == "mean":
                per_hyp.append(_mean_float(scores_h))
            else:
                per_hyp.append(max(scores_h))

        if outer_agg == "max":
            agg = max(per_hyp)
        elif outer_agg == "trimmed_mean":
            # Drop top 25% of per-hypothesis scores (most divergent) before averaging.
            # More robust to outlier sentences that NLI can't match.
            sorted_scores = sorted(per_hyp)
            keep = max(1, int(len(sorted_scores) * 0.75))
            agg = _sum_float_list(sorted_scores[:keep]) / keep
        else:
            agg = _mean_float(per_hyp)

        metrics.observe("nli_premise_chunks", n_prem)
        metrics.observe("nli_hypothesis_chunks", n_hyp)
        return agg, per_hyp, n_prem, n_hyp

    def score_chunked(
        self,
        premise: str,
        hypothesis: str,
        outer_agg: str = "max",
        inner_agg: str = "max",
        premise_ratio: float = 0.4,
        overlap_ratio: float = 0.0,
    ) -> tuple[float, list[float]]:
        """Bidirectional chunked scoring for long premises and hypotheses.

        Returns (aggregated_score, per_hypothesis_chunk_scores).
        """
        agg, per_hyp, _, _ = self._score_chunked_with_counts(
            premise,
            hypothesis,
            outer_agg=outer_agg,
            inner_agg=inner_agg,
            premise_ratio=premise_ratio,
            overlap_ratio=overlap_ratio,
        )
        return agg, per_hyp

    def score_chunked_confidence_weighted(
        self,
        premise: str,
        hypothesis: str,
        inner_agg: str = "max",
        premise_ratio: float = 0.4,
        overlap_ratio: float = 0.0,
    ) -> tuple[float, list[float]]:
        """Chunked scoring with confidence-weighted outer aggregation.

        Instead of max/mean over hypothesis chunks, weights each chunk's
        divergence by the model's confidence (1 - normalised entropy).
        Uncertain chunks contribute less to the aggregate.
        """
        hyp_budget = int(self.max_length * (1 - premise_ratio))
        prem_budget = int(self.max_length * premise_ratio)

        hyp_fits = self._estimate_tokens(hypothesis) <= hyp_budget
        prem_fits = self._estimate_tokens(premise) <= prem_budget

        if hyp_fits and prem_fits:
            result = self.score_batch_with_confidence([(premise, hypothesis)])
            s, _ = result[0]
            return s, [s]

        hyp_sents = self._split_sentences(hypothesis)
        hyp_chunks = (
            self._build_chunks(hyp_sents, hyp_budget, overlap_ratio)
            if not hyp_fits and len(hyp_sents) > 1
            else [hypothesis]
        )

        prem_sents = self._split_sentences(premise)
        prem_chunks = (
            self._build_chunks(prem_sents, prem_budget, overlap_ratio)
            if not prem_fits and len(prem_sents) > 1
            else [premise]
        )

        pairs = [(pc, hc) for pc in prem_chunks for hc in hyp_chunks]
        results_with_conf = self.score_batch_with_confidence(pairs)

        n_prem = len(prem_chunks)
        n_hyp = len(hyp_chunks)
        if _nli_accel._RUST_NLI:
            with mandatory_execution(__name__, component="mandatory accelerated path"):
                flat_scores = [float(v[0]) for v in results_with_conf]
                flat_conf = [float(v[1]) for v in results_with_conf]
                agg_rust, per_hyp_rust = (
                    _nli_accel.rust_aggregate_chunk_scores_confidence_weighted(
                        flat_scores,
                        flat_conf,
                        n_prem,
                        n_hyp,
                        inner_agg,
                    )
                )
                return float(agg_rust), [float(v) for v in per_hyp_rust]

        per_hyp: list[float] = []
        per_hyp_conf: list[float] = []

        for h_idx in range(n_hyp):
            scores_h = [results_with_conf[p * n_hyp + h_idx] for p in range(n_prem)]
            divs = [s[0] for s in scores_h]
            if inner_agg == "min":
                per_hyp.append(min(divs))
            elif inner_agg == "mean":
                per_hyp.append(_mean_float(divs))
            else:
                per_hyp.append(max(divs))
            avg_conf = _mean_float([s[1] for s in scores_h])
            per_hyp_conf.append(avg_conf)

        # Confidence-weighted mean
        total_weight = _sum_float_list(per_hyp_conf)
        if total_weight > 1e-9:
            agg = _weighted_sum_float(per_hyp, per_hyp_conf) / total_weight
        else:
            agg = _mean_float(per_hyp)

        return agg, per_hyp
