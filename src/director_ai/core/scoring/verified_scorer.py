# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Verified Scorer (Atomic Multi-Span Multi-Signal)

"""Atomic fact verification with multi-span, multi-signal consensus.

Decomposes responses into sentences or atomic claims, ranks source
sentences for each claim, aggregates the top evidence spans, scores the
claim/evidence pair with multiple independent signals, and reports
per-claim verdicts with calibrated confidence and provenance.

Signals:
1. NLI entailment (FactCG — primary)
2. Entity consistency (named entities must match)
3. Numerical consistency (numbers/dates must match)
4. Negation detection (response must not negate source)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from ..mandatory import mandatory_execution
from ..text_overlap import word_overlap

logger = logging.getLogger("DirectorAI.VerifiedScorer")

try:
    from backfire_kernel import (
        rust_entity_overlap,
        rust_negation_flip,
        rust_numerical_consistency,
        rust_split_sentences,
        rust_sum_f64,
        rust_sum_i64,
        rust_traceability,
    )

    _RUST_SIGNALS = True
except ImportError:
    _RUST_SIGNALS = True

    def rust_sum_i64(_values: list[int]) -> int:
        """Raise to signal the mandatory Rust int-sum accelerator is missing."""
        raise RuntimeError("backfire_kernel rust_sum_i64 is unavailable")

    def rust_sum_f64(_values: list[float]) -> float:
        """Raise to signal the mandatory Rust float-sum accelerator is missing."""
        raise RuntimeError("backfire_kernel rust_sum_f64 is unavailable")

    def rust_entity_overlap(_claim: str, _source: str) -> float:
        """Raise to signal the mandatory Rust entity-overlap signal is missing."""
        raise RuntimeError("backfire_kernel rust_entity_overlap is unavailable")

    def rust_negation_flip(_claim: str, _source: str) -> bool:
        """Raise to signal the mandatory Rust negation-flip signal is missing."""
        raise RuntimeError("backfire_kernel rust_negation_flip is unavailable")

    def rust_numerical_consistency(_claim: str, _source: str) -> bool:
        """Raise to signal the mandatory Rust numeric-consistency signal is missing."""
        raise RuntimeError("backfire_kernel rust_numerical_consistency is unavailable")

    def rust_split_sentences(_text: str) -> list[str]:
        """Raise to signal the mandatory Rust sentence splitter is missing."""
        raise RuntimeError("backfire_kernel rust_split_sentences is unavailable")

    def rust_traceability(_claim: str, _source: str) -> float:
        """Raise to signal the mandatory Rust traceability signal is missing."""
        raise RuntimeError("backfire_kernel rust_traceability is unavailable")


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(
    r",?\s+(?:and|but|while|whereas|although|however|moreover|furthermore)\s+",
    re.IGNORECASE,
)
_NUM_RE = re.compile(r"\b\d[\d,.]*\b")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
_NEG_WORDS = frozenset(
    {
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "cannot",
        "can't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "won't",
        "wouldn't",
        "shouldn't",
        "couldn't",
        "doesn't",
        "didn't",
        "hasn't",
        "haven't",
        "hadn't",
        "without",
        "none",
        "nobody",
    }
)


@dataclass
class SourceSpan:
    """A source text span that supports or contradicts a claim."""

    text: str
    index: int
    nli_divergence: float
    entity_match: float = 1.0
    numerical_match: bool | None = None


@dataclass
class ClaimVerdict:
    """Per-claim verification outcome with each contributing signal."""

    claim: str
    claim_index: int
    matched_source: str
    source_index: int
    nli_divergence: float
    entity_match: float
    numerical_match: bool | None
    negation_flip: bool
    traceability: float
    verdict: str  # "supported", "contradicted", "unverifiable", "fabricated"
    confidence: float
    traceability_mode: str = "lexical"
    aggregated_source: str = ""
    source_indices: list[int] = field(default_factory=list)
    evidence_spans: list[SourceSpan] = field(default_factory=list)
    evidence_mode: str = "single_span"
    is_atomic: bool = False


@dataclass
class VerificationResult:
    """Whole-response verdict: approval, score, and per-claim breakdown."""

    approved: bool
    overall_score: float
    confidence: str  # "high", "medium", "low"
    claims: list[ClaimVerdict] = field(default_factory=list)
    supported_count: int = 0
    contradicted_count: int = 0
    fabricated_count: int = 0
    unverifiable_count: int = 0
    coverage: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Render the verdict and every claim as a JSON-ready mapping."""
        return {
            "approved": self.approved,
            "overall_score": round(self.overall_score, 4),
            "confidence": self.confidence,
            "supported": self.supported_count,
            "contradicted": self.contradicted_count,
            "fabricated": self.fabricated_count,
            "unverifiable": self.unverifiable_count,
            "coverage": round(self.coverage, 4),
            "claims": [
                {
                    "claim": c.claim,
                    "matched_source": c.matched_source,
                    "aggregated_source": c.aggregated_source,
                    "source_indices": c.source_indices,
                    "nli_divergence": round(c.nli_divergence, 4),
                    "entity_match": round(c.entity_match, 2),
                    "numerical_match": c.numerical_match,
                    "negation_flip": c.negation_flip,
                    "traceability": round(c.traceability, 2),
                    "traceability_mode": c.traceability_mode,
                    "verdict": c.verdict,
                    "evidence_mode": c.evidence_mode,
                    "is_atomic": c.is_atomic,
                    "evidence_spans": [
                        {
                            "text": s.text,
                            "nli_divergence": round(s.nli_divergence, 4),
                        }
                        for s in c.evidence_spans
                    ],
                    "confidence": round(c.confidence, 2),
                }
                for c in self.claims
            ],
        }


class TraceabilityScorer(Protocol):
    """Semantic traceability backend used by ``VerifiedScorer``.

    Implementations return a groundedness similarity in [0, 1] for
    ``premise`` and ``hypothesis``. ``EmbedBackend`` satisfies this
    protocol without importing sentence-transformers on the default path.
    """

    def score(self, premise: str, hypothesis: str) -> float:
        """Return the groundedness similarity of ``hypothesis`` to ``premise``."""
        ...


class VerifiedScorer:
    """Multi-signal atomic fact verifier with source-span provenance.

    Parameters
    ----------
    nli_scorer : NLIScorer or None — for NLI entailment signal.
    nli_threshold : float — divergence above this = contradiction (default 0.65).
    support_threshold : float — divergence below this = supported (default 0.35).
    min_confidence : float — below this, verdict is "unverifiable" (default 0.4).
    traceability_scorer : object or None — optional semantic traceability scorer.
    traceability_mode : "auto" | "lexical" | "semantic" | "disabled".
    """

    def __init__(
        self,
        nli_scorer: Any = None,
        nli_threshold: float = 0.65,
        support_threshold: float = 0.35,
        min_confidence: float = 0.4,
        traceability_scorer: TraceabilityScorer | None = None,
        traceability_mode: Literal["auto", "lexical", "semantic", "disabled"] = "auto",
    ):
        if traceability_mode not in {"auto", "lexical", "semantic", "disabled"}:
            raise ValueError(
                "traceability_mode must be one of: auto, lexical, semantic, disabled"
            )
        if traceability_mode == "semantic" and traceability_scorer is None:
            raise ValueError("semantic traceability requires traceability_scorer")
        self._nli = nli_scorer
        self._nli_threshold = nli_threshold
        self._support_threshold = support_threshold
        self._min_confidence = min_confidence
        self._traceability_scorer = traceability_scorer
        self._traceability_mode = traceability_mode

    def verify(
        self,
        response: str,
        source: str,
        atomic: bool = False,
        evidence_top_k: int = 3,
    ) -> VerificationResult:
        """Verify response against source.

        Parameters
        ----------
        atomic : bool
            When True, decomposes compound sentences into atomic claims
            before matching. Catches errors in compound sentences where
            one half is correct and the other is fabricated.
        evidence_top_k : int
            Number of source spans to attach as evidence per claim.
        """
        if atomic:
            response_sents = _decompose_atomic(response)
        else:
            response_sents = _split_sentences(response)
        source_sents = _split_sentences(source)

        if not response_sents:
            return VerificationResult(
                approved=True,
                overall_score=1.0,
                confidence="low",
            )
        if not source_sents:
            return VerificationResult(
                approved=False,
                overall_score=0.0,
                confidence="low",
                unverifiable_count=len(response_sents),
            )

        # Match each response sentence to best source sentence via NLI
        claims = []
        for r_idx, r_sent in enumerate(response_sents):
            if len(r_sent.split()) < 3:
                continue

            spans = self._find_top_k_matches(r_sent, source_sents, k=evidence_top_k)
            best_span = spans[0] if spans else SourceSpan("", 0, 1.0)
            best_src_idx = best_span.index
            best_src = best_span.text
            aggregate_src, source_indices = _aggregate_evidence_spans(spans)
            evidence_src = aggregate_src or best_src
            evidence_mode = "multi_span" if len(source_indices) > 1 else "single_span"
            nli_div = self._score_evidence_premise(evidence_src, r_sent, best_span)

            entity_score = _entity_overlap(r_sent, evidence_src)
            num_match = _numerical_consistency(r_sent, evidence_src)
            neg_flip = _negation_flip(r_sent, evidence_src)
            trace, trace_mode = self._score_traceability(r_sent, evidence_src)

            verdict, conf = self._multi_signal_verdict(
                nli_div,
                entity_score,
                num_match,
                neg_flip,
                trace,
                nli_available=self._nli_available(),
                traceability_mode=trace_mode,
            )

            claims.append(
                ClaimVerdict(
                    claim=r_sent,
                    claim_index=r_idx,
                    matched_source=best_src,
                    source_index=best_src_idx,
                    aggregated_source=evidence_src,
                    source_indices=source_indices or [best_src_idx],
                    nli_divergence=nli_div,
                    entity_match=entity_score,
                    numerical_match=num_match,
                    negation_flip=neg_flip,
                    traceability=trace,
                    verdict=verdict,
                    confidence=conf,
                    traceability_mode=trace_mode,
                    evidence_spans=spans,
                    evidence_mode=evidence_mode,
                    is_atomic=atomic,
                )
            )

        if not claims:
            return VerificationResult(
                approved=True,
                overall_score=1.0,
                confidence="low",
            )

        supported = _sum_int([1 if c.verdict == "supported" else 0 for c in claims])
        contradicted = _sum_int(
            [1 if c.verdict == "contradicted" else 0 for c in claims]
        )
        fabricated = _sum_int([1 if c.verdict == "fabricated" else 0 for c in claims])
        unverifiable = _sum_int(
            [1 if c.verdict == "unverifiable" else 0 for c in claims]
        )
        coverage = supported / len(claims)

        # Fail if ANY claim is contradicted/fabricated with high confidence
        high_conf_failures = _sum_int(
            [
                1
                if c.verdict in ("contradicted", "fabricated") and c.confidence >= 0.6
                else 0
                for c in claims
            ]
        )
        approved = high_conf_failures == 0

        # Overall score: weighted by confidence
        scores = []
        for c in claims:
            if c.verdict == "supported":
                scores.append(1.0 * c.confidence)
            elif c.verdict in ("contradicted", "fabricated"):
                scores.append(0.0)
            else:
                scores.append(0.5)
        overall = _sum_float(scores) / len(scores) if scores else 0.5

        # Confidence level
        avg_conf = _sum_float([c.confidence for c in claims]) / len(claims)
        if avg_conf >= 0.7:
            conf_level = "high"
        elif avg_conf >= 0.4:
            conf_level = "medium"
        else:
            conf_level = "low"

        return VerificationResult(
            approved=approved,
            overall_score=overall,
            confidence=conf_level,
            claims=claims,
            supported_count=supported,
            contradicted_count=contradicted,
            fabricated_count=fabricated,
            unverifiable_count=unverifiable,
            coverage=coverage,
        )

    def _find_best_match(
        self,
        claim: str,
        source_sents: list[str],
    ) -> tuple[int, float]:
        """Find the source sentence with lowest NLI divergence to claim."""
        if self._nli and self._nli.model_available:
            pairs = [(src, claim) for src in source_sents]
            divs = self._nli.score_batch(pairs)
            best_idx = int(min(range(len(divs)), key=lambda i: divs[i]))
            return best_idx, divs[best_idx]

        # Fallback: word overlap
        best_idx, best_overlap = 0, 0.0
        for i, src in enumerate(source_sents):
            overlap = _word_overlap(claim, src)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        return best_idx, 1.0 - best_overlap

    def _find_top_k_matches(
        self,
        claim: str,
        source_sents: list[str],
        k: int = 3,
    ) -> list[SourceSpan]:
        """Find top-K source spans ranked by relevance to the claim."""
        if self._nli and self._nli.model_available:
            pairs = [(src, claim) for src in source_sents]
            divs = self._nli.score_batch(pairs)
            ranked = sorted(range(len(divs)), key=lambda i: divs[i])
            return [
                SourceSpan(
                    text=source_sents[i],
                    index=i,
                    nli_divergence=divs[i],
                    entity_match=_entity_overlap(claim, source_sents[i]),
                    numerical_match=_numerical_consistency(claim, source_sents[i]),
                )
                for i in ranked[:k]
            ]

        # Fallback: word overlap
        scored = []
        for i, src in enumerate(source_sents):
            overlap = _word_overlap(claim, src)
            scored.append((i, 1.0 - overlap))
        scored.sort(key=lambda x: x[1])
        return [
            SourceSpan(
                text=source_sents[i],
                index=i,
                nli_divergence=div,
                entity_match=_entity_overlap(claim, source_sents[i]),
                numerical_match=_numerical_consistency(claim, source_sents[i]),
            )
            for i, div in scored[:k]
        ]

    def _score_evidence_premise(
        self,
        premise: str,
        claim: str,
        fallback_span: SourceSpan,
    ) -> float:
        """Score the aggregated evidence premise when NLI is available."""
        if self._nli and self._nli.model_available and premise != fallback_span.text:
            return float(self._nli.score_batch([(premise, claim)])[0])
        return fallback_span.nli_divergence

    def _nli_available(self) -> bool:
        return bool(self._nli and self._nli.model_available)

    def _score_traceability(self, claim: str, source: str) -> tuple[float, str]:
        """Score claim grounding against evidence with the configured backend."""
        if self._traceability_mode == "disabled":
            return 1.0, "disabled"
        if (
            self._traceability_mode in {"auto", "semantic"}
            and self._traceability_scorer is not None
        ):
            score = float(self._traceability_scorer.score(source, claim))
            return max(0.0, min(1.0, score)), "semantic"
        return _traceability(claim, source), "lexical"

    def _multi_signal_verdict(
        self,
        nli_div: float,
        entity_score: float,
        num_match: bool | None,
        neg_flip: bool,
        traceability: float,
        *,
        nli_available: bool = False,
        traceability_mode: str = "lexical",
    ) -> tuple[str, float]:
        """Combine signals into verdict + confidence."""
        signals_support = 0
        signals_contradict = 0
        signals_fabricate = 0
        total_signals = 0

        # Signal 1: NLI
        total_signals += 1
        if nli_div < self._support_threshold:
            signals_support += 1
        elif nli_div > self._nli_threshold:
            signals_contradict += 1

        # Signal 2: Entity overlap
        if entity_score > 0:
            total_signals += 1
            if entity_score >= 0.5:
                signals_support += 1
            elif entity_score < 0.2:
                signals_contradict += 1

        # Signal 3: Numerical consistency
        if not num_match and num_match is not None:
            total_signals += 1
            signals_contradict += 1
        elif num_match:
            total_signals += 1
            signals_support += 1

        # Signal 4: Negation flip
        if neg_flip:
            total_signals += 1
            signals_contradict += 1

        # Signal 5: Traceability (fabrication detection)
        # Low traceability is only decisive when NLI is unavailable or uncertain.
        nli_uncertain = self._support_threshold <= nli_div <= self._nli_threshold
        traceability_applies = traceability_mode != "disabled" and (
            not nli_available or nli_uncertain
        )
        if traceability_applies:
            total_signals += 1
            if traceability >= 0.5:
                signals_support += 1
            elif traceability < 0.2:
                signals_fabricate += 1

        if total_signals == 0:
            return "unverifiable", 0.0

        support_ratio = signals_support / total_signals
        contradict_ratio = signals_contradict / total_signals
        fabricate_ratio = signals_fabricate / total_signals

        # Fabrication: very low traceability overrides only when NLI cannot decide.
        if traceability_applies and traceability < 0.15:
            return "fabricated", 0.7 + (1.0 - traceability) * 0.3
        if fabricate_ratio > 0 and contradict_ratio == 0 and support_ratio < 0.5:
            return "fabricated", 0.5 + (1.0 - traceability) * 0.5

        if contradict_ratio >= 0.5:
            return "contradicted", contradict_ratio
        if support_ratio >= 0.5:
            return "supported", support_ratio

        return "unverifiable", max(support_ratio, contradict_ratio)


def _split_sentences(text: str) -> list[str]:
    if _RUST_SIGNALS:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            sentences = [s.strip() for s in rust_split_sentences(text) if s.strip()]
            filtered = [s for s in sentences if len(s.split()) >= 3]
            if filtered:
                return filtered
    return [
        s.strip() for s in _SENT_SPLIT.split(text) if s.strip() and len(s.split()) >= 3
    ]


def _aggregate_evidence_spans(spans: list[SourceSpan]) -> tuple[str, list[int]]:
    """Return source-order evidence text and source indices for a claim."""
    ordered = sorted(spans, key=lambda span: span.index)
    texts: list[str] = []
    indices: list[int] = []
    seen: set[int] = set()
    for span in ordered:
        if span.index in seen or not span.text.strip():
            continue
        seen.add(span.index)
        texts.append(span.text.strip())
        indices.append(span.index)
    return " ".join(texts), indices


def _decompose_atomic(text: str) -> list[str]:
    """Decompose text into atomic claims.

    Splits sentences on coordinating/adversative conjunctions when both
    halves have enough content to stand alone (>= 4 words).  Compound
    claims like "X is A and Y is B" become two atomic claims.
    """
    sentences = _split_sentences(text)
    claims: list[str] = []
    for sent in sentences:
        parts = _CLAUSE_SPLIT.split(sent)
        for part in parts:
            part = part.strip()
            if len(part.split()) >= 4:
                claims.append(part)
            elif claims:
                claims[-1] = claims[-1] + " " + part
    return claims if claims else sentences


def _entity_overlap(text_a: str, text_b: str) -> float:
    if _RUST_SIGNALS:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(rust_entity_overlap(text_a, text_b))
    ents_a = set(_ENTITY_RE.findall(text_a))
    ents_b = set(_ENTITY_RE.findall(text_b))
    if not ents_a and not ents_b:
        return 1.0
    union = ents_a | ents_b
    if not union:
        return 1.0
    return len(ents_a & ents_b) / len(union)


def _numerical_consistency(text_a: str, text_b: str) -> bool | None:
    """Check if numbers in text_a match numbers in text_b."""
    if _RUST_SIGNALS:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return cast("bool | None", rust_numerical_consistency(text_a, text_b))
    nums_a = set(_NUM_RE.findall(text_a))
    nums_b = set(_NUM_RE.findall(text_b))
    if not nums_a and not nums_b:
        return None
    if not nums_a or not nums_b:
        return None
    return bool(nums_a & nums_b)


def _negation_flip(claim: str, source: str) -> bool:
    """Detect if claim negates something the source states positively, or vice versa."""
    if _RUST_SIGNALS:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return cast(bool, rust_negation_flip(claim, source))
    claim_words = set(claim.lower().split())
    source_words = set(source.lower().split())
    claim_has_neg = bool(claim_words & _NEG_WORDS)
    source_has_neg = bool(source_words & _NEG_WORDS)
    content_overlap = len((claim_words - _NEG_WORDS) & (source_words - _NEG_WORDS))
    return content_overlap >= 3 and claim_has_neg != source_has_neg


_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "and",
        "but",
        "or",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
    }
)


def _traceability(claim: str, source: str) -> float:
    """Measure how much of the claim's content words appear in the source.

    Returns 0.0-1.0. Low traceability means the claim contains
    information not present in the source (potential fabrication).
    """
    if _RUST_SIGNALS:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(rust_traceability(claim, source))
    # Mirror the Rust ``signals::traceability`` kernel exactly: whitespace-split
    # tokens (``to_lower_words``), then drop stop and negation words. The Rust and
    # Python stop/negation sets are kept identical (see the parity test); using
    # ``\w+`` here instead would split hyphens/contractions and strip trailing
    # punctuation, diverging from the kernel.
    claim_words = set(claim.lower().split()) - _STOP_WORDS - _NEG_WORDS
    source_words = set(source.lower().split()) - _STOP_WORDS - _NEG_WORDS
    if not claim_words:
        return 1.0
    return len(claim_words & source_words) / len(claim_words)


def _word_overlap(text_a: str, text_b: str) -> float:
    """Jaccard lexical overlap in ``[0, 1]`` for two texts.

    Delegates to the shared measured-fast-path helper (pure Python below a large
    -input threshold, Rust above it). See :mod:`director_ai.core.text_overlap`.
    """
    return word_overlap(text_a, text_b, logger_name=__name__)


def _sum_int(values: list[int]) -> int:
    if _RUST_SIGNALS:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return int(rust_sum_i64(values))
    return sum(values)


def _sum_float(values: list[float]) -> float:
    if _RUST_SIGNALS:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return float(rust_sum_f64(values))
    return sum(values)
