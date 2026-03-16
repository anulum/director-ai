# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Verified Scorer (Sentence-Level Multi-Signal)

"""Sentence-level fact verification with multi-signal consensus.

Decomposes both response and source into sentences, matches each
response sentence to its best source sentence, scores each pair
with multiple independent signals, and reports per-claim verdicts
with calibrated confidence.

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

logger = logging.getLogger("DirectorAI.VerifiedScorer")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
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
        "don't",
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
class ClaimVerdict:
    claim: str
    claim_index: int
    matched_source: str
    source_index: int
    nli_divergence: float
    entity_match: float
    numerical_match: bool
    negation_flip: bool
    verdict: str  # "supported", "contradicted", "unverifiable"
    confidence: float  # 0.0-1.0


@dataclass
class VerificationResult:
    approved: bool
    overall_score: float
    confidence: str  # "high", "medium", "low"
    claims: list[ClaimVerdict] = field(default_factory=list)
    supported_count: int = 0
    contradicted_count: int = 0
    unverifiable_count: int = 0
    coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "overall_score": round(self.overall_score, 4),
            "confidence": self.confidence,
            "supported": self.supported_count,
            "contradicted": self.contradicted_count,
            "unverifiable": self.unverifiable_count,
            "coverage": round(self.coverage, 4),
            "claims": [
                {
                    "claim": c.claim,
                    "matched_source": c.matched_source,
                    "nli_divergence": round(c.nli_divergence, 4),
                    "entity_match": round(c.entity_match, 2),
                    "numerical_match": c.numerical_match,
                    "negation_flip": c.negation_flip,
                    "verdict": c.verdict,
                    "confidence": round(c.confidence, 2),
                }
                for c in self.claims
            ],
        }


class VerifiedScorer:
    """Multi-signal sentence-level fact verifier.

    Parameters
    ----------
    nli_scorer : NLIScorer or None — for NLI entailment signal.
    nli_threshold : float — divergence above this = contradiction (default 0.65).
    support_threshold : float — divergence below this = supported (default 0.35).
    min_confidence : float — below this, verdict is "unverifiable" (default 0.4).
    """

    def __init__(
        self,
        nli_scorer=None,
        nli_threshold: float = 0.65,
        support_threshold: float = 0.35,
        min_confidence: float = 0.4,
    ):
        self._nli = nli_scorer
        self._nli_threshold = nli_threshold
        self._support_threshold = support_threshold
        self._min_confidence = min_confidence

    def verify(
        self,
        response: str,
        source: str,
    ) -> VerificationResult:
        """Verify response against source at sentence level.

        Decomposes both into sentences, matches each response sentence
        to its best source sentence, runs multi-signal checks.
        """
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

            best_src_idx, nli_div = self._find_best_match(r_sent, source_sents)
            best_src = source_sents[best_src_idx]

            entity_score = _entity_overlap(r_sent, best_src)
            num_match = _numerical_consistency(r_sent, best_src)
            neg_flip = _negation_flip(r_sent, best_src)

            verdict, conf = self._multi_signal_verdict(
                nli_div,
                entity_score,
                num_match,
                neg_flip,
            )

            claims.append(
                ClaimVerdict(
                    claim=r_sent,
                    claim_index=r_idx,
                    matched_source=best_src,
                    source_index=best_src_idx,
                    nli_divergence=nli_div,
                    entity_match=entity_score,
                    numerical_match=num_match,
                    negation_flip=neg_flip,
                    verdict=verdict,
                    confidence=conf,
                )
            )

        if not claims:
            return VerificationResult(
                approved=True,
                overall_score=1.0,
                confidence="low",
            )

        supported = sum(1 for c in claims if c.verdict == "supported")
        contradicted = sum(1 for c in claims if c.verdict == "contradicted")
        unverifiable = sum(1 for c in claims if c.verdict == "unverifiable")
        coverage = supported / len(claims)

        # Overall: fail if ANY claim is contradicted with high confidence
        high_conf_contradictions = sum(
            1 for c in claims if c.verdict == "contradicted" and c.confidence >= 0.6
        )
        approved = high_conf_contradictions == 0

        # Overall score: weighted by confidence
        scores = []
        for c in claims:
            if c.verdict == "supported":
                scores.append(1.0 * c.confidence)
            elif c.verdict == "contradicted":
                scores.append(0.0)
            else:
                scores.append(0.5)
        overall = sum(scores) / len(scores) if scores else 0.5

        # Confidence level
        avg_conf = sum(c.confidence for c in claims) / len(claims)
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
        claim_words = set(claim.lower().split())
        for i, src in enumerate(source_sents):
            src_words = set(src.lower().split())
            union = claim_words | src_words
            overlap = len(claim_words & src_words) / len(union) if union else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        return best_idx, 1.0 - best_overlap

    def _multi_signal_verdict(
        self,
        nli_div: float,
        entity_score: float,
        num_match: bool,
        neg_flip: bool,
    ) -> tuple[str, float]:
        """Combine signals into verdict + confidence."""
        signals_support = 0
        signals_contradict = 0
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

        if total_signals == 0:
            return "unverifiable", 0.0

        support_ratio = signals_support / total_signals
        contradict_ratio = signals_contradict / total_signals

        if contradict_ratio >= 0.5:
            confidence = contradict_ratio
            return "contradicted", confidence
        if support_ratio >= 0.5:
            confidence = support_ratio
            return "supported", confidence

        return "unverifiable", max(support_ratio, contradict_ratio)


def _split_sentences(text: str) -> list[str]:
    return [
        s.strip() for s in _SENT_SPLIT.split(text) if s.strip() and len(s.split()) >= 3
    ]


def _entity_overlap(text_a: str, text_b: str) -> float:
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
    nums_a = set(_NUM_RE.findall(text_a))
    nums_b = set(_NUM_RE.findall(text_b))
    if not nums_a and not nums_b:
        return None  # no numbers to compare
    if not nums_a or not nums_b:
        return None
    return bool(nums_a & nums_b)


def _negation_flip(claim: str, source: str) -> bool:
    """Detect if claim negates something the source states positively, or vice versa."""
    claim_words = set(claim.lower().split())
    source_words = set(source.lower().split())
    claim_has_neg = bool(claim_words & _NEG_WORDS)
    source_has_neg = bool(source_words & _NEG_WORDS)
    # Shared content words (excluding negation and stop words)
    content_overlap = len((claim_words - _NEG_WORDS) & (source_words - _NEG_WORDS))
    return content_overlap >= 3 and claim_has_neg != source_has_neg
