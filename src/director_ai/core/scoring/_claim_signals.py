# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Claim Verification Signals (Rust-accelerated text kernels)

"""Text-level verification signals shared by the verified scorer.

Pure functions over claim/source strings: sentence splitting, atomic-claim
decomposition, entity overlap, numerical consistency, negation-flip
detection, lexical traceability, word overlap and the summation reducers.
Each signal dispatches to its mandatory ``backfire_kernel`` accelerator
when present (``_RUST_SIGNALS`` gates the dispatch in this module's
globals) and falls back to the exact-parity Python implementation when the
:mod:`director_ai.core.mandatory` policy permits.
"""

from __future__ import annotations

import logging
import re
from typing import cast

from ..mandatory import mandatory_execution, require_rust_kernel
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
        require_rust_kernel("rust_sum_i64")

    def rust_sum_f64(_values: list[float]) -> float:
        """Raise to signal the mandatory Rust float-sum accelerator is missing."""
        require_rust_kernel("rust_sum_f64")

    def rust_entity_overlap(_claim: str, _source: str) -> float:
        """Raise to signal the mandatory Rust entity-overlap signal is missing."""
        require_rust_kernel("rust_entity_overlap")

    def rust_negation_flip(_claim: str, _source: str) -> bool:
        """Raise to signal the mandatory Rust negation-flip signal is missing."""
        require_rust_kernel("rust_negation_flip")

    def rust_numerical_consistency(_claim: str, _source: str) -> bool:
        """Raise to signal the mandatory Rust numeric-consistency signal is missing."""
        require_rust_kernel("rust_numerical_consistency")

    def rust_split_sentences(_text: str) -> list[str]:
        """Raise to signal the mandatory Rust sentence splitter is missing."""
        require_rust_kernel("rust_split_sentences")

    def rust_traceability(_claim: str, _source: str) -> float:
        """Raise to signal the mandatory Rust traceability signal is missing."""
        require_rust_kernel("rust_traceability")


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
