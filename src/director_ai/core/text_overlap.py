# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — measured fast-path lexical word overlap

"""Lexical Jaccard word overlap dispatched to the *measured* faster path.

Many scoring and retrieval heuristics need the Jaccard overlap of two token
sets. The Rust ``rust_word_overlap`` kernel exists, but it is not the faster path
at the input sizes these call sites see: ``benchmarks/word_overlap_crossover.py``
shows Python's ``set``/``split`` is competitive-or-faster across the realistic
10–500-word range, and the kernel's per-call FFI marshalling only amortises at
roughly a thousand words or more. So this shared helper defaults to pure Python
and reaches for Rust only above ``large_input_words``. The two paths are
bit-for-bit identical (case-folded, whitespace-split, punctuation retained on the
token), so the dispatch is purely a speed choice.
"""

from __future__ import annotations

from .mandatory import mandatory_execution

try:
    from backfire_kernel import rust_word_overlap

    _RUST_WORD_OVERLAP = True
except ImportError:  # pragma: no cover - exercised on installs without the kernel
    rust_word_overlap = None
    _RUST_WORD_OVERLAP = False

__all__ = ["word_overlap", "LARGE_INPUT_WORDS"]

# Word count at/above which the Rust kernel is dispatched: below it the pure
# -Python path wins (see benchmarks/word_overlap_crossover.py).
LARGE_INPUT_WORDS = 1000


def _python_jaccard(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    union = words_a | words_b
    return len(words_a & words_b) / len(union) if union else 0.0


def word_overlap(
    text_a: str,
    text_b: str,
    *,
    logger_name: str = __name__,
    large_input_words: int = LARGE_INPUT_WORDS,
) -> float:
    """Return the lexical Jaccard overlap of *text_a* and *text_b* in ``[0, 1]``.

    Pure Python by default; the Rust kernel is used only when either input has at
    least ``large_input_words`` words, where its FFI cost is amortised. ``logger_name``
    labels the mandatory-accelerator audit trail for the Rust path.
    """
    if (
        _RUST_WORD_OVERLAP
        and rust_word_overlap is not None
        and (
            text_a.count(" ") >= large_input_words
            or text_b.count(" ") >= large_input_words
        )
    ):
        with mandatory_execution(
            logger_name, component="rust_word_overlap (large input)"
        ):
            return float(rust_word_overlap(text_a, text_b))
    return _python_jaccard(text_a, text_b)
