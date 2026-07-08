# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Reasoning chain verification — verify the logic, not just the answer.

Extracts reasoning steps from chain-of-thought responses and verifies
that each step follows from its premises. Detects:
- Non-sequiturs (conclusion doesn't follow from premise)
- Circular reasoning (step references itself)
- Unsupported leaps (conclusion introduced without any premise)

Uses NLI scoring for semantic entailment and mandatory Rust lexical kernels
for deterministic overlap checks.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from ..mandatory import mandatory_execution
from ..text_overlap import word_overlap
from ..text_segmentation import extract_reasoning_steps as _py_extract_steps
from ..text_segmentation import split_sentences as _py_split_sentences
from .citation_tracer import TraceResult, trace_citations
from .fallacy_detector import FallacyMatch, detect_fallacies
from .math_consistency import ArithmeticCheck, verify_arithmetic

__all__ = [
    "ReasoningStep",
    "ReasoningVerdict",
    "ReasoningChainResult",
    "verify_reasoning_chain",
]

try:  # pragma: no cover - optional acceleration
    from backfire_kernel import (
        rust_extract_reasoning_steps,
        rust_split_sentences,
    )

    _RUST_REASONING = True
except ImportError:  # pragma: no cover - bit-exact pure-Python fallback (ADR-0001)
    # ADR-0001: backfire-kernel is the opt-in ``[rust]`` extra. Absent it, the step
    # extractor and sentence splitter fall back to the bit-exact ``text_segmentation``
    # ports (proven byte-for-byte against the kernel in
    # tests/test_text_segmentation_parity.py). The flag is False so the rust-or-port
    # helpers below take their Python branch.
    _RUST_REASONING = False
    rust_extract_reasoning_steps = _py_extract_steps
    rust_split_sentences = _py_split_sentences


def _raw_reasoning_steps(text: str) -> list[str]:
    """Rust step extractor when installed, else the bit-exact Python port."""
    if _RUST_REASONING:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return list(rust_extract_reasoning_steps(text))
    return _py_extract_steps(text)


def _raw_split_sentences(text: str) -> list[str]:
    """Rust sentence splitter when installed, else the bit-exact Python port.

    A present-but-erroring kernel degrades to an empty list here (the caller has a
    regex secondary fallback), preserving the module's long-standing behaviour.
    """
    if _RUST_REASONING:
        try:
            return list(rust_split_sentences(text))
        except Exception:
            return []
    return _py_split_sentences(text)


_STEP_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:Step\s+)?(\d+)[.):]\s*(.+?)(?=\n\s*(?:Step\s+)?\d+[.):]+|\Z)",
        re.DOTALL,
    ),
    re.compile(r"(?:^|\n)\s*[-*]\s+(.+?)(?=\n\s*[-*]|\Z)", re.DOTALL),
    re.compile(
        r"(?:^|\n)(?:First|Second|Third|Next|Then|Finally|Therefore|Thus|Hence|So),?\s+(.+?)(?=\n(?:First|Second|Third|Next|Then|Finally|Therefore|Thus|Hence|So)|\Z)",
        re.DOTALL | re.IGNORECASE,
    ),
]

_CONCLUSION_MARKERS = re.compile(
    r"^(?:therefore|thus|hence|so|consequently|as a result|in conclusion|this means|it follows|we can conclude)",
    re.IGNORECASE,
)

_STEP_LABEL_RE = re.compile(
    r"^\s*(?:Step\s+)?\d+[.:)]\s*",
    re.IGNORECASE,
)

_BECAUSE_PATTERN = re.compile(
    r"(.+?)\s+because\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""

    index: int
    text: str
    is_conclusion: bool = False


@dataclass
class ReasoningVerdict:
    """Verdict on a single reasoning step."""

    step_index: int
    step_text: str
    verdict: str  # "supported", "non_sequitur", "unsupported_leap", "circular"
    confidence: float  # 0-1
    reason: str = ""
    premise_text: str = ""  # which prior step was used as premise


@dataclass
class ReasoningChainResult:
    """Result of verifying an entire reasoning chain."""

    steps_found: int
    verdicts: list[ReasoningVerdict] = field(default_factory=list)
    chain_valid: bool = True
    issues_found: int = 0
    math_errors: list[ArithmeticCheck] = field(default_factory=list)
    fallacies: list[FallacyMatch] = field(default_factory=list)
    citation_trace: TraceResult | None = None

    @property
    def non_sequiturs(self) -> list[ReasoningVerdict]:
        """Return steps whose conclusion does not follow from prior steps."""
        return [v for v in self.verdicts if v.verdict == "non_sequitur"]

    @property
    def unsupported_leaps(self) -> list[ReasoningVerdict]:
        """Return steps that introduce claims without supporting premises."""
        return [v for v in self.verdicts if v.verdict == "unsupported_leap"]


def _strip_step_label(text: str) -> str:
    """Remove a leading ``Step N:`` / ``1.`` / ``2)`` prefix from a step.

    This is critical for circular-detection accuracy: the ``_word_overlap``
    comparator uses Jaccard over whitespace-split tokens, and retaining
    ``Step 1:`` / ``Step 2:`` introduces two unique tokens per step that
    dilute the overlap score by ~0.05-0.15, enough to push genuinely
    circular claims below the 0.85 detection threshold.
    """
    return _STEP_LABEL_RE.sub("", text).strip()


def extract_steps(text: str) -> list[ReasoningStep]:
    """Extract reasoning steps from chain-of-thought text.

    Uses the Rust accelerator for extraction when installed, else the bit-exact
    ``text_segmentation`` port (ADR-0001) — identical either way. Every extracted
    step has its ``Step N:``-style prefix stripped so that downstream word-overlap
    comparisons are not diluted by the label tokens.
    """
    raw_steps = _raw_reasoning_steps(text)
    if len(raw_steps) >= 2 or (len(raw_steps) == 1 and raw_steps[0] != text.strip()):
        return [
            ReasoningStep(
                index=i,
                text=_strip_step_label(s),
                is_conclusion=bool(_CONCLUSION_MARKERS.match(s)),
            )
            for i, s in enumerate(raw_steps)
        ]

    # Python fallback
    # Try numbered steps first
    for pattern in _STEP_PATTERNS:
        matches = pattern.findall(text)
        if len(matches) >= 2:
            steps = []
            for i, m in enumerate(matches):
                step_text = m[-1].strip() if isinstance(m, tuple) else m.strip()
                step_text = _strip_step_label(step_text)
                is_conclusion = bool(_CONCLUSION_MARKERS.match(step_text))
                steps.append(
                    ReasoningStep(index=i, text=step_text, is_conclusion=is_conclusion)
                )
            return steps

    # Fallback: split on sentence boundaries with "because" decomposition
    sentences = [s.strip() for s in _raw_split_sentences(text) if len(s.strip()) > 10]
    if not sentences:
        sentences = [
            s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 10
        ]
    if len(sentences) >= 2:
        steps = []
        for i, s in enumerate(sentences):
            s = _strip_step_label(s)
            is_conclusion = bool(_CONCLUSION_MARKERS.match(s))
            steps.append(ReasoningStep(index=i, text=s, is_conclusion=is_conclusion))
        return steps

    return []


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap as a heuristic for logical support.

    Delegates to the shared measured-fast-path helper
    (:mod:`director_ai.core.text_overlap`): pure Python below a large-input
    threshold, the Rust kernel above it. Tokens are case-folded and
    whitespace-split with punctuation retained — the same tokenisation the kernel
    used in production, so the score is unchanged.
    """
    return word_overlap(a, b, logger_name=__name__)


def verify_reasoning_chain(
    text: str,
    score_fn: Callable[[str, str], float] | None = None,
    support_threshold: float = 0.3,
    *,
    check_math: bool = True,
    check_fallacies: bool = True,
    check_citations: bool = False,
) -> ReasoningChainResult:
    """Verify the logical structure of a reasoning chain.

    Parameters
    ----------
    text : str
        Chain-of-thought response text.
    score_fn : callable | None
        Function(premise: str, hypothesis: str) -> float (0=entailed, 1=contradicted).
        If None, uses Jaccard word overlap heuristic.
    support_threshold : float
        Maximum divergence for a step to be considered supported.
    check_math : bool
        When True (default), also verify the explicit arithmetic in the chain
        (``3 + 4 = 8`` and friends) via
        :func:`~director_ai.core.verification.math_consistency.verify_arithmetic`;
        any wrong equation counts as a chain issue.
    check_fallacies : bool
        When True (default), also flag informal-fallacy markers via
        :func:`~director_ai.core.verification.fallacy_detector.detect_fallacies`.
        These are reported on ``fallacies`` as a heuristic signal and, being
        lower-precision than the structural checks, do not affect ``chain_valid``.
    check_citations : bool
        When True (default False, since most chains carry no citations), trace
        claims to their inline citations via
        :func:`~director_ai.core.verification.citation_tracer.trace_citations`;
        the result is reported on ``citation_trace`` and does not affect
        ``chain_valid``.

    Returns
    -------
    ReasoningChainResult
        Per-step verdicts, arithmetic errors, fallacy markers, optional citation
        trace, and chain validity.
    """
    math_errors = verify_arithmetic(text).errors if check_math else []
    fallacies = detect_fallacies(text).matches if check_fallacies else []
    citation_trace = trace_citations(text) if check_citations else None
    steps = extract_steps(text)
    if len(steps) < 2:
        return ReasoningChainResult(
            steps_found=len(steps),
            chain_valid=not math_errors,
            issues_found=len(math_errors),
            math_errors=math_errors,
            fallacies=fallacies,
            citation_trace=citation_trace,
        )

    verdicts: list[ReasoningVerdict] = []
    issues = 0

    # First step is always accepted (it's the premise)
    verdicts.append(
        ReasoningVerdict(
            step_index=0,
            step_text=steps[0].text,
            verdict="supported",
            confidence=1.0,
            reason="Initial premise",
        )
    )

    for i in range(1, len(steps)):
        current = steps[i].text
        best_support = 0.0
        best_premise = ""
        best_divergence = 1.0

        # Check against all prior steps
        for j in range(i):
            prior = steps[j].text
            if score_fn is not None:
                div = score_fn(prior, current)
            else:
                overlap = _word_overlap(prior, current)
                div = 1.0 - overlap
            if div < best_divergence:
                best_divergence = div
                best_support = 1.0 - div
                best_premise = prior

        # Circular: current step is near-identical to a prior step
        for j in range(i):
            if _word_overlap(steps[j].text, current) > 0.85:
                verdicts.append(
                    ReasoningVerdict(
                        step_index=i,
                        step_text=current,
                        verdict="circular",
                        confidence=0.9,
                        reason="Near-identical to a prior step",
                        premise_text=steps[j].text,
                    )
                )
                issues += 1
                break
        else:
            if best_divergence <= support_threshold:
                verdicts.append(
                    ReasoningVerdict(
                        step_index=i,
                        step_text=current,
                        verdict="supported",
                        confidence=best_support,
                        premise_text=best_premise,
                    )
                )
            elif best_support < 0.1:
                verdicts.append(
                    ReasoningVerdict(
                        step_index=i,
                        step_text=current,
                        verdict="unsupported_leap",
                        confidence=1.0 - best_support,
                        reason="No prior step supports this conclusion",
                        premise_text=best_premise,
                    )
                )
                issues += 1
            else:
                verdicts.append(
                    ReasoningVerdict(
                        step_index=i,
                        step_text=current,
                        verdict="non_sequitur",
                        confidence=best_divergence,
                        reason=f"Best support ({best_support:.2f}) below threshold",
                        premise_text=best_premise,
                    )
                )
                issues += 1

    issues += len(math_errors)
    return ReasoningChainResult(
        steps_found=len(steps),
        verdicts=verdicts,
        chain_valid=issues == 0,
        issues_found=issues,
        math_errors=math_errors,
        fallacies=fallacies,
        citation_trace=citation_trace,
    )
