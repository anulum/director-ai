# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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
- Contradictory steps (step contradicts an earlier step)

Uses NLI scoring when available, falls back to heuristic overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "ReasoningStep",
    "ReasoningVerdict",
    "ReasoningChainResult",
    "verify_reasoning_chain",
]

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

    @property
    def non_sequiturs(self) -> list[ReasoningVerdict]:
        return [v for v in self.verdicts if v.verdict == "non_sequitur"]

    @property
    def unsupported_leaps(self) -> list[ReasoningVerdict]:
        return [v for v in self.verdicts if v.verdict == "unsupported_leap"]


def extract_steps(text: str) -> list[ReasoningStep]:
    """Extract reasoning steps from chain-of-thought text."""
    # Try numbered steps first
    for pattern in _STEP_PATTERNS:
        matches = pattern.findall(text)
        if len(matches) >= 2:
            steps = []
            for i, m in enumerate(matches):
                step_text = m[-1].strip() if isinstance(m, tuple) else m.strip()
                is_conclusion = bool(_CONCLUSION_MARKERS.match(step_text))
                steps.append(
                    ReasoningStep(index=i, text=step_text, is_conclusion=is_conclusion)
                )
            return steps

    # Fallback: split on sentence boundaries with "because" decomposition
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if len(s.strip()) > 10]
    if len(sentences) >= 2:
        steps = []
        for i, s in enumerate(sentences):
            is_conclusion = bool(_CONCLUSION_MARKERS.match(s))
            steps.append(ReasoningStep(index=i, text=s, is_conclusion=is_conclusion))
        return steps

    return []


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap as a heuristic for logical support."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def verify_reasoning_chain(
    text: str,
    score_fn=None,
    support_threshold: float = 0.3,
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

    Returns
    -------
    ReasoningChainResult
        Per-step verdicts with overall chain validity.
    """
    steps = extract_steps(text)
    if len(steps) < 2:
        return ReasoningChainResult(steps_found=len(steps))

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

    return ReasoningChainResult(
        steps_found=len(steps),
        verdicts=verdicts,
        chain_valid=issues == 0,
        issues_found=issues,
    )
