# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Task-Specific Scoring Paths

"""Dialogue and summarisation scoring helpers for CoherenceScorer.

Extracted from scorer.py to reduce module size.  These functions
implement the bidirectional NLI + baseline calibration paths for
dialogue and summarisation tasks.
"""

from __future__ import annotations

import logging
import re

from ..types import ScoringEvidence

try:
    from backfire_kernel import rust_detect_task_type

    _RUST_TASK = True
except ImportError:
    _RUST_TASK = False

logger = logging.getLogger("DirectorAI")

# Dialogue detection: ≥2 speaker-turn markers → dialogue task.
_DIALOGUE_TURN_RE = re.compile(
    r"(?:^|\s)(?:"
    r"(?:User|Human|Customer|Student|Interviewer|Speaker"
    r"|Assistant|AI|Bot|Agent|Interviewee|System)"
    r"[\s\d]*:"
    r"|\[(?:User|Human|Assistant|AI|System)\]"
    r")",
    re.IGNORECASE,
)


def detect_task_type(prompt: str, response: str = "") -> str:
    """Detect task type from prompt content and length ratio.

    Returns one of: ``"dialogue"``, ``"summarization"``, ``"rag"``,
    ``"fact_check"``, ``"qa"``, or ``"default"``.

    Uses Rust accelerator when available, Python fallback otherwise.

    When *response* is provided, a length-ratio heuristic detects
    summarisation even when the prompt lacks explicit keywords.
    A prompt longer than 1 000 chars whose response is shorter than
    30 % of the prompt length is classified as summarisation —
    unless dialogue markers are present.
    """
    if _RUST_TASK:
        return str(rust_detect_task_type(prompt, response))
    matches = _DIALOGUE_TURN_RE.findall(prompt)
    if len(matches) >= 2:
        return "dialogue"

    lower = prompt.lower()
    if any(
        kw in lower for kw in ("summarize", "summary", "summarise", "tldr", "abstract")
    ):
        return "summarization"

    if (
        response
        and len(prompt) > 1000
        and len(response) > 20
        and len(response) / len(prompt) < 0.30
    ):
        return "summarization"

    if any(
        kw in lower
        for kw in (
            "based on the context",
            "based on the following",
            "given the document",
            "given the passage",
            "retrieved",
            "source document",
            "reference text",
        )
    ):
        return "rag"
    if any(
        kw in lower for kw in ("verify", "fact-check", "is it true", "claim", "support")
    ):
        return "fact_check"
    if "?" in prompt or any(
        kw in lower for kw in ("answer the question", "based on the", "according to")
    ):
        return "qa"
    return "default"


def dialogue_factual_divergence(
    nli_scorer,
    prompt: str,
    response: str,
    tenant_id: str,
    *,
    calculate_factual_with_evidence,
    baseline: float = 0.80,
) -> tuple[float, ScoringEvidence | None]:
    """Bidirectional NLI scoring with baseline calibration for dialogue.

    1. Scores both directions (forward + reverse).
    2. Takes the **minimum** (most lenient direction).
    3. Applies baseline calibration to shift out expected divergence.

    Parameters
    ----------
    nli_scorer : NLIScorer
        The NLI scorer instance (must have model loaded).
    prompt : str
        Conversation context.
    response : str
        AI response to score.
    tenant_id : str
        Tenant isolation key.
    calculate_factual_with_evidence : callable
        Bound method ``scorer.calculate_factual_divergence_with_evidence``.
    baseline : float
        Expected NLI divergence for correct dialogue (default 0.80).
    """
    # Forward pass: full evidence path
    h_fact_fwd, evidence = calculate_factual_with_evidence(
        prompt,
        response,
        tenant_id,
        _inner_agg="min",
        _outer_agg="mean",
    )

    # Reverse pass: does the response support the context?
    h_fact_rev, _ = nli_scorer.score_chunked(
        response,
        prompt,
        inner_agg="min",
        outer_agg="mean",
        premise_ratio=0.4,
    )

    # Bidirectional minimum (most lenient direction)
    raw_div = min(h_fact_fwd, h_fact_rev)

    # Baseline calibration: shift expected dialogue divergence to 0
    denom = 1.0 - baseline
    adjusted = max(0.0, (raw_div - baseline) / denom) if denom > 1e-9 else raw_div

    return adjusted, evidence


def summarization_factual_divergence(
    nli_scorer,
    prompt: str,
    response: str,
    tenant_id: str,
    *,
    calculate_factual_with_evidence,
    fact_inner_agg: str = "max",
    fact_outer_agg: str = "max",
    premise_ratio: float = 0.4,
    claim_coverage_enabled: bool = True,
    claim_support_threshold: float = 0.6,
    claim_coverage_alpha: float = 0.4,
    baseline: float = 0.20,
    get_minicheck_scorer=None,
) -> tuple[float, ScoringEvidence | None]:
    """Bidirectional NLI + claim coverage for summarisation.

    Layer A: bidirectional FactCG NLI with baseline calibration.
    Layer M (MiniCheck): sentence-level MiniCheck scoring + coverage.
    Layer C (fallback): FactCG claim decomposition + coverage.

    Final divergence = alpha * (1 - coverage) + (1 - alpha) * layer_a.

    Parameters
    ----------
    nli_scorer : NLIScorer
        The NLI scorer instance.
    get_minicheck_scorer : callable | None
        Returns MiniCheck NLIScorer or None.
    """
    # Layer A: bidirectional FactCG NLI
    h_fact_fwd, evidence = calculate_factual_with_evidence(
        prompt,
        response,
        tenant_id,
        _inner_agg=fact_inner_agg,
        _outer_agg=fact_outer_agg,
    )

    h_fact_rev, _ = nli_scorer.score_chunked(
        response,
        prompt,
        inner_agg="min",
        outer_agg="mean",
        premise_ratio=premise_ratio,
    )

    raw_div = min(h_fact_fwd, h_fact_rev)

    if baseline > 0.0:
        layer_a = max(0.0, (raw_div - baseline) / (1.0 - baseline))
    else:
        layer_a = raw_div

    # Layer M: MiniCheck sentence-level scoring (preferred over Layer C).
    mc_scorer = get_minicheck_scorer() if get_minicheck_scorer else None
    if mc_scorer is not None:
        coverage, per_claim_divs, claims = minicheck_claim_coverage(
            mc_scorer, prompt[:3000], response
        )
        mc_alpha = 0.6
        adjusted = mc_alpha * (1.0 - coverage) + (1.0 - mc_alpha) * layer_a

        if evidence is not None:
            evidence.claim_coverage = coverage
            evidence.per_claim_divergences = per_claim_divs
            evidence.claims = claims
        return adjusted, evidence

    # Layer C (fallback): FactCG claim decomposition + coverage.
    if claim_coverage_enabled:
        truncated_premise = prompt[:3000]
        try:
            coverage, per_claim_divs, claims, attributions = (
                nli_scorer.score_claim_coverage_with_attribution(
                    truncated_premise,
                    response,
                    support_threshold=claim_support_threshold,
                )
            )
        except (RuntimeError, Exception) as exc:
            if "out of memory" in str(exc).lower():
                import torch

                torch.cuda.empty_cache()
                logger.warning("OOM in claim coverage — falling back to layer A only")
                return layer_a, evidence
            raise
        alpha = claim_coverage_alpha
        adjusted = alpha * (1.0 - coverage) + (1.0 - alpha) * layer_a

        if evidence is not None:
            evidence.claim_coverage = coverage
            evidence.per_claim_divergences = per_claim_divs
            evidence.claims = claims
            evidence.attributions = attributions
    else:
        adjusted = layer_a

    return adjusted, evidence


def minicheck_claim_coverage(
    mc_scorer,
    source: str,
    summary: str,
) -> tuple[float, list[float], list[str]]:
    """Score each summary sentence with MiniCheck, return coverage.

    Returns (coverage, per_sentence_divergences, sentences).
    Coverage = fraction of sentences with divergence < 0.5.
    """
    try:
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(summary)
    except (ImportError, LookupError):
        sentences = [s.strip() + "." for s in summary.split(".") if s.strip()]
    if not sentences:
        return 1.0, [], []

    divs = [mc_scorer.score(source, sent) for sent in sentences]
    supported = sum(1 for d in divs if d < 0.5)
    coverage = supported / len(sentences) if sentences else 1.0
    return coverage, divs, sentences
