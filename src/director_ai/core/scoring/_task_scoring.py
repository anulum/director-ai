# SPDX-License-Identifier: Apache-2.0
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
from collections.abc import Callable
from typing import Any

from ..accelerator_fallback import sum_i64
from ..mandatory import mandatory_execution
from ..types import ScoringEvidence
from . import _task_accel

logger = logging.getLogger("DirectorAI")

_TERMINAL_SENTENCE_RE = re.compile(r"[.!?]\s*$")

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

    Uses the Rust accelerator when the compiled kernel is present and an
    equivalent pure-Python classifier (identical labels, ADR-0001) otherwise.

    When *response* is provided, a length-ratio heuristic detects
    summarisation even when the prompt lacks explicit keywords.
    A prompt longer than 1 000 chars whose response is shorter than
    30 % of the prompt length is classified as summarisation —
    unless dialogue markers are present.
    """
    if _task_accel._RUST_TASK:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return str(_task_accel.rust_detect_task_type(prompt, response))
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
    nli_scorer: Any,
    prompt: str,
    response: str,
    tenant_id: str,
    *,
    calculate_factual_with_evidence: Callable[
        ..., tuple[float, ScoringEvidence | None]
    ],
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


def weakest_link_claim_divergence(
    nli_scorer: Any,
    premise: str,
    response: str,
) -> tuple[float, list[float], list[str]]:
    """Weakest-link claim divergence of *response* against *premise*.

    Scores every response claim independently with chunked NLI (the
    best-matching source chunk wins per claim) and returns the MAXIMUM
    per-claim divergence: a response is only as supported as its
    least-supported claim. This is the production twin of the WCS-1
    sweep's ``min`` support aggregation (BENCHMARK_REPORT §16) —
    weakest-link support ``= 1 − divergence``.

    Returns ``(divergence, per_claim_divergences, claims)``.
    """
    _coverage, divs, claims = nli_scorer.score_claim_coverage(premise, response)
    return max(divs), divs, claims


def _raw_support_evidence(
    premise: str,
    response: str,
    divergence: float,
    per_claim_divs: list[float],
    claims: list[str],
) -> ScoringEvidence:
    """Package raw-support claim scores as scoring evidence."""
    return ScoringEvidence(
        chunks=[],
        nli_premise=premise,
        nli_hypothesis=response,
        nli_score=divergence,
        per_claim_divergences=per_claim_divs,
        claims=claims,
    )


def dialogue_raw_support_divergence(
    nli_scorer: Any,
    prompt: str,
    response: str,
) -> tuple[float, ScoringEvidence | None]:
    """Raw weakest-link divergence for the dialogue route (WCS-2a).

    No baseline squeeze: the returned divergence is ``1 − support``
    where support is the weakest-link per-claim support of the response
    against the full conversation context. The review gate compares the
    resulting coherence (= support) directly against the dialogue
    support operating point — matched-FPR calibrated per deployment
    (:mod:`director_ai.core.calibration.operating_points`). The WCS-1
    E2E proof showed the 0.80-baseline squeeze absorbs evidence gains
    (decisions identical at catch 4.5 %); the raw operating point lifts
    measured catch to 27 % at the same FPR (BENCHMARK_REPORT §16).
    """
    divergence, per_claim, claims = weakest_link_claim_divergence(
        nli_scorer, prompt, response
    )
    return divergence, _raw_support_evidence(
        prompt, response, divergence, per_claim, claims
    )


def summarization_factual_divergence(
    nli_scorer: Any,
    prompt: str,
    response: str,
    tenant_id: str,
    *,
    calculate_factual_with_evidence: Callable[
        ..., tuple[float, ScoringEvidence | None]
    ],
    fact_inner_agg: str = "max",
    fact_outer_agg: str = "max",
    premise_ratio: float = 0.4,
    claim_coverage_enabled: bool = True,
    claim_support_threshold: float = 0.6,
    claim_coverage_alpha: float = 0.4,
    baseline: float = 0.20,
    get_minicheck_scorer: Callable[[], Any] | None = None,
    premise_chars: int = 0,
    aggregation: str = "blend",
) -> tuple[float, ScoringEvidence | None]:
    """Bidirectional NLI + claim coverage for summarisation.

    Layer A: bidirectional FactCG NLI with baseline calibration.
    Layer M (MiniCheck): sentence-level MiniCheck scoring + coverage.
    Layer C (fallback): FactCG claim decomposition + coverage.

    Final divergence = alpha * (1 - coverage) + (1 - alpha) * layer_a.

    With ``aggregation="weakest_link"`` the blend is replaced by the
    raw weakest-link claim divergence against the source document —
    the only WCS-1 configuration that improved on both HaluEval and
    RAGTruth (BENCHMARK_REPORT §16). The review gate then compares the
    resulting coherence (= support) against the summarisation support
    operating point instead of a composite-coherence threshold.

    Parameters
    ----------
    nli_scorer : NLIScorer
        The NLI scorer instance.
    get_minicheck_scorer : callable | None
        Returns MiniCheck NLIScorer or None.
    premise_chars : int
        Source-document budget for the coverage layers; ``0`` (default)
        passes the WHOLE document and lets each backend chunk long inputs
        itself. The pre-WCS-1 behaviour was a 3000-char truncation, which
        the 2026-07-15 evidence sweep showed hides late-document evidence
        on both HaluEval and RAGTruth (BENCHMARK_REPORT §16).
    aggregation : str
        ``"blend"`` (default) keeps the coverage/Layer-A alpha blend;
        ``"weakest_link"`` returns the raw weakest-link claim divergence
        (WCS-2a) and skips the blend layers entirely.
    """
    coverage_premise = prompt[:premise_chars] if premise_chars > 0 else prompt

    if aggregation == "weakest_link":
        mc_scorer = get_minicheck_scorer() if get_minicheck_scorer else None
        if mc_scorer is not None:
            _coverage, per_claim, claims = minicheck_claim_coverage(
                mc_scorer, coverage_premise, response
            )
            divergence = max(per_claim) if per_claim else 0.0
        else:
            divergence, per_claim, claims = weakest_link_claim_divergence(
                nli_scorer, coverage_premise, response
            )
        return divergence, _raw_support_evidence(
            coverage_premise, response, divergence, per_claim, claims
        )

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
            mc_scorer, coverage_premise, response
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
        try:
            coverage, per_claim_divs, claims, attributions = (
                nli_scorer.score_claim_coverage_with_attribution(
                    coverage_premise,
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
    mc_scorer: Any,
    source: str,
    summary: str,
) -> tuple[float, list[float], list[str]]:
    """Score each summary sentence with MiniCheck, return coverage.

    Returns (coverage, per_sentence_divergences, sentences).
    Coverage = fraction of sentences with divergence < 0.5.
    """
    if _task_accel._RUST_TASK:
        try:
            sentences = [
                _normalize_claim_sentence(s)
                for s in _task_accel.rust_split_sentences(summary)
                if s.strip()
            ]
        except Exception:
            sentences = []
    else:
        sentences = []
    if not sentences:
        try:
            from nltk.tokenize import sent_tokenize

            sentences = sent_tokenize(summary)
        except (ImportError, LookupError):
            sentences = [s.strip() + "." for s in summary.split(".") if s.strip()]
    if not sentences:
        return 1.0, [], []

    divs = [mc_scorer.score(source, sent) for sent in sentences]
    if _task_accel._RUST_TASK:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            coverage, _supported = _task_accel.rust_coverage_from_divergences(
                [float(d) for d in divs],
                0.5,
            )
            return float(coverage), divs, sentences
    supported = _sum_int([1 if d < 0.5 else 0 for d in divs])
    coverage = supported / len(sentences)
    return coverage, divs, sentences


def _normalize_claim_sentence(sentence: str) -> str:
    """Return a stripped claim sentence with stable terminal punctuation."""
    normalized = sentence.strip()
    if normalized and not _TERMINAL_SENTENCE_RE.search(normalized):
        return f"{normalized}."
    return normalized


def _sum_int(values: list[int]) -> int:
    if _task_accel._RUST_TASK:
        with mandatory_execution(__name__, component="mandatory accelerated path"):
            return int(_task_accel.rust_sum_i64(values))
    # kernel absent: use the bit-exact pure-Python fallback (ADR-0001)
    return sum_i64(values)
