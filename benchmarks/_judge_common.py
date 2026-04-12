# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — Shared utilities for AggreFact LLM-as-judge scripts
"""Constants and helpers shared across ``benchmarks/gemma_aggrefact_*.py``
and related LLM-as-judge benchmark scripts.

Extracted 2026-04-12 to eliminate 6-way duplication of
``compute_balanced_accuracy``, ``parse_response``,
``DATASET_TO_FAMILY``, and the three routed prompt templates.

Import convention::

    from benchmarks._judge_common import (
        DATASET_TO_FAMILY,
        PROMPTS,
        compute_balanced_accuracy,
        parse_response,
    )
"""

from __future__ import annotations

import re

# ── Per-task-family prompt routing ───────────────────────────────────────

PROMPT_SUMM = (
    "You are a careful summarisation evaluator. Decide if the SUMMARY "
    "claim is fully supported by the SOURCE document. Be strict: any "
    "added detail, paraphrased number, or unsupported entity is "
    "NOT_SUPPORTED.\n\n"
    "SOURCE:\n{premise}\n\n"
    "SUMMARY CLAIM:\n{hypothesis}\n\n"
    "Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."
)

PROMPT_RAG = (
    "You are a fact-checking assistant for retrieval-augmented "
    "generation outputs. Decide if the CLAIM is fully grounded in "
    "the retrieved CONTEXT. Reject claims that depend on world "
    "knowledge not present in the CONTEXT.\n\n"
    "CONTEXT:\n{premise}\n\n"
    "CLAIM:\n{hypothesis}\n\n"
    "Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."
)

PROMPT_CLAIM = (
    "You are a fact-checking assistant. Decide if the CLAIM is fully "
    "supported by the CONTEXT. Focus on whether every assertion in "
    "the CLAIM matches the CONTEXT verbatim or by direct entailment.\n\n"
    "CONTEXT:\n{premise}\n\n"
    "CLAIM:\n{hypothesis}\n\n"
    "Answer with exactly one word: SUPPORTED or NOT_SUPPORTED."
)

DATASET_TO_FAMILY: dict[str, str] = {
    "AggreFact-CNN": "summ",
    "AggreFact-XSum": "summ",
    "TofuEval-MediaS": "summ",
    "TofuEval-MeetB": "summ",
    "RAGTruth": "rag",
    "ClaimVerify": "rag",
    "FactCheck-GPT": "rag",
    "ExpertQA": "rag",
    "Reveal": "claim",
    "Lfqa": "claim",
    "Wice": "claim",
}

PROMPTS: dict[str, str] = {
    "summ": PROMPT_SUMM,
    "rag": PROMPT_RAG,
    "claim": PROMPT_CLAIM,
}

#: The 11 AggreFact datasets in the order they appear on the leaderboard.
AGGREFACT_DATASETS: tuple[str, ...] = tuple(sorted(DATASET_TO_FAMILY))


# ── Response parsing ─────────────────────────────────────────────────────


def parse_response(text: str) -> int:
    """Parse an LLM verdict into a binary label.

    Returns 1 (supported), 0 (not supported), or -1 (unparseable).
    Handles common model-output variations:

    - ``SUPPORTED``, ``NOT_SUPPORTED``, ``NOT SUPPORTED``,
      ``NOT-SUPPORTED`` (case-insensitive)
    - ``Yes`` / ``No`` (prefix match, case-insensitive)
    - ``True`` / ``False`` (prefix match, case-insensitive)
    - Anything else → -1 (unknown)
    """
    t = text.strip().upper()
    if "NOT_SUPPORTED" in t or "NOT SUPPORTED" in t or "NOT-SUPPORTED" in t:
        return 0
    if "SUPPORTED" in t:
        return 1
    if t.startswith("YES") or t.startswith("TRUE"):
        return 1
    if t.startswith("NO") or t.startswith("FALSE"):
        return 0
    return -1


# ── Balanced accuracy ────────────────────────────────────────────────────


# ── HiSS decomposition helpers ──────────────────────────────────────────

DECOMPOSE_PROMPT = """Break the CLAIM into 1-4 atomic sub-claims that can be checked independently.
Return them as a numbered list, one sub-claim per line. Do not add explanation. Do not repeat the original claim.

CLAIM:
{claim}

Sub-claims:
1."""

#: Match a leading list marker: "1.", "1)", "- ", "* "
LIST_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$")


def parse_subclaims(raw: str, original_claim: str, max_n: int = 5) -> list[str]:
    """Extract atomic sub-claims from a decomposition response.

    If no list items are found, returns ``[original_claim]`` as a fallback.
    Filters out meta-labels like "sub-claims" or "claim".
    """
    text = "1. " + raw if not raw.lstrip().startswith(("1", "-", "*")) else raw
    out: list[str] = []
    for line in text.splitlines():
        m = LIST_LINE_RE.match(line)
        if m:
            sub = m.group(1).strip()
            if sub and sub.lower() not in {"sub-claims", "sub-claim", "claim"}:
                out.append(sub)
        if len(out) >= max_n:
            break
    if not out:
        out = [original_claim]
    return out


# ── Per-dataset / per-family aggregation ────────────────────────────────


def aggregate_per_dataset(
    preds: list[int],
    labels: list[int],
    datasets: list[str],
) -> dict[str, dict]:
    """Group predictions by dataset and compute BA for each."""
    from collections import defaultdict

    by_ds: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for p_, l_, d_ in zip(preds, labels, datasets, strict=True):
        by_ds[d_][0].append(p_)
        by_ds[d_][1].append(l_)
    return {
        ds: {"samples": len(l_), "balanced_accuracy": compute_balanced_accuracy(p_, l_)}
        for ds, (p_, l_) in by_ds.items()
    }


def aggregate_per_family(
    preds: list[int],
    labels: list[int],
    families: list[str],
) -> dict[str, dict]:
    """Group predictions by task family and compute BA for each."""
    from collections import defaultdict

    by_fam: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for p_, l_, f_ in zip(preds, labels, families, strict=True):
        by_fam[f_][0].append(p_)
        by_fam[f_][1].append(l_)
    return {
        f: {"samples": len(l_), "balanced_accuracy": compute_balanced_accuracy(p_, l_)}
        for f, (p_, l_) in by_fam.items()
    }


# ── Balanced accuracy ────────────────────────────────────────────────────


def compute_balanced_accuracy(
    preds: list[int],
    labels: list[int],
) -> float:
    """Two-class balanced accuracy with unknown (-1) filtering.

    Predictions equal to -1 are silently dropped from the count.
    Returns 0.0 when either class has zero predictions after
    filtering.

    This is **sample-pooled BA** (one number across the full pool).
    For **per-dataset-mean BA** (the AggreFact leaderboard
    convention), call this function once per dataset and average the
    results.
    """
    pos = neg = tp = tn = 0
    for p, lab in zip(preds, labels, strict=True):
        if p < 0:
            continue
        if lab == 1:
            pos += 1
            if p == 1:
                tp += 1
        else:
            neg += 1
            if p == 0:
                tn += 1
    if pos == 0 or neg == 0:
        return 0.0
    return (tp / pos + tn / neg) / 2
