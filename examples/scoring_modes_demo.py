# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-Class AI — 30-second scoring modes demo
"""Three scoring modes against the same hallucinated answer.

Compares prompt-only, threshold-only, and KB-grounded verification on
a shared fixture so the differences in catch rate are visible in one
run.

Usage::

    python examples/scoring_modes_demo.py
"""

from __future__ import annotations

import time

FACTS = {
    "pricing": "Team Plan costs $19 per user per month, up to 25 users.",
    "support": "Phone support is Enterprise only.",
    "trial": "All paid plans include a 14-day free trial.",
    "compliance": "SOC 2 Type II and ISO 27001 certified.",
}

SOURCE_TEXT = (
    "Team Plan costs $19 per user per month, up to 25 users. "
    "Phone support is Enterprise only. "
    "All paid plans include a 14-day free trial. "
    "SOC 2 Type II and ISO 27001 certified."
)

QA_PAIRS = [
    {
        "q": "How much does the Team plan cost?",
        "a": "The Team Plan costs $19 per user per month and supports up to 25 users.",
        "label": "correct",
    },
    {
        "q": "Is phone support available on Team?",
        "a": "Yes, phone support is available for all paid plans including Team.",
        "label": "hallucinated",
        "why": "Phone support is Enterprise only",
    },
    {
        "q": "What compliance certs do you have?",
        "a": "We are SOC 2 Type II, ISO 27001, HIPAA, and FedRAMP certified.",
        "label": "hallucinated",
        "why": "HIPAA and FedRAMP are fabricated",
    },
    {
        "q": "Is there a free trial?",
        "a": "Yes, all paid plans include a 30-day free trial.",
        "label": "hallucinated",
        "why": "14-day trial, not 30-day",
    },
]


def run_mode_1_threshold_only():
    """Mode 1: Simple threshold (what most teams start with)."""
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    for k, v in FACTS.items():
        store.add(k, v)

    scorer = CoherenceScorer(
        threshold=0.5,
        ground_truth_store=store,
        use_nli=False,
    )

    print("MODE 1: Heuristic threshold (fast, no NLI)")
    print("-" * 60)
    tp = fp = tn = fn = 0
    t0 = time.perf_counter()
    for qa in QA_PAIRS:
        approved, score = scorer.review(qa["q"], qa["a"])
        is_bad = qa["label"] == "hallucinated"
        detected = not approved
        if is_bad and detected:
            tp += 1
        elif is_bad and not detected:
            fn += 1
        elif not is_bad and detected:
            fp += 1
        else:
            tn += 1
        tag = (
            "CAUGHT"
            if (is_bad and detected)
            else "MISSED"
            if (is_bad and not detected)
            else "FALSE+"
            if (not is_bad and detected)
            else "OK"
        )
        print(f"  [{tag}] score={score.score:.3f}  {qa['a'][:60]}")
    elapsed = (time.perf_counter() - t0) * 1000
    catch = tp / (tp + fn) if (tp + fn) else 0
    print(f"  Catch: {tp}/{tp + fn} ({catch:.0%})  FP: {fp}  Time: {elapsed:.0f}ms")
    print()
    return catch


def run_mode_2_coherence_scorer():
    """Mode 2: Director-AI default (heuristic + KB grounding)."""
    from director_ai.core import CoherenceScorer, GroundTruthStore

    store = GroundTruthStore()
    for k, v in FACTS.items():
        store.add(k, v)

    scorer = CoherenceScorer(
        threshold=0.3,
        ground_truth_store=store,
        use_nli=False,
    )

    print("MODE 2: Director-AI default (KB-grounded, threshold 0.3)")
    print("-" * 60)
    tp = fp = tn = fn = 0
    t0 = time.perf_counter()
    for qa in QA_PAIRS:
        approved, score = scorer.review(qa["q"], qa["a"])
        is_bad = qa["label"] == "hallucinated"
        detected = not approved
        if is_bad and detected:
            tp += 1
        elif is_bad and not detected:
            fn += 1
        elif not is_bad and detected:
            fp += 1
        else:
            tn += 1
        tag = (
            "CAUGHT"
            if (is_bad and detected)
            else "MISSED"
            if (is_bad and not detected)
            else "FALSE+"
            if (not is_bad and detected)
            else "OK"
        )
        print(f"  [{tag}] score={score.score:.3f}  {qa['a'][:60]}")
    elapsed = (time.perf_counter() - t0) * 1000
    catch = tp / (tp + fn) if (tp + fn) else 0
    print(f"  Catch: {tp}/{tp + fn} ({catch:.0%})  FP: {fp}  Time: {elapsed:.0f}ms")
    print()
    return catch


def run_mode_3_verified():
    """Mode 3: Director-AI claim-level verification."""
    from director_ai.core.scoring.verified_scorer import VerifiedScorer

    vs = VerifiedScorer()

    print("MODE 3: Director-AI per-claim verification (atomic)")
    print("-" * 60)
    tp = fp = tn = fn = 0
    t0 = time.perf_counter()
    for qa in QA_PAIRS:
        vr = vs.verify(qa["a"], SOURCE_TEXT, atomic=True)
        is_bad = qa["label"] == "hallucinated"
        detected = not vr.approved
        if is_bad and detected:
            tp += 1
        elif is_bad and not detected:
            fn += 1
        elif not is_bad and detected:
            fp += 1
        else:
            tn += 1
        tag = (
            "CAUGHT"
            if (is_bad and detected)
            else "MISSED"
            if (is_bad and not detected)
            else "FALSE+"
            if (not is_bad and detected)
            else "OK"
        )
        print(
            f"  [{tag}] score={vr.overall_score:.3f}  claims={len(vr.claims)}  {qa['a'][:50]}"
        )
        for c in vr.claims:
            markers = {
                "supported": "+",
                "contradicted": "X",
                "fabricated": "!",
                "unverifiable": "?",
            }
            m = markers.get(c.verdict, "?")
            print(f'    [{m}] {c.verdict}: "{c.claim[:55]}"')
    elapsed = (time.perf_counter() - t0) * 1000
    catch = tp / (tp + fn) if (tp + fn) else 0
    print(f"  Catch: {tp}/{tp + fn} ({catch:.0%})  FP: {fp}  Time: {elapsed:.0f}ms")
    print()
    return catch


def main():
    print("=" * 60)
    print("  Director-AI — scoring modes demo")
    print("  Same 4 Q&A pairs, 3 hallucinated, 3 scoring modes")
    print("=" * 60)
    print()

    c1 = run_mode_1_threshold_only()
    c2 = run_mode_2_coherence_scorer()
    c3 = run_mode_3_verified()

    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Mode 1 (threshold only):     {c1:.0%} catch rate")
    print(f"  Mode 2 (KB-grounded):        {c2:.0%} catch rate")
    print(f"  Mode 3 (per-claim verified): {c3:.0%} catch rate")
    print()
    print("  Mode 3 shows WHY each claim failed — not just a score.")
    print("  That's the difference between a guardrail and an audit trail.")
    print()


if __name__ == "__main__":
    main()
