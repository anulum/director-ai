#!/usr/bin/env python3
"""
Evidence inspection demo — see exactly why Director-AI rejected a response.

Every CoherenceScore carries an `evidence` field containing:
- Top-K ground truth chunks with relevance distances
- NLI premise (retrieved context) and hypothesis (LLM output)
- NLI contradiction score

Usage::

    python examples/evidence_inspection_demo.py
"""

from __future__ import annotations

from director_ai import CoherenceScorer, GroundTruthStore


def main():
    store = GroundTruthStore()
    store.add("refund", "Refunds are available within 30 days of purchase.")
    store.add("warranty", "All products have a 1-year warranty.")
    store.add("shipping", "Free shipping on orders over $50.")

    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store, use_nli=False)

    # ── Case 1: Truthful response (should pass) ──────────────────────
    prompt = "What is the refund policy?"
    response = "Refunds are available within 30 days of purchase."
    approved, score = scorer.review(prompt, response)

    print("=" * 60)
    print("CASE 1: Truthful response")
    print(f"  Approved: {approved}")
    print(f"  Score:    {score.score:.3f}")
    print(f"  Warning:  {score.warning}")
    _print_evidence(score)

    # ── Case 2: Hallucinated response (should fail) ──────────────────
    prompt = "What is the refund policy?"
    response = "We offer a 90-day money back guarantee on all items."
    approved, score = scorer.review(prompt, response)

    print("\n" + "=" * 60)
    print("CASE 2: Hallucinated response")
    print(f"  Approved: {approved}")
    print(f"  Score:    {score.score:.3f}")
    _print_evidence(score)

    # ── Case 3: Partially correct ─────────────────────────────────────
    prompt = "Tell me about shipping."
    response = "Free shipping on orders over $100 with express delivery."
    approved, score = scorer.review(prompt, response)

    print("\n" + "=" * 60)
    print("CASE 3: Partially correct")
    print(f"  Approved: {approved}")
    print(f"  Score:    {score.score:.3f}")
    _print_evidence(score)


def _print_evidence(score):
    ev = score.evidence
    if not ev:
        print("  Evidence: (none — no ground truth matched)")
        return

    print("  Evidence:")
    print(f"    NLI premise:    {ev.nli_premise[:80]}")
    print(f"    NLI hypothesis: {ev.nli_hypothesis[:80]}")
    print(f"    NLI score:      {ev.nli_score:.3f}")
    print(f"    Chunks ({len(ev.chunks)}):")
    for i, chunk in enumerate(ev.chunks):
        print(f"      [{i}] dist={chunk.distance:.3f}  src={chunk.source}")
        print(f"          {chunk.text[:70]}")


if __name__ == "__main__":
    main()
