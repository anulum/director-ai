#!/usr/bin/env python3
"""
Customer support guardrail with Director-AI.

Requires: pip install director-ai
"""
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("refund policy", "Refunds within 30 days of purchase.")
store.add("shipping", "Standard shipping 5-7 business days.")
store.add("pricing", "Pro plan $49/month, Enterprise $199/month.")

scorer = CoherenceScorer(threshold=0.55, ground_truth_store=store)

cases = [
    ("Refund policy?",
     "We offer refunds within 30 days of purchase.", True),
    ("Refund policy?",
     "Full refunds anytime within 90 days.", False),
    ("How much is the Pro plan?",
     "The Pro plan costs $49 per month.", True),
    ("How much is the Pro plan?",
     "The Pro plan is free forever.", False),
]

for prompt, response, expected_approve in cases:
    approved, score = scorer.review(prompt, response)
    status = "PASS" if approved == expected_approve else "FAIL"
    print(f"[{status}] approved={approved} score={score.score:.2f} | {response[:60]}")
    if score.evidence:
        for chunk in score.evidence.chunks:
            print(f"       evidence: {chunk.text[:80]}")
