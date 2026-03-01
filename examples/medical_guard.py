#!/usr/bin/env python3
"""
Medical domain guardrail with Director-AI.

Requires: pip install director-ai
"""
from director_ai.core import CoherenceScorer, GroundTruthStore

store = GroundTruthStore()
store.add("aspirin", "Not for children under 16 — Reye's syndrome risk.")
store.add("blood pressure", "Normal is below 120/80 mmHg.")
store.add("diabetes", "Type 2 diagnosed when fasting glucose > 126 mg/dL.")

scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

cases = [
    ("Safe to give aspirin to a child?",
     "Aspirin should not be given to children under 16.", True),
    ("Safe to give aspirin to a child?",
     "Yes, aspirin is safe for all ages.", False),
    ("What is normal blood pressure?",
     "Normal blood pressure is below 120/80 mmHg.", True),
    ("What is normal blood pressure?",
     "Normal blood pressure is 180/110 mmHg.", False),
]

for prompt, response, expected_approve in cases:
    approved, score = scorer.review(prompt, response)
    status = "PASS" if approved == expected_approve else "FAIL"
    print(f"[{status}] approved={approved} score={score.score:.2f} | {response[:60]}")
    if score.evidence:
        print(f"       evidence: {score.evidence.chunks[0].text[:80]}")
