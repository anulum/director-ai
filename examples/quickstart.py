#!/usr/bin/env python3
"""
Director-AI quickstart — guard any LLM in 10 lines.

    pip install director-ai
    python examples/quickstart.py

For real NLI-based scoring (detects contradiction in any domain):
    pip install director-ai[nli]
    scorer = CoherenceScorer(use_nli=True, threshold=0.6, ground_truth_store=store)
"""

from director_ai.core import CoherenceScorer, GroundTruthStore

# 1. Load your facts
store = GroundTruthStore()
store.facts["sky color"] = "blue"

# 2. Create a scorer
scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

# 3. Check LLM outputs
tests = [
    ("What color is the sky?", "The sky is blue on a clear day."),
    ("What color is the sky?", "The sky is green, obviously."),
]

for prompt, llm_output in tests:
    approved, score = scorer.review(prompt, llm_output)
    status = "PASS" if approved else "BLOCKED"
    print(f"[{status}] {llm_output}")
    print(
        f"  coherence={score.score:.3f}"
        f"  h_logical={score.h_logical:.2f}"
        f"  h_factual={score.h_factual:.2f}"
    )
    print()
