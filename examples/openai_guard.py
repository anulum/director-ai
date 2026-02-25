#!/usr/bin/env python3
"""
Guard OpenAI chat completions with Director-AI.

Requires:
    pip install director-ai openai

Usage:
    export OPENAI_API_KEY=sk-...
    python examples/openai_guard.py
"""

from __future__ import annotations

import os

from openai import OpenAI

from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel


def score_and_gate():
    """Score a complete response, reject if below threshold."""
    client = OpenAI()

    store = GroundTruthStore()
    store.facts["refund policy"] = "within 30 days"
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is your refund policy?"}],
    )
    answer = response.choices[0].message.content

    approved, score = scorer.review("What is your refund policy?", answer)
    print(f"Answer: {answer}")
    print(f"Coherence: {score.score:.3f}  Approved: {approved}")
    if not approved:
        print("BLOCKED — response contradicts knowledge base")


def streaming_guard():
    """Monitor a streaming response token-by-token; halt on incoherence."""
    client = OpenAI()

    store = GroundTruthStore()
    store.facts["capital of France"] = "Paris"
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
    kernel = StreamingKernel(hard_limit=0.35, window_size=5, window_threshold=0.45)

    prompt = "What is the capital of France?"
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    # Collect tokens from the OpenAI stream
    tokens = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            tokens.append(delta)

    accumulated = ""

    def coherence_cb(token: str) -> float:
        nonlocal accumulated
        accumulated += token
        _, sc = scorer.review(prompt, accumulated)
        return sc.score

    session = kernel.stream_tokens(iter(tokens), coherence_cb)

    for event in session.events:
        print(event.token, end="", flush=True)
    print()

    if session.halted:
        print(f"\nHALTED at token {session.halt_index}: {session.halt_reason}")
    else:
        print(f"\nApproved — avg coherence {session.avg_coherence:.3f}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example.")
        print("Showing code structure only — see docstrings for usage.")
    else:
        score_and_gate()
        print("\n---\n")
        streaming_guard()
