#!/usr/bin/env python3
"""
Guard Ollama responses with Director-AI.

Requires:
    pip install director-ai requests

Usage:
    ollama serve  # in another terminal
    python examples/ollama_guard.py
"""

from __future__ import annotations

import requests

from director_ai.core import CoherenceScorer, GroundTruthStore, StreamingKernel

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2"


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def score_response():
    """Score a complete Ollama response."""
    store = GroundTruthStore()
    store.facts["boiling point"] = "100 degrees Celsius"
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)

    prompt = "At what temperature does water boil?"
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    answer = resp.json()["response"]

    approved, score = scorer.review(prompt, answer)
    print(f"Prompt:  {prompt}")
    print(f"Answer:  {answer[:200]}")
    print(f"Score:   {score.score:.3f}  Approved: {approved}")


def streaming_guard():
    """Stream tokens from Ollama with real-time halt."""
    store = GroundTruthStore()
    store.facts["speed of light"] = "299,792 km/s"
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
    kernel = StreamingKernel(hard_limit=0.35, window_size=5, window_threshold=0.45)

    prompt = "What is the speed of light?"
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": True},
        timeout=60,
        stream=True,
    )

    # Collect streamed tokens
    tokens = []
    for line in resp.iter_lines():
        if line:
            import json

            data = json.loads(line)
            tok = data.get("response", "")
            if tok:
                tokens.append(tok)

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
    if not check_ollama():
        print(f"Ollama not running at {OLLAMA_URL}")
        print("Start with: ollama serve")
    else:
        score_response()
        print("\n---\n")
        streaming_guard()
