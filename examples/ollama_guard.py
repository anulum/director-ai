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
    store.add("boiling point", "100 degrees Celsius")
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


def _ollama_token_stream(prompt: str):
    """Yield tokens from Ollama's streaming endpoint.

    The HTTP connection stays open; the caller can stop iterating to
    close it mid-generation (true mid-stream halt).
    """
    import json

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": True},
        timeout=60,
        stream=True,
    )
    with resp:
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            tok = data.get("response", "")
            if tok:
                yield tok
            if data.get("done"):
                break


def streaming_guard():
    """Stream tokens from Ollama with real-time halt.

    Tokens are scored as they arrive from the network. If the kernel
    fires, iteration stops and the HTTP connection closes — the LLM
    never finishes generating.
    """
    store = GroundTruthStore()
    store.add("speed of light", "299,792 km/s")
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
    kernel = StreamingKernel(hard_limit=0.35, window_size=5, window_threshold=0.45)

    prompt = "What is the speed of light?"
    accumulated = ""

    def coherence_cb(token: str) -> float:
        nonlocal accumulated
        accumulated += token
        _, sc = scorer.review(prompt, accumulated)
        return sc.score

    session = kernel.stream_tokens(
        _ollama_token_stream(prompt),
        coherence_cb,
    )

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
