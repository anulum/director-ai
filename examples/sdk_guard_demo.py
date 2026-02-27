#!/usr/bin/env python3
"""
SDK guard demo — three modes: raise, log, metadata + streaming.

Before (15+ lines of boilerplate):

    store = GroundTruthStore()
    store.add("refund", "within 30 days")
    scorer = CoherenceScorer(threshold=0.6, ground_truth_store=store)
    response = client.chat.completions.create(...)
    answer = response.choices[0].message.content
    approved, score = scorer.review(prompt, answer)
    if not approved:
        raise ...

After (2 lines):

    from director_ai import guard
    client = guard(OpenAI(), facts={"refund": "within 30 days"})
    response = client.chat.completions.create(...)  # auto-scored

Requires: pip install director-ai[openai] or director-ai[anthropic]
"""

from __future__ import annotations

import os

from director_ai import HallucinationError, get_score, guard

FACTS = {"refund": "Refunds are available within 30 days of purchase."}
PROMPT = [{"role": "user", "content": "What is your refund policy?"}]


# ── Mode 1: raise (default) ──────────────────────────────────────────


def openai_raise_demo():
    """on_fail='raise' — blocked responses throw HallucinationError."""
    from openai import OpenAI

    client = guard(OpenAI(), facts=FACTS)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=PROMPT,
        )
        print("[raise] Approved:", resp.choices[0].message.content[:80])
    except HallucinationError as e:
        print(f"[raise] Blocked (coherence={e.score.score:.3f}): {e.response[:80]}")


# ── Mode 2: metadata — score stored in ContextVar ────────────────────


def openai_metadata_demo():
    """on_fail='metadata' — no exception; retrieve score via get_score()."""
    from openai import OpenAI

    client = guard(OpenAI(), facts=FACTS, on_fail="metadata")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=PROMPT,
    )
    score = get_score()
    text = resp.choices[0].message.content or ""
    if score and not score.approved:
        print(f"[metadata] Low confidence ({score.score:.3f}): {text[:60]}")
    else:
        print(f"[metadata] Approved: {text[:80]}")


# ── Mode 3: streaming with periodic checks ───────────────────────────


def openai_streaming_demo():
    """Streaming: coherence checked every 8 tokens + final check."""
    from openai import OpenAI

    client = guard(OpenAI(), facts=FACTS, on_fail="log")
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=PROMPT,
            stream=True,
        )
        tokens = []
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                tokens.append(delta)
                print(delta, end="", flush=True)
        print(f"\n[stream] Complete ({len(tokens)} chunks)")
    except HallucinationError as e:
        print(f"\n[stream] Halted mid-stream (coherence={e.score.score:.3f})")


# ── Anthropic variant ─────────────────────────────────────────────────


def anthropic_demo():
    from anthropic import Anthropic

    client = guard(Anthropic(), facts=FACTS)
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=PROMPT,
        )
        print("[anthropic] Approved:", resp.content[0].text[:80])
    except HallucinationError as e:
        print(f"[anthropic] Blocked (coherence={e.score.score:.3f}): {e.response[:80]}")


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        print("=== OpenAI: raise mode ===")
        openai_raise_demo()
        print("\n=== OpenAI: metadata mode ===")
        openai_metadata_demo()
        print("\n=== OpenAI: streaming ===")
        openai_streaming_demo()
    elif os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_demo()
    else:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this demo.")
        print("See docstring for before/after comparison.")
