#!/usr/bin/env python3
"""
One-liner SDK guard demo.

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

Requires: pip install director-ai[openai]
"""

from __future__ import annotations

import os

from director_ai import HallucinationError, guard


def openai_demo():
    from openai import OpenAI

    client = guard(
        OpenAI(),
        facts={"refund": "Refunds are available within 30 days of purchase."},
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is your refund policy?"}],
        )
        print("Approved:", resp.choices[0].message.content)
    except HallucinationError as e:
        print(f"Blocked (coherence={e.score.score:.3f}): {e.response[:80]}")


def anthropic_demo():
    from anthropic import Anthropic

    client = guard(
        Anthropic(),
        facts={"refund": "Refunds are available within 30 days of purchase."},
    )
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": "What is your refund policy?"}],
        )
        print("Approved:", resp.content[0].text)
    except HallucinationError as e:
        print(f"Blocked (coherence={e.score.score:.3f}): {e.response[:80]}")


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        openai_demo()
    elif os.environ.get("ANTHROPIC_API_KEY"):
        anthropic_demo()
    else:
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run this demo.")
        print("See docstring for before/after comparison.")
