#!/usr/bin/env python3
"""
Graceful fallback patterns — three ways to handle rejected output.

Director-AI supports three fallback modes when all candidates fail:

1. **Hard halt** (default): Return "[SYSTEM HALT]" message.
2. **Retrieval fallback**: Return ground truth context directly.
3. **Disclaimer fallback**: Prepend warning to best-rejected candidate.

Plus: soft warning zone and streaming on_halt callback.

Usage::

    python examples/graceful_fallbacks.py
"""

from __future__ import annotations

from director_ai import CoherenceAgent, GroundTruthStore


def demo_hard_halt():
    """Default: all candidates rejected → system halts."""
    agent = CoherenceAgent(use_nli=False)
    agent.store.add("capital", "Paris is the capital of France.")

    result = agent.process("What is the capital of France?")
    print("HARD HALT MODE")
    print(f"  Output:   {result.output[:80]}")
    print(f"  Halted:   {result.halted}")
    print(f"  Fallback: {result.fallback_used}")
    print()


def demo_retrieval_fallback():
    """Retrieval: on halt, serve ground truth directly."""
    agent = CoherenceAgent(use_nli=False, fallback="retrieval")
    agent.store.add("capital", "Paris is the capital of France.")

    result = agent.process("What is the capital of France?")
    print("RETRIEVAL FALLBACK MODE")
    print(f"  Output:   {result.output[:80]}")
    print(f"  Halted:   {result.halted}")
    print(f"  Fallback: {result.fallback_used}")
    print()


def demo_disclaimer_fallback():
    """Disclaimer: prepend warning to best rejected candidate."""
    agent = CoherenceAgent(
        use_nli=False,
        fallback="disclaimer",
        disclaimer_prefix="[Unverified] ",
    )
    agent.store.add("capital", "Paris is the capital of France.")

    result = agent.process("What is the capital of France?")
    print("DISCLAIMER FALLBACK MODE")
    print(f"  Output:   {result.output[:80]}")
    print(f"  Halted:   {result.halted}")
    print(f"  Fallback: {result.fallback_used}")
    print()


def demo_streaming_on_halt():
    """Streaming: on_halt callback fires when coherence drops mid-stream."""
    from director_ai import StreamingKernel

    halted_sessions = []

    def on_halt(session):
        halted_sessions.append(session)
        print(
            f"  [on_halt] Halted at token {session.halt_index}: {session.halt_reason}"
        )
        print(f"  [on_halt] Partial output: {session.output[:60]!r}")

    kernel = StreamingKernel(hard_limit=0.3, on_halt=on_halt)

    tokens = [
        "The ",
        "sky ",
        "is ",
        "blue ",
        "and ",
        "also ",
        "green ",
        "and ",
        "purple.",
    ]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    score_iter = iter(scores)

    session = kernel.stream_tokens(
        iter(tokens),
        lambda _tok: next(score_iter),
    )

    print("STREAMING ON_HALT CALLBACK")
    print(f"  Halted:   {session.halted}")
    print(f"  Reason:   {session.halt_reason}")
    print(f"  Tokens:   {session.token_count}")
    print(f"  Output:   {session.output!r}")
    print(f"  Callback fired: {len(halted_sessions)} time(s)")
    print()


def demo_soft_warning():
    """Soft warning: output approved but with confidence warning."""
    from director_ai import CoherenceScorer

    store = GroundTruthStore()
    store.add("sky", "The sky is blue.")

    scorer = CoherenceScorer(
        threshold=0.5,
        soft_limit=0.7,
        ground_truth_store=store,
        use_nli=False,
    )

    approved, score = scorer.review(
        "What color is the sky?",
        "The sky appears blue most of the time.",
    )

    print("SOFT WARNING ZONE")
    print(f"  Approved: {approved}")
    print(f"  Score:    {score.score:.3f}")
    print(f"  Warning:  {score.warning}")
    if score.warning:
        print(
            f"  (Score {score.score:.3f} is between threshold 0.5 and soft_limit 0.7)"
        )
    print()


if __name__ == "__main__":
    demo_hard_halt()
    demo_retrieval_fallback()
    demo_disclaimer_fallback()
    demo_streaming_on_halt()
    demo_soft_warning()
