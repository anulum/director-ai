# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — OpenAI TTS Voice AI Example

"""Guardrailed voice output with OpenAI TTS.

Streams an LLM response through VoiceGuard, collects only approved
text, then sends it to OpenAI's TTS API for speech synthesis.
Saves the audio to a WAV file.

Requirements::

    pip install director-ai[nli] openai

Set environment variables::

    OPENAI_API_KEY=sk-...

Usage::

    python examples/voice_openai_tts.py
    python examples/voice_openai_tts.py --output response.mp3
"""

from __future__ import annotations

import sys
from pathlib import Path

PRODUCT_FACTS = {
    "refund": "Full refund within 30 days of purchase, no questions asked.",
    "warranty": "12-month warranty covering manufacturing defects only.",
    "shipping": "Free shipping on orders over $50. Standard delivery 3-5 business days.",
    "returns": "Return requests must be submitted through the customer portal.",
}

USER_QUESTION = "What is the refund and warranty policy?"


def main(argv: list[str] | None = None):
    args = argv or sys.argv[1:]
    output_path = Path("guardrailed_response.mp3")
    if "--output" in args:
        idx = args.index("--output")
        if idx + 1 < len(args):
            output_path = Path(args[idx + 1])

    from openai import OpenAI

    from director_ai import VoiceGuard

    client = OpenAI()

    guard = VoiceGuard(
        facts=PRODUCT_FACTS,
        prompt=USER_QUESTION,
        threshold=0.3,
        hard_limit=0.25,
        score_every=4,
        soft_halt=True,
        recovery="I need to verify that information before I can confirm.",
    )

    print(f"User: {USER_QUESTION}")
    print("Generating response with guardrail...")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a customer support agent. Answer based on company policy.",
            },
            {"role": "user", "content": USER_QUESTION},
        ],
        stream=True,
    )

    approved_tokens = []
    stats = {"total_tokens": 0, "scored": 0, "halted": False}

    for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if not token:
            continue

        stats["total_tokens"] += 1
        result = guard.feed(token)

        if result.coherence > 0:
            stats["scored"] += 1

        if result.halted:
            approved_tokens.append(f" {result.recovery_text}")
            stats["halted"] = True
            print(f"\n[HALT] {result.halt_reason} (coherence={result.coherence:.3f})")
            break
        approved_tokens.append(token)

    final_text = "".join(approved_tokens)
    print(f"\nAssistant: {final_text}")
    print(
        f"\nTokens: {stats['total_tokens']} generated, "
        f"{stats['scored']} scored, halted={stats['halted']}"
    )

    # Send approved text to OpenAI TTS
    print(f"\nSynthesising speech -> {output_path}")
    tts_response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=final_text,
    )
    tts_response.stream_to_file(str(output_path))
    print(f"Audio saved: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
