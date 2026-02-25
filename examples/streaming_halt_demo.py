#!/usr/bin/env python3
"""
Streaming halt demo -- Director-AI kills a hallucinating response mid-stream.

Run:
    python examples/streaming_halt_demo.py

No GPU, no API keys, no external services required.
Demonstrates all three halt mechanisms: hard_limit, sliding window, downward trend.
"""

from __future__ import annotations

import logging
import sys
import time

from director_ai.core.streaming import StreamingKernel

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def colour_for(score: float) -> str:
    if score >= 0.6:
        return GREEN
    if score >= 0.45:
        return YELLOW
    return RED


def run_scenario(
    label: str,
    tokens: list[str],
    scores: list[float],
    kernel: StreamingKernel,
) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}{label}{RESET}\n")

    idx = 0

    def token_gen():
        yield from tokens

    def coherence_cb(_token: str) -> float:
        nonlocal idx
        s = scores[min(idx, len(scores) - 1)]
        idx += 1
        return s

    session = kernel.stream_tokens(token_gen(), coherence_cb)

    sys.stdout.write(f"  {BOLD}LLM:{RESET} ")
    for event in session.events:
        c = colour_for(event.coherence)
        sys.stdout.write(f"{c}{event.token}{RESET}")
        sys.stdout.flush()
        time.sleep(0.05)
    print()

    if session.halted:
        print(f"\n  {RED}{BOLD}>>> HALTED{RESET}  {DIM}{session.halt_reason}{RESET}")
        n = f"{session.halt_index}/{len(tokens)}"
        print(f"  {DIM}Partial output preserved ({n} tokens){RESET}")
    else:
        print(f"\n  {GREEN}{BOLD}>>> APPROVED{RESET}")

    print(
        f"  {DIM}avg={session.avg_coherence:.3f}  "
        f"min={session.min_coherence:.3f}  "
        f"tokens={session.token_count}/{len(tokens)}{RESET}"
    )


def fresh_kernel() -> StreamingKernel:
    return StreamingKernel(
        hard_limit=0.35,
        window_size=5,
        window_threshold=0.45,
        trend_window=4,
        trend_threshold=0.20,
    )


def main() -> None:
    logging.disable(logging.CRITICAL)  # suppress kernel log noise in demo

    print(f"\n{BOLD}Director-AI -- Streaming Halt Demo{RESET}")
    print(f"{DIM}Token-by-token coherence monitoring with real-time halt{RESET}")

    # 1. Truthful response -- coherence stays high, all checks pass
    run_scenario(
        "1. Truthful response  -->  APPROVED",
        tokens=[
            "Water",
            " boils",
            " at",
            " 100",
            " degrees",
            " Celsius",
            " (212",
            " F)",
            " at",
            " standard",
            " atmospheric",
            " pressure",
            ".",
        ],
        scores=[
            0.92,
            0.90,
            0.88,
            0.91,
            0.93,
            0.90,
            0.88,
            0.87,
            0.89,
            0.91,
            0.90,
            0.89,
            0.90,
        ],
        kernel=fresh_kernel(),
    )

    # 2. Blatant hallucination -- hard_limit fires on a single catastrophic token
    #    Coherence is fine through "... Celsius. But the real", then crashes
    #    when the LLM claims a physically impossible temperature.
    #    hard_limit (Check 1) fires before trend (Check 3) is evaluated.
    run_scenario(
        "2. Blatant hallucination  -->  hard_limit halt",
        tokens=[
            "Water",
            " boils",
            " at",
            " 100",
            " degrees",
            " Celsius",
            ".",
            " But",
            " the",
            " real",
            " temperature",
            " is",
            " negative",
            " forty",
            " degrees",
            ".",
        ],
        scores=[
            0.92,
            0.90,
            0.91,
            0.89,
            0.88,
            0.87,
            0.86,
            0.85,
            0.84,
            0.83,
            0.30,  # hard_limit fires (0.30 < 0.35)
            0.15,
            0.10,
            0.08,
            0.05,
            0.03,
        ],
        kernel=fresh_kernel(),
    )

    # 3. Gradual drift -- downward_trend catches steady coherence decay
    #    The response starts accurate, then drifts into fabrication.
    #    No single token is catastrophic, but the 4-token trend exceeds 0.20.
    run_scenario(
        "3. Gradual drift  -->  downward_trend halt",
        tokens=[
            "Water",
            " boils",
            " at",
            " 100",
            " C.",
            " However",
            " at",
            " high",
            " altitude",
            " it",
            " actually",
            " boils",
            " at",
            " only",
            " 50",
            " C,",
            " which",
            " means",
            " climbers",
            " can",
            " boil",
            " water",
            " with",
            " body",
            " heat",
            " alone",
            ".",
        ],
        scores=[
            0.91,
            0.89,
            0.87,
            0.90,
            0.88,
            0.78,
            0.72,
            0.65,
            0.58,
            0.52,
            0.46,
            0.41,
            0.38,
            0.33,
            0.28,
            0.22,
            0.18,
            0.15,
            0.12,
            0.10,
            0.08,
            0.05,
            0.03,
            0.02,
            0.01,
            0.01,
            0.01,
        ],
        kernel=fresh_kernel(),
    )

    print(f"\n{CYAN}{'=' * 60}{RESET}")
    mechs = "hard_limit | sliding window avg | downward trend"
    print(f"{DIM}Three halt mechanisms: {mechs}{RESET}\n")


if __name__ == "__main__":
    main()
