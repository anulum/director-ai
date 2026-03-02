#!/usr/bin/env python3
"""
Self-contained streaming halt demo with colored terminal output.

Run:
    python examples/streaming_demo.py

No GPU, no API keys, no dependencies beyond director-ai.
"""

from __future__ import annotations

import sys
import time

from director_ai.core.streaming import StreamingKernel

# ANSI escape codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    if score >= 0.6:
        colour = GREEN
    elif score >= 0.45:
        colour = YELLOW
    else:
        colour = RED
    return f"{colour}{'#' * filled}{'.' * (width - filled)}{RESET} {score:.2f}"


def _run(
    label: str,
    tokens: list[str],
    scores: list[float],
) -> None:
    print(f"\n{BOLD}{CYAN}-- {label} {'-' * (56 - len(label))}{RESET}\n")

    kernel = StreamingKernel(
        hard_limit=0.35,
        window_size=5,
        window_threshold=0.45,
        trend_window=4,
        trend_threshold=0.20,
    )

    idx = 0

    def coherence_cb(_tok: str) -> float:
        nonlocal idx
        s = scores[min(idx, len(scores) - 1)]
        idx += 1
        return s

    session = kernel.stream_tokens(iter(tokens), coherence_cb)

    sys.stdout.write(f"  {BOLD}LLM:{RESET} ")
    for ev in session.events:
        if ev.coherence >= 0.6:
            c = GREEN
        elif ev.coherence >= 0.45:
            c = YELLOW
        else:
            c = RED
        sys.stdout.write(f"{c}{ev.token}{RESET}")
        sys.stdout.flush()
        time.sleep(0.04)

    if session.halted:
        sys.stdout.write(f" {RED}{BOLD}[HALT]{RESET}")
    print()

    # Score trace
    print(f"\n  {DIM}Score trace:{RESET}")
    for ev in session.events:
        marker = f" {RED}< HALT{RESET}" if ev.halted else ""
        print(f"  {DIM}[{ev.index:2d}]{RESET} {_bar(ev.coherence)}{marker}")

    if session.halted:
        print(
            f"\n  {RED}{BOLD}HALTED{RESET} at token {session.halt_index}/{len(tokens)}"
            f"  {DIM}reason: {session.halt_reason}{RESET}"
        )
    else:
        n = session.token_count
        print(f"\n  {GREEN}{BOLD}APPROVED{RESET} -- all {n} tokens passed")

    print(
        f"  {DIM}avg={session.avg_coherence:.3f}  "
        f"min={session.min_coherence:.3f}{RESET}"
    )


def main() -> None:
    print(f"\n{BOLD}Director-AI -- Streaming Halt Demo{RESET}")
    print(f"{DIM}Token-by-token coherence monitoring with real-time halt{RESET}")
    print(f"{DIM}Three mechanisms: hard_limit | sliding window | downward trend{RESET}")

    _run(
        "Truthful response -> APPROVED",
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
    )

    _run(
        "Blatant hallucination -> hard_limit halt",
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
            0.30,
            0.15,
            0.10,
            0.08,
            0.05,
            0.03,
        ],
    )

    _run(
        "Gradual drift -> trend halt",
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
        ],
    )

    print(f"\n{CYAN}{'-' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
