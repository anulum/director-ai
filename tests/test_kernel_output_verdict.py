# SPDX-License-Identifier: Apache-2.0
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Director-AI — structured halt monitor output

"""Exercise the synchronous monitor's return contract through its public API."""

from __future__ import annotations

import time
from collections.abc import Iterator

from director_ai.core.kernel import HaltMonitor, HaltOutput


def test_monitor_reports_refusal_and_rearm_without_interpreting_response_text() -> None:
    """Distinguish generated marker text from real hard-floor and latching events."""
    scores: list[float] = []
    monitor = HaltMonitor(on_halt=scores.append)
    marker = "[HALT: COHERENCE BELOW THRESHOLD]"
    permitted = monitor.stream_output_result([marker], lambda _: 0.5)
    assert permitted == HaltOutput(marker, last_score=0.5)
    assert not permitted.halted
    assert monitor.stream_output([marker], lambda _: 0.5) == marker

    stopped = monitor.stream_output_result(["candidate"], lambda _: float("nan"))
    assert stopped == HaltOutput(marker, "hard_limit", 0.0)
    assert stopped.halted and scores == [0.0] and not monitor.is_active

    scored_tokens: list[str] = []

    def score_candidate(token: str) -> float:
        """Record each candidate the monitor admits to scoring."""
        scored_tokens.append(token)
        return 0.5

    inactive = monitor.stream_output_result(["candidate"], score_candidate)
    assert inactive == HaltOutput("[HALT: KERNEL INACTIVE]", "kernel_inactive")
    assert scored_tokens == []
    monitor.reactivate()
    assert monitor.stream_output_result([], score_candidate) == HaltOutput("")
    assert scored_tokens == []

    resumed = monitor.stream_output_result(["candidate"], score_candidate)
    assert resumed == HaltOutput("candidate", last_score=0.5)
    assert scored_tokens == ["candidate"]


def test_monitor_reports_total_timeout_and_keeps_previous_score() -> None:
    """A running stream stops on its total deadline before checking more output."""

    def tokens() -> Iterator[str]:
        """Deliver a prefix and then delay the next token."""
        yield "prefix"
        time.sleep(0.30)
        yield "additional token"

    monitor = HaltMonitor(total_timeout=0.25)
    result = monitor.stream_output_result(tokens(), lambda _: 0.9)
    assert result == HaltOutput("[HALT: TOTAL TIMEOUT EXCEEDED]", "total_timeout", 0.9)
    assert not monitor.is_active


def test_monitor_reports_callback_and_token_timeout_separately() -> None:
    """Distinguish a failed callback from a late score."""

    def interrupted(_: str) -> float:
        """Report that the scoring operation timed out."""
        raise TimeoutError("score worker exceeded its deadline")

    monitor = HaltMonitor()
    error = monitor.stream_output_result(["candidate"], interrupted)
    assert error == HaltOutput("[HALT: CALLBACK TIMEOUT]", "callback_timeout")
    assert not monitor.is_active

    def delayed(_: str) -> float:
        """Return an approved score after the token deadline."""
        time.sleep(0.02)
        return 0.9

    token_monitor = HaltMonitor(token_timeout=0.01)
    timeout = token_monitor.stream_output_result(["candidate"], delayed)
    assert timeout == HaltOutput("[HALT: TOKEN TIMEOUT EXCEEDED]", "token_timeout", 0.9)
    assert not token_monitor.is_active
